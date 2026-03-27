"""Compile Sana 0.6B to WebGPU WASM using Modal.

Usage:
    modal run scripts/compile_sana_wasm.py

This will:
1. Download Sana 0.6B from HuggingFace
2. Trace the model into TVM Relax IR
3. Compile to WebGPU WASM + WGSL shaders
4. Export weight shards for browser loading
5. Upload artifacts to a Modal volume (download after)

Requires: modal, HuggingFace token (for gated models)
"""

import modal

# Modal setup
app = modal.App("sana-wasm-compiler")

# Persistent volume to store compiled artifacts
vol = modal.Volume.from_name("sana-wasm-artifacts", create_if_missing=True)

# Image with all dependencies
compiler_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "cmake", "ninja-build", "build-essential",
        "wget", "curl", "unzip",
        "nodejs", "npm",
    )
    .pip_install(
        "torch>=2.1.0",
        "torchvision",
        "transformers>=4.38.0",
        "diffusers>=0.27.0",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "numpy",
        "Pillow",
    )
    .run_commands(
        # Install MLC AI / TVM Unity — CUDA 12.4 wheel for A100
        "pip install mlc-ai-nightly-cu124 -f https://mlc.ai/wheels",
        # Install Emscripten
        "git clone https://github.com/emscripten-core/emsdk.git /opt/emsdk",
        "cd /opt/emsdk && ./emsdk install latest && ./emsdk activate latest",
    )
    .env({
        "EMSDK": "/opt/emsdk",
        "PATH": "/opt/emsdk:/opt/emsdk/upstream/emscripten:$PATH",
    })
)

SANA_MODEL_ID = "Efficient-Large-Model/Sana_600M_1024px"
OUTPUT_DIR = "/artifacts"

# Compile variants for different GPU capabilities
# Sana uses 32x compression, so latent = resolution / 32
RESOLUTION_VARIANTS = [
    {"name": "1024", "resolution": 1024, "latent_size": 32},
    {"name": "2048", "resolution": 2048, "latent_size": 64},
    {"name": "4096", "resolution": 4096, "latent_size": 128},
]


@app.function(
    image=compiler_image,
    gpu="A100",
    timeout=7200,  # 2 hours
    volumes={OUTPUT_DIR: vol},
    memory=32768,
)
def compile_sana_to_wasm():
    """Compile Sana 0.6B to WebGPU WASM."""
    import os
    import json
    import shutil
    import subprocess

    import numpy as np
    import torch

    print("=" * 60)
    print("Step 1: Loading Sana 0.6B model")
    print("=" * 60)

    from diffusers import SanaPipeline

    pipe = SanaPipeline.from_pretrained(
        SANA_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")

    print(f"Model loaded: {SANA_MODEL_ID}")
    print(f"  Transformer params: {sum(p.numel() for p in pipe.transformer.parameters()) / 1e6:.1f}M")
    print(f"  VAE params: {sum(p.numel() for p in pipe.vae.parameters()) / 1e6:.1f}M")
    print(f"  Text encoder params: {sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e6:.1f}M")

    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 2: Tracing model components with TorchDynamo → TVM")
    print("=" * 60)

    import tvm
    from tvm import relax
    from tvm.relax.frontend.torch import from_fx

    # We need to trace 3 components:
    # 1. Text encoder (Gemma-based)
    # 2. Transformer (Linear DiT)
    # 3. DC-AE decoder (32x decompression)

    os.makedirs(f"{OUTPUT_DIR}/sana-wasm", exist_ok=True)

    vae = pipe.vae
    vae.eval()
    transformer = pipe.transformer
    transformer.eval()
    latent_channels = pipe.transformer.config.in_channels

    target = tvm.target.Target("webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm")

    # Activate emscripten once
    os.environ["EMSDK"] = "/opt/emsdk"
    subprocess.run(["bash", "-c", "source /opt/emsdk/emsdk_env.sh"], check=True, capture_output=True)

    for variant in RESOLUTION_VARIANTS:
        res_name = variant["name"]
        latent_h = latent_w = variant["latent_size"]

        print("=" * 60)
        print(f"Compiling {res_name}x{res_name} variant (latent {latent_h}x{latent_w})")
        print("=" * 60)

        variant_dir = f"{OUTPUT_DIR}/sana-wasm/{res_name}"
        os.makedirs(variant_dir, exist_ok=True)

        # --- Trace DC-AE decoder ---
        print(f"  Tracing DC-AE decoder for {res_name}...")
        dummy_latent = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.float16, device="cuda")
        vae_mod = None
        try:
            from torch.export import export
            from tvm.relax.frontend.torch import from_exported_program
            with torch.no_grad():
                exported_vae = export(vae.decode, (dummy_latent,))
            vae_mod = from_exported_program(exported_vae, keep_params_as_input=True)
            print(f"  ✓ DC-AE decoder traced for {res_name}")
        except Exception as e:
            print(f"  torch.export failed ({e}), trying FX...")
            try:
                import torch.fx
                with torch.no_grad():
                    traced_vae = torch.fx.symbolic_trace(vae.decode)
                vae_mod = from_fx(traced_vae, [(1, latent_channels, latent_h, latent_w)], keep_params_as_input=True)
                print(f"  ✓ DC-AE decoder traced via FX for {res_name}")
            except Exception as e2:
                print(f"  ✗ VAE trace failed for {res_name}: {e2}")

        # --- Trace Transformer ---
        print(f"  Tracing Linear DiT for {res_name}...")
        dummy_hidden = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.float16, device="cuda")
        dummy_encoder_hidden = torch.randn(1, 300, transformer.config.cross_attention_dim, dtype=torch.float16, device="cuda")
        dummy_timestep = torch.tensor([500.0], dtype=torch.float16, device="cuda")
        dit_mod = None
        try:
            from torch.export import export
            from tvm.relax.frontend.torch import from_exported_program
            with torch.no_grad():
                exported_dit = export(transformer, (dummy_hidden, dummy_encoder_hidden, dummy_timestep))
            dit_mod = from_exported_program(exported_dit, keep_params_as_input=True)
            print(f"  ✓ Transformer traced for {res_name}")
        except Exception as e:
            print(f"  torch.export failed ({e}), trying FX...")
            try:
                import torch.fx
                with torch.no_grad():
                    traced_dit = torch.fx.symbolic_trace(transformer)
                dit_mod = from_fx(
                    traced_dit,
                    [(1, latent_channels, latent_h, latent_w), (1, 300, transformer.config.cross_attention_dim), (1,)],
                    keep_params_as_input=True,
                )
                print(f"  ✓ Transformer traced via FX for {res_name}")
            except Exception as e2:
                print(f"  ✗ Transformer trace failed for {res_name}: {e2}")

        # --- Compile each module ---
        for comp_name, mod in [("vae", vae_mod), ("dit", dit_mod)]:
            if mod is None:
                continue
            print(f"  Compiling {comp_name} for {res_name}...")
            try:
                with tvm.transform.PassContext(opt_level=3):
                    mod = relax.transform.DecomposeOpsForInference()(mod)
                    mod = relax.transform.LegalizeOps()(mod)

                lib = relax.build(mod, target=target)
                wasm_path = f"{variant_dir}/sana_{comp_name}_{res_name}.wasm"
                lib.export_library(wasm_path)
                print(f"  ✓ {comp_name} → {wasm_path} ({os.path.getsize(wasm_path) / 1e6:.1f} MB)")
            except Exception as e:
                print(f"  ✗ {comp_name} build failed for {res_name}: {e}")
                with open(f"{variant_dir}/{comp_name}_ir.txt", "w") as f:
                    f.write(str(mod))

        # Free GPU memory between variants
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 4: Exporting weight shards")
    print("=" * 60)

    shard_dir = f"{OUTPUT_DIR}/sana-wasm/weight-shards"
    os.makedirs(shard_dir, exist_ok=True)

    SHARD_SIZE = 32 * 1024 * 1024  # 32MB per shard

    def export_weights(state_dict, prefix, shard_size=SHARD_SIZE):
        """Export model weights as sharded binary files with an NDArray cache manifest."""
        params = []
        current_shard = bytearray()
        shard_idx = 0
        records = []

        for name, tensor in state_dict.items():
            arr = tensor.detach().cpu().half().numpy()
            raw = arr.tobytes()

            if len(current_shard) + len(raw) > shard_size and len(current_shard) > 0:
                shard_name = f"params_shard_{shard_idx}.bin"
                with open(f"{shard_dir}/{prefix}_{shard_name}", "wb") as f:
                    f.write(current_shard)
                shard_idx += 1
                current_shard = bytearray()

            records.append({
                "name": f"{prefix}.{name}",
                "shape": list(arr.shape),
                "dtype": "float16",
                "byteOffset": len(current_shard),
                "dataPath": f"{prefix}_params_shard_{shard_idx}.bin",
            })
            current_shard.extend(raw)

        # Write remaining shard
        if current_shard:
            shard_name = f"params_shard_{shard_idx}.bin"
            with open(f"{shard_dir}/{prefix}_{shard_name}", "wb") as f:
                f.write(current_shard)

        return records

    print("Exporting transformer weights...")
    dit_records = export_weights(pipe.transformer.state_dict(), "transformer")
    print(f"  {len(dit_records)} tensors exported")

    print("Exporting VAE weights...")
    vae_records = export_weights(pipe.vae.state_dict(), "vae")
    print(f"  {len(vae_records)} tensors exported")

    # Write NDArray cache manifest
    manifest = {
        "records": dit_records + vae_records,
        "model": SANA_MODEL_ID,
        "format": "float16",
    }
    with open(f"{shard_dir}/ndarray-cache.json", "w") as f:
        json.dump(manifest, f)
    print(f"Manifest written with {len(manifest['records'])} records")

    # Calculate total size
    total_size = sum(
        os.path.getsize(os.path.join(shard_dir, f))
        for f in os.listdir(shard_dir)
        if f.endswith(".bin")
    )
    print(f"Total weight shards: {total_size / 1e9:.2f} GB")

    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 5: Writing browser config")
    print("=" * 60)

    variants_config = {}
    for variant in RESOLUTION_VARIANTS:
        res = variant["name"]
        variants_config[res] = {
            "resolution": variant["resolution"],
            "latent_size": variant["latent_size"],
            "transformer_wasm": f"{res}/sana_dit_{res}.wasm",
            "vae_wasm": f"{res}/sana_vae_{res}.wasm",
            "min_vram_gb": {
                "1024": 3, "2048": 6, "4096": 16,
            }.get(res, 3),
        }

    config = {
        "model": "Sana-0.6B",
        "compression_ratio": 32,
        "default_variant": "1024",
        "variants": variants_config,
        "params": "weight-shards/",
        "text_encoder": "google/gemma-2b",
        "scheduler": {
            "type": "flow-dpm-solver",
            "num_steps": 20,
        },
        "vram_auto_select": {
            "description": "Browser auto-selects variant based on GPU VRAM",
            "thresholds": [
                {"min_vram_gb": 16, "variant": "4096"},
                {"min_vram_gb": 6, "variant": "2048"},
                {"min_vram_gb": 0, "variant": "1024"},
            ],
        },
    }
    with open(f"{OUTPUT_DIR}/sana-wasm/sana-config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 6: Summary")
    print("=" * 60)

    artifacts = os.listdir(f"{OUTPUT_DIR}/sana-wasm")
    for a in sorted(artifacts):
        path = os.path.join(f"{OUTPUT_DIR}/sana-wasm", a)
        if os.path.isfile(path):
            print(f"  {a}: {os.path.getsize(path) / 1e6:.1f} MB")
        else:
            sub_files = os.listdir(path)
            sub_size = sum(os.path.getsize(os.path.join(path, f)) for f in sub_files)
            print(f"  {a}/: {len(sub_files)} files, {sub_size / 1e6:.1f} MB total")

    vol.commit()
    print("\n✓ All artifacts saved to Modal volume 'sana-wasm-artifacts'")
    print("Download with: modal volume get sana-wasm-artifacts /artifacts/sana-wasm ./sana-wasm")


@app.local_entrypoint()
def main():
    print("Starting Sana 0.6B WASM compilation on Modal A100...")
    print("This will take ~30-60 minutes.")
    compile_sana_to_wasm.remote()
    print("\nDone! Download artifacts with:")
    print("  modal volume get sana-wasm-artifacts /artifacts/sana-wasm ./sana-wasm")
