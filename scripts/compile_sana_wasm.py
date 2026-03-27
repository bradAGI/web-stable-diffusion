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
        # Emscripten deps
        "python3", "nodejs", "npm",
    )
    .pip_install(
        "mlc-ai-nightly",
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

    # --- Trace DC-AE decoder ---
    print("Tracing DC-AE decoder...")
    vae = pipe.vae
    vae.eval()

    # Sana uses 32x compression, so 1024x1024 → 32x32 latent
    latent_h, latent_w = 32, 32
    latent_channels = pipe.transformer.config.in_channels

    dummy_latent = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.float16, device="cuda")

    # Try tracing with torch.export or fx
    try:
        from torch.export import export
        from tvm.relax.frontend.torch import from_exported_program

        print("  Using torch.export path...")
        with torch.no_grad():
            exported_vae = export(vae.decode, (dummy_latent,))
        vae_mod = from_exported_program(exported_vae, keep_params_as_input=True)
        print("  DC-AE decoder traced successfully")
    except Exception as e:
        print(f"  torch.export failed ({e}), trying FX trace...")
        import torch.fx
        with torch.no_grad():
            traced_vae = torch.fx.symbolic_trace(vae.decode)
        vae_mod = from_fx(traced_vae, [(1, latent_channels, latent_h, latent_w)], keep_params_as_input=True)
        print("  DC-AE decoder traced via FX")

    # --- Trace Transformer (Linear DiT) ---
    print("Tracing Linear DiT transformer...")
    transformer = pipe.transformer
    transformer.eval()

    # Sana transformer inputs: hidden_states, encoder_hidden_states, timestep
    dummy_hidden = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.float16, device="cuda")
    dummy_encoder_hidden = torch.randn(1, 300, transformer.config.cross_attention_dim, dtype=torch.float16, device="cuda")
    dummy_timestep = torch.tensor([500.0], dtype=torch.float16, device="cuda")

    try:
        from torch.export import export
        from tvm.relax.frontend.torch import from_exported_program

        print("  Using torch.export path...")
        with torch.no_grad():
            exported_dit = export(
                transformer,
                (dummy_hidden, dummy_encoder_hidden, dummy_timestep),
            )
        dit_mod = from_exported_program(exported_dit, keep_params_as_input=True)
        print("  Transformer traced successfully")
    except Exception as e:
        print(f"  torch.export failed ({e}), trying manual trace...")
        # Fall back to manual tracing
        print("  Attempting fx trace...")
        try:
            import torch.fx
            with torch.no_grad():
                traced_dit = torch.fx.symbolic_trace(transformer)
            dit_mod = from_fx(
                traced_dit,
                [
                    (1, latent_channels, latent_h, latent_w),
                    (1, 300, transformer.config.cross_attention_dim),
                    (1,),
                ],
                keep_params_as_input=True,
            )
            print("  Transformer traced via FX")
        except Exception as e2:
            print(f"  FX trace also failed: {e2}")
            print("  Saving model state dicts for manual compilation...")
            torch.save(transformer.state_dict(), f"{OUTPUT_DIR}/sana-wasm/transformer_state_dict.pt")
            torch.save(vae.state_dict(), f"{OUTPUT_DIR}/sana-wasm/vae_state_dict.pt")
            dit_mod = None

    # ----------------------------------------------------------------
    print("=" * 60)
    print("Step 3: Compiling to WebGPU target")
    print("=" * 60)

    target = tvm.target.Target("webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm")

    modules_to_compile = {}
    if vae_mod is not None:
        modules_to_compile["vae"] = vae_mod
    if dit_mod is not None:
        modules_to_compile["dit"] = dit_mod

    for name, mod in modules_to_compile.items():
        print(f"Compiling {name}...")

        # Apply optimizations
        with tvm.transform.PassContext(opt_level=3):
            mod = relax.transform.DecomposeOpsForInference()(mod)
            mod = relax.transform.LegalizeOps()(mod)

        # MetaSchedule auto-tuning
        print(f"  Auto-tuning {name} for WebGPU...")
        try:
            from tvm import meta_schedule as ms

            with ms.database.JSONDatabase(
                f"{OUTPUT_DIR}/sana-wasm/{name}_tuning_db.json"
            ) as db:
                with target:
                    mod = relax.transform.MetaScheduleApplyDatabase(db)(mod)
        except Exception as e:
            print(f"  Auto-tuning skipped: {e}")

        # Build
        print(f"  Building {name} WASM...")
        try:
            # Activate emscripten
            os.environ["EMSDK"] = "/opt/emsdk"
            subprocess.run(
                ["bash", "-c", "source /opt/emsdk/emsdk_env.sh"],
                check=True, capture_output=True,
            )

            lib = relax.build(mod, target=target)
            wasm_path = f"{OUTPUT_DIR}/sana-wasm/sana_{name}.wasm"
            lib.export_library(wasm_path)
            print(f"  ✓ {name} compiled to {wasm_path}")
            print(f"    Size: {os.path.getsize(wasm_path) / 1e6:.1f} MB")
        except Exception as e:
            print(f"  ✗ Build failed: {e}")
            # Save IR for debugging
            with open(f"{OUTPUT_DIR}/sana-wasm/{name}_ir.txt", "w") as f:
                f.write(str(mod))
            print(f"  Saved IR to {name}_ir.txt for debugging")

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

    config = {
        "model": "Sana-0.6B",
        "resolution": 1024,
        "latent_size": 32,
        "compression_ratio": 32,
        "components": {
            "transformer": {
                "wasm": "sana_dit.wasm",
                "params": "weight-shards/",
            },
            "vae_decoder": {
                "wasm": "sana_vae.wasm",
                "params": "weight-shards/",
            },
        },
        "text_encoder": "google/gemma-2b",
        "scheduler": {
            "type": "flow-dpm-solver",
            "num_steps": 20,
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
