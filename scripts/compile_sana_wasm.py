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
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
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
        "numpy<2.0",
        "Pillow",
        "onnx",
        "onnxscript",
    )
    .run_commands(
        # Install MLC AI nightly (CUDA 12.4) from direct URL
        "pip install https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_ai_nightly_cu124-0.24.dev0-py3-none-manylinux_2_28_x86_64.whl",
        # Debug: find what the package installs
        "pip show mlc-ai-nightly-cu124 && python -c 'import importlib; import pkgutil; [print(m.name) for m in pkgutil.iter_modules() if \"tvm\" in m.name or \"mlc\" in m.name]'",
        # Install Emscripten
        "git clone https://github.com/emscripten-core/emsdk.git /opt/emsdk",
        "cd /opt/emsdk && ./emsdk install latest && ./emsdk activate latest",
    )
    .env({
        "EMSDK": "/opt/emsdk",
        "PATH": "/opt/emsdk:/opt/emsdk/upstream/emscripten:$PATH",
    })
)

SANA_MODEL_ID = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
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

    # ----------------------------------------------------------------
    # Export text encoder (Gemma) — shared across all resolutions
    # ----------------------------------------------------------------
    print("Exporting Gemma text encoder to ONNX...")
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    tokenizer = pipe.tokenizer

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

    text_enc_wrapper = TextEncoderWrapper(text_encoder).cuda().half().eval()

    # Sana uses max 300 tokens
    max_seq_len = 300
    dummy_input_ids = torch.ones(1, max_seq_len, dtype=torch.long, device="cuda")
    dummy_attention_mask = torch.ones(1, max_seq_len, dtype=torch.long, device="cuda")

    text_enc_onnx_path = f"{OUTPUT_DIR}/sana-wasm/sana_text_encoder.onnx"
    try:
        with torch.no_grad():
            torch.onnx.export(
                text_enc_wrapper,
                (dummy_input_ids, dummy_attention_mask),
                text_enc_onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["hidden_states"],
                dynamic_axes={
                    "input_ids": {1: "seq_len"},
                    "attention_mask": {1: "seq_len"},
                    "hidden_states": {1: "seq_len"},
                },
                opset_version=18,
            )
        enc_size = os.path.getsize(text_enc_onnx_path) / 1e6
        data_path = text_enc_onnx_path + ".data"
        if os.path.exists(data_path):
            enc_size += os.path.getsize(data_path) / 1e6
        print(f"  ✓ Text encoder exported: {text_enc_onnx_path} ({enc_size:.0f} MB)")
    except Exception as e:
        print(f"  ✗ Text encoder export failed: {e}")

    # Save tokenizer config for browser use
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/sana-wasm/tokenizer")
    print("  ✓ Tokenizer saved")

    vae = pipe.vae
    vae.eval()
    transformer = pipe.transformer
    transformer.eval()
    latent_channels = pipe.transformer.config.in_channels

    # New TVM requires JSON dict format for targets
    target = tvm.target.Target(
        {"kind": "webgpu"},
        host={"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"},
    )

    # Activate emscripten once
    os.environ["EMSDK"] = "/opt/emsdk"
    subprocess.run(["bash", "-c", "source /opt/emsdk/emsdk_env.sh"], check=True, capture_output=True)

    # Wrap VAE decoder as a proper nn.Module for tracing
    class VAEDecodeWrapper(torch.nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            # Convert entire VAE to float32 to avoid mixed-precision tracing issues
            self.vae = vae_model.float()

        def forward(self, latent):
            return self.vae.decode(latent).sample

    # Wrap transformer for tracing (disable control flow)
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, dit_model):
            super().__init__()
            self.dit = dit_model

        def forward(self, hidden_states, encoder_hidden_states, timestep):
            return self.dit(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                return_dict=False,
            )[0]

    vae_wrapper = VAEDecodeWrapper(vae).cuda().eval()  # float32 for VAE
    dit_wrapper = TransformerWrapper(transformer).cuda().half().eval()

    # Get the correct text encoder hidden size from model config
    # Sana uses Gemma text encoder — caption_projection expects its hidden dim
    text_hidden_size = transformer.config.caption_channels
    print(f"  Text encoder hidden size (caption_channels): {text_hidden_size}")
    print(f"  Cross attention dim: {transformer.config.cross_attention_dim}")

    for variant in RESOLUTION_VARIANTS:
        res_name = variant["name"]
        latent_h = latent_w = variant["latent_size"]

        print("=" * 60)
        print(f"Compiling {res_name}x{res_name} variant (latent {latent_h}x{latent_w})")
        print("=" * 60)

        variant_dir = f"{OUTPUT_DIR}/sana-wasm/{res_name}"
        os.makedirs(variant_dir, exist_ok=True)

        # --- Export DC-AE decoder to ONNX ---
        print(f"  Exporting DC-AE decoder for {res_name}...")
        dummy_latent = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.float32, device="cuda")
        vae_mod = None
        onnx_vae_path = f"{variant_dir}/sana_vae_{res_name}.onnx"
        try:
            # ONNX export directly — most reliable for complex models
            with torch.no_grad():
                torch.onnx.export(
                    vae_wrapper, (dummy_latent,), onnx_vae_path,
                    input_names=["latent"], output_names=["image"],
                    dynamic_axes=None, opset_version=18,
                )
            onnx_size = os.path.getsize(onnx_vae_path) / 1e6
            # Check for external data file
            data_path = onnx_vae_path + ".data"
            if os.path.exists(data_path):
                onnx_size += os.path.getsize(data_path) / 1e6
            print(f"  ✓ VAE exported to ONNX: {onnx_vae_path} ({onnx_size:.1f} MB)")
        except Exception as e:
            print(f"  ONNX export failed ({e}), trying JIT trace + ONNX...")
            try:
                with torch.no_grad():
                    traced_vae = torch.jit.trace(vae_wrapper, (dummy_latent,))
                torch.onnx.export(
                    traced_vae, (dummy_latent,), onnx_vae_path,
                    input_names=["latent"], output_names=["image"],
                    dynamic_axes=None, opset_version=18,
                )
                print(f"  ✓ VAE exported via JIT trace → ONNX: {onnx_vae_path}")
            except Exception as e2:
                print(f"  ✗ VAE export failed for {res_name}: {e2}")
                # Save TVM IR for debugging
                try:
                    from tvm.relax.frontend.torch import from_fx
                    import torch.fx
                    with torch.no_grad():
                        traced = torch.jit.trace(vae_wrapper, (dummy_latent,))
                    fx_mod = torch.fx.symbolic_trace(traced)
                    from tvm.relax.frontend.torch import from_exported_program
                    from torch.export import export
                    exported = export(vae_wrapper, (dummy_latent,))
                    vae_mod = from_exported_program(exported, keep_params_as_input=True)
                    with open(f"{variant_dir}/vae_ir.txt", "w") as f:
                        f.write(str(vae_mod))
                except Exception:
                    pass

        # --- Trace Transformer ---
        print(f"  Tracing Linear DiT for {res_name}...")
        dummy_hidden = torch.randn(1, latent_channels, latent_h, latent_w, dtype=torch.float16, device="cuda")
        # caption_projection.linear_1 expects (batch, seq, caption_channels)
        max_seq_len = 300  # Sana uses max 300 text tokens
        dummy_encoder_hidden = torch.randn(1, max_seq_len, text_hidden_size, dtype=torch.float16, device="cuda")
        dummy_timestep = torch.tensor([500.0], dtype=torch.float16, device="cuda")
        dit_mod = None
        try:
            with torch.no_grad():
                traced_dit = torch.jit.trace(dit_wrapper, (dummy_hidden, dummy_encoder_hidden, dummy_timestep))
            from tvm.relax.frontend.torch import from_fx
            import torch.fx
            fx_dit = torch.fx.symbolic_trace(traced_dit)
            dit_mod = from_fx(
                fx_dit,
                [(1, latent_channels, latent_h, latent_w), (1, max_seq_len, text_hidden_size), (1,)],
                keep_params_as_input=True,
            )
            print(f"  ✓ Transformer traced for {res_name}")
        except Exception as e:
            print(f"  JIT+FX trace failed ({e}), trying torch.export...")
            try:
                from torch.export import export
                from tvm.relax.frontend.torch import from_exported_program
                with torch.no_grad():
                    exported_dit = export(dit_wrapper, (dummy_hidden, dummy_encoder_hidden, dummy_timestep))
                dit_mod = from_exported_program(exported_dit, keep_params_as_input=True)
                print(f"  ✓ Transformer traced via torch.export for {res_name}")
            except Exception as e2:
                print(f"  ✗ Transformer trace failed for {res_name}: {e2}")
                # Save ONNX as fallback
                try:
                    onnx_path = f"{variant_dir}/sana_dit_{res_name}.onnx"
                    torch.onnx.export(
                        dit_wrapper, (dummy_hidden, dummy_encoder_hidden, dummy_timestep), onnx_path,
                        input_names=["hidden_states", "encoder_hidden_states", "timestep"],
                        output_names=["output"],
                        dynamic_axes=None, opset_version=18,
                    )
                    print(f"  ✓ Transformer exported to ONNX: {onnx_path}")
                except Exception as e3:
                    print(f"  ✗ ONNX export also failed: {e3}")

        # --- Compile each TVM module ---
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


@app.function(
    image=compiler_image,
    gpu="A100",
    timeout=3600,
    volumes={OUTPUT_DIR: vol},
    memory=32768,
)
def export_text_encoder_only():
    """Export only the Gemma text encoder + tokenizer (skip DiT/VAE)."""
    import os
    import torch

    print("=" * 60)
    print("Exporting Gemma text encoder only")
    print("=" * 60)

    from diffusers import SanaPipeline

    pipe = SanaPipeline.from_pretrained(
        SANA_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")

    text_encoder = pipe.text_encoder
    text_encoder.eval()
    tokenizer = pipe.tokenizer

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

    text_enc_wrapper = TextEncoderWrapper(text_encoder).cuda().half().eval()

    max_seq_len = 300
    dummy_input_ids = torch.ones(1, max_seq_len, dtype=torch.long, device="cuda")
    dummy_attention_mask = torch.ones(1, max_seq_len, dtype=torch.long, device="cuda")

    os.makedirs(f"{OUTPUT_DIR}/sana-wasm", exist_ok=True)
    text_enc_onnx_path = f"{OUTPUT_DIR}/sana-wasm/sana_text_encoder.onnx"

    print("Exporting text encoder to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            text_enc_wrapper,
            (dummy_input_ids, dummy_attention_mask),
            text_enc_onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {1: "seq_len"},
                "attention_mask": {1: "seq_len"},
                "hidden_states": {1: "seq_len"},
            },
            opset_version=18,
        )
    enc_size = os.path.getsize(text_enc_onnx_path) / 1e6
    data_path = text_enc_onnx_path + ".data"
    if os.path.exists(data_path):
        enc_size += os.path.getsize(data_path) / 1e6
    print(f"  ✓ Text encoder: {enc_size:.0f} MB")

    tokenizer.save_pretrained(f"{OUTPUT_DIR}/sana-wasm/tokenizer")
    print("  ✓ Tokenizer saved")

    vol.commit()
    print(f"\n✓ Text encoder saved to volume")
    print("Download: modal volume get sana-wasm-artifacts /artifacts/sana-wasm/sana_text_encoder.onnx* .")


@app.local_entrypoint()
def main():
    # Text encoder only — DiT/VAE already exported
    print("Exporting Gemma text encoder only...")
    export_text_encoder_only.remote()
    print("\nDone! Download with:")
    print("  modal volume get sana-wasm-artifacts /artifacts/sana-wasm ./sana-wasm")
