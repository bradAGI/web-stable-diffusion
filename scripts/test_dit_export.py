"""Quick Modal test: verify DiT ONNX export produces non-zero output."""
import modal

app = modal.App("sana-dit-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch>=2.1.0", "diffusers>=0.27.0", "transformers>=4.38.0", "accelerate", "safetensors", "sentencepiece", "protobuf", "numpy<2.0", "Pillow", "onnx", "onnxscript", "onnxruntime-gpu")
)

@app.function(image=image, gpu="A100", timeout=1800, memory=32768)
def test_dit():
    import torch
    import numpy as np
    from diffusers import SanaPipeline

    pipe = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        torch_dtype=torch.float16, variant="fp16",
    ).to("cuda")

    transformer = pipe.transformer
    transformer.eval()

    # Test 1: Run transformer directly in PyTorch
    print("=== Test 1: PyTorch direct inference ===")
    with torch.no_grad():
        h = torch.randn(1, 32, 32, 32, dtype=torch.float32, device="cuda")
        e = torch.randn(1, 300, 2304, dtype=torch.float32, device="cuda")
        t = torch.tensor([500.0], dtype=torch.float32, device="cuda")

        transformer_f32 = transformer.float()
        out = transformer_f32(h, encoder_hidden_states=e, timestep=t, return_dict=False)[0]
        print(f"  PyTorch output shape: {out.shape}")
        print(f"  PyTorch output stats: min={out.min().item():.4f} max={out.max().item():.4f} mean={out.mean().item():.4f}")
        print(f"  Any non-zero: {(out != 0).any().item()}")

    # Test 2: Export to ONNX and run with onnxruntime
    print("\n=== Test 2: ONNX export + inference ===")

    class Wrapper(torch.nn.Module):
        def __init__(self, dit):
            super().__init__()
            self.dit = dit
        def forward(self, hidden_states, encoder_hidden_states, timestep):
            return self.dit(hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, return_dict=False)[0]

    wrapper = Wrapper(transformer_f32).eval()

    onnx_path = "/tmp/test_dit.onnx"

    # Use legacy exporter (not dynamo) — dynamo produces NaN for Sana's attention
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (h, e, t), onnx_path,
            input_names=["hidden_states", "encoder_hidden_states", "timestep"],
            output_names=["output"],
            opset_version=14,
            dynamo=False,
        )
    print(f"  ONNX exported (legacy): {onnx_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    ort_out = sess.run(None, {
        "hidden_states": h.cpu().numpy(),
        "encoder_hidden_states": e.cpu().numpy(),
        "timestep": t.cpu().numpy(),
    })
    result = ort_out[0]
    print(f"  ONNX output shape: {result.shape}")
    print(f"  ONNX output stats: min={result.min():.4f} max={result.max():.4f} mean={result.mean():.4f}")
    print(f"  Any non-zero: {(result != 0).any()}")

    # Compare
    pt_np = out.cpu().numpy()
    diff = np.abs(pt_np - result).max()
    print(f"\n  Max diff PyTorch vs ONNX: {diff:.6f}")

@app.local_entrypoint()
def main():
    test_dit.remote()
