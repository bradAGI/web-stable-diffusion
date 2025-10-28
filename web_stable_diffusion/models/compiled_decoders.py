"""Deterministic ONNX/TVM decoder stubs for omni-modal generation.

This module provides lightweight compiled-graph facsimiles that mimic the
behaviour of ONNX and TVM exported decoder weights.  The actual numerical
payloads are tiny and deterministic so that unit tests can validate the wiring
logic without requiring heavyweight runtimes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Dict, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import tvm

    _TVM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    tvm = None  # type: ignore[assignment]
    _TVM_AVAILABLE = False

from .modal_generators import (
    BACKEND_NUMPY,
    BACKEND_TORCH,
    BACKEND_TVM,
    BACKEND_WEBGPU,
    DeviceAwareResult,
    DeviceSpec,
)


def _deterministic_matrix(name: str, rows: int, cols: int) -> np.ndarray:
    seed = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols), dtype=np.float32)


def _deterministic_bias(name: str, size: int) -> np.ndarray:
    seed = int(hashlib.sha1((name + ":bias").encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size, dtype=np.float32) * 0.1


@dataclass(frozen=True)
class CompiledDecoder:
    """Descriptor for a compiled decoder graph."""

    name: str
    runtime: str  # ``onnx`` or ``tvm``
    version: str
    max_input_dim: int
    max_output_dim: int
    weights: np.ndarray
    bias: np.ndarray

    def project(
        self,
        embedding: np.ndarray,
        *,
        seed: int,
        features: int | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float | str]]]:
        """Project ``embedding`` through the compiled graph.

        The routine mimics a compiled execution by performing a small dense
        projection, applying a non-linearity and returning metadata that
        includes versioning, latency and resource information.
        """

        features = features or self.max_output_dim
        features = min(max(features, 1), self.max_output_dim)

        padded = np.zeros(self.max_input_dim, dtype=np.float32)
        trunc = embedding.astype(np.float32)[: self.max_input_dim]
        padded[: trunc.size] = trunc

        start = time.perf_counter()
        weights = self.weights[: trunc.size, :features]
        bias = self.bias[:features]
        projected = np.tanh(padded[: weights.shape[0]] @ weights + bias)

        # Inject a tiny deterministic modulation using the seed to emulate
        # stochastic sampling from the compiled runtime.
        rng = np.random.default_rng(seed)
        modulation = rng.standard_normal(features, dtype=np.float32) * 0.01
        output = (projected + modulation).astype(np.float32)
        latency_ms = (time.perf_counter() - start) * 1000.0

        param_count = weights.size + bias.size
        memory_mb = param_count * 4 / 1_000_000
        graph_hash = hashlib.sha1(weights.tobytes() + bias.tobytes()).hexdigest()

        metadata: Dict[str, Dict[str, float | str]] = {
            "model": {
                "name": self.name,
                "version": self.version,
                "runtime": self.runtime,
                "graph_hash": graph_hash,
            },
            "metrics": {"latency_ms": float(latency_ms)},
            "resources": {
                "parameters": float(param_count),
                "estimated_flops": float(param_count * 2),
                "memory_mb": float(memory_mb),
            },
        }
        return output, metadata


def _make_decoder(name: str, runtime: str, version: str, input_dim: int, output_dim: int) -> CompiledDecoder:
    weights = _deterministic_matrix(f"{name}:{runtime}:W", input_dim, output_dim)
    bias = _deterministic_bias(f"{name}:{runtime}:b", output_dim)
    return CompiledDecoder(name, runtime, version, input_dim, output_dim, weights, bias)


def _pack_device(array: np.ndarray, spec: DeviceSpec) -> DeviceAwareResult:
    metadata: Dict[str, str] = {"dtype": str(array.dtype), "shape": str(array.shape)}
    if spec.backend == BACKEND_TORCH:
        assert torch is not None  # pragma: no cover - gated by select_device
        tensor = torch.from_numpy(array).to(spec.device)
        return DeviceAwareResult(spec.backend, spec.device, tensor, metadata)
    if spec.backend == BACKEND_TVM:
        assert tvm is not None  # pragma: no cover - gated by select_device
        return DeviceAwareResult(spec.backend, spec.device, tvm.nd.array(array), metadata)
    if spec.backend == BACKEND_WEBGPU:
        return DeviceAwareResult(spec.backend, spec.device, {"buffer": array, "emulated": True}, metadata)
    return DeviceAwareResult(BACKEND_NUMPY, spec.device, array, metadata)


def load_compiled_decoders(device_spec: DeviceSpec) -> Dict[str, CompiledDecoder]:
    """Return deterministic decoder descriptors for the selected backend."""

    runtime = "tvm" if device_spec.backend == BACKEND_TVM else "onnx"
    version_suffix = "llvm" if runtime == "tvm" else "cpu"
    return {
        "audio": _make_decoder("audio_decoder", runtime, f"1.0.{version_suffix}", 8, 8),
        "image": _make_decoder("image_decoder", runtime, f"2.1.{version_suffix}", 16, 16),
        "volume": _make_decoder("volume_decoder", runtime, f"0.9.{version_suffix}", 12, 12),
        "video": _make_decoder("video_decoder", runtime, f"3.4.{version_suffix}", 24, 24),
    }


__all__ = ["CompiledDecoder", "load_compiled_decoders", "_pack_device"]
