"""Device-aware prototype generators for multiple modalities.

The implementations in this module intentionally keep the numerical
workloads lightweight while providing deterministic, backend-aware
behaviour.  They are primarily designed so that unit tests can assert
that the high-level omni-modal facade wires the correct components
together.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, Tuple

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

try:  # pragma: no cover - optional dependency
    import wgpu

    _WGPU_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    wgpu = None  # type: ignore[assignment]
    _WGPU_AVAILABLE = False


BACKEND_TORCH = "torch"
BACKEND_TVM = "tvm"
BACKEND_WEBGPU = "webgpu"
BACKEND_NUMPY = "numpy"


@dataclass
class DeviceSpec:
    """Simple description of the selected compute backend and device."""

    backend: str
    device: str


@dataclass
class DeviceAwareResult:
    """Container bundling the backend metadata with the generated payload."""

    backend: str
    device: str
    data: Any
    metadata: Dict[str, Any]

    def to_numpy(self) -> np.ndarray:
        """Return the payload as a NumPy array for comparisons."""

        if self.backend == BACKEND_TORCH:
            assert torch is not None  # for type-checkers
            return self.data.detach().cpu().numpy()
        if self.backend == BACKEND_TVM:
            # ``numpy()`` exists on tvm.nd.NDArray, but type-checkers do not
            # know about it, so we fall back to ``np.array``.
            if hasattr(self.data, "numpy"):
                return np.array(self.data.numpy())  # type: ignore[attr-defined]
            return np.array(self.data)
        if self.backend == BACKEND_WEBGPU:
            buffer = self.data.get("buffer", self.data) if isinstance(self.data, dict) else self.data
            return np.array(buffer)
        return np.array(self.data)


def select_device() -> DeviceSpec:
    """Select the best available backend for lightweight generation."""

    if _TORCH_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DeviceSpec(BACKEND_TORCH, device)
    if _TVM_AVAILABLE:
        assert tvm is not None  # for type-checkers
        try:
            cuda_device = tvm.cuda()
            if getattr(cuda_device, "exist", False):
                return DeviceSpec(BACKEND_TVM, "cuda")
        except Exception:  # pragma: no cover - optional dependency
            pass
        return DeviceSpec(BACKEND_TVM, "llvm")
    if _WGPU_AVAILABLE:
        return DeviceSpec(BACKEND_WEBGPU, "wgpu")
    return DeviceSpec(BACKEND_NUMPY, "cpu")


def _prompt_seed(prompt: str) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def _audio_waveform(prompt: str, length: int = 64) -> np.ndarray:
    seed = _prompt_seed(prompt)
    rng = np.random.default_rng(seed)
    freq = (seed % 5) + 1
    t = np.linspace(0, 2 * np.pi, num=length, endpoint=False, dtype=np.float32)
    wave = np.sin(freq * t).astype(np.float32)
    noise = rng.normal(scale=0.05, size=length).astype(np.float32)
    return (0.5 * wave + noise).astype(np.float32)


def audio_reference(prompt: str, length: int = 64) -> np.ndarray:
    """Public helper returning the deterministic audio reference signal."""

    return _audio_waveform(prompt, length)


def _image_tensor(prompt: str, height: int = 32, width: int = 32) -> np.ndarray:
    seed = _prompt_seed(prompt)
    rng = np.random.default_rng(seed)
    return rng.random((height, width, 3), dtype=np.float32)


def image_reference(prompt: str, height: int = 32, width: int = 32) -> np.ndarray:
    return _image_tensor(prompt, height, width)


def _volume_tensor(prompt: str, size: int = 16) -> np.ndarray:
    seed = _prompt_seed(prompt)
    rng = np.random.default_rng(seed)
    grid = rng.random((size, size, size), dtype=np.float32)
    # Add simple radial falloff to make the content spatially structured.
    coords = np.linspace(-1.0, 1.0, num=size, dtype=np.float32)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    radius = np.sqrt(x**2 + y**2 + z**2)
    falloff = np.exp(-3.0 * radius).astype(np.float32)
    return (grid * falloff).astype(np.float32)


def volume_reference(prompt: str, size: int = 16) -> np.ndarray:
    return _volume_tensor(prompt, size)


def _video_tensor(prompt: str, frames: int = 8, height: int = 16, width: int = 16) -> np.ndarray:
    seed = _prompt_seed(prompt)
    rng = np.random.default_rng(seed)
    video = rng.random((frames, height, width, 3), dtype=np.float32)
    # Introduce mild temporal smoothing by averaging neighbouring frames.
    alpha = 0.25
    for i in range(1, frames):
        video[i] = alpha * video[i - 1] + (1 - alpha) * video[i]
    return video.astype(np.float32)


def video_reference(prompt: str, frames: int = 8, height: int = 16, width: int = 16) -> np.ndarray:
    return _video_tensor(prompt, frames, height, width)


class DeviceAwareGenerator:
    """Base class shared by all modality generators."""

    def __init__(self, device_spec: DeviceSpec | None = None) -> None:
        self.device_spec = device_spec or select_device()

    def _pack_result(self, array: np.ndarray, metadata: Dict[str, Any]) -> DeviceAwareResult:
        backend = self.device_spec.backend
        device = self.device_spec.device
        if backend == BACKEND_TORCH:
            assert torch is not None  # for type-checkers
            tensor = torch.from_numpy(array).to(device)
            return DeviceAwareResult(backend, device, tensor, metadata)
        if backend == BACKEND_TVM:
            assert tvm is not None  # for type-checkers
            tvm_array = tvm.nd.array(array)
            return DeviceAwareResult(backend, device, tvm_array, metadata)
        if backend == BACKEND_WEBGPU:
            # We cannot execute WGSL in the unit-test environment, so we
            # emulate the behaviour by returning the NumPy payload along with
            # metadata that would otherwise drive shader execution.
            emulated = {"emulated": True, "buffer": array}
            return DeviceAwareResult(backend, device, emulated, metadata)
        return DeviceAwareResult(backend, device, array, metadata)


class AudioGenerator(DeviceAwareGenerator):
    """Prototype audio generator producing deterministic waveforms."""

    def generate(self, prompt: str) -> DeviceAwareResult:
        waveform = _audio_waveform(prompt)
        metadata = {"dtype": str(waveform.dtype), "shape": waveform.shape}
        return self._pack_result(waveform, metadata)


class ImageGenerator(DeviceAwareGenerator):
    """Prototype image generator producing deterministic 2D tensors."""

    def generate(self, prompt: str, resolution: int = 32) -> DeviceAwareResult:
        pixels = _image_tensor(prompt, resolution, resolution)
        metadata = {
            "dtype": str(pixels.dtype),
            "shape": pixels.shape,
            "resolution": resolution,
        }
        return self._pack_result(pixels, metadata)


class VolumeGenerator(DeviceAwareGenerator):
    """Prototype volumetric generator producing structured 3D grids."""

    def generate(self, prompt: str, size: int = 16) -> DeviceAwareResult:
        volume = _volume_tensor(prompt, size)
        metadata = {"dtype": str(volume.dtype), "shape": volume.shape}
        return self._pack_result(volume, metadata)


class VideoGenerator(DeviceAwareGenerator):
    """Prototype video generator producing temporally-smoothed tensors."""

    def generate(self, prompt: str, frames: int = 8, resolution: Tuple[int, int] = (16, 16)) -> DeviceAwareResult:
        height, width = resolution
        video = _video_tensor(prompt, frames, height, width)
        metadata = {
            "dtype": str(video.dtype),
            "shape": video.shape,
            "frames": frames,
            "resolution": resolution,
        }
        return self._pack_result(video, metadata)


__all__ = [
    "AudioGenerator",
    "audio_reference",
    "DeviceAwareResult",
    "DeviceSpec",
    "ImageGenerator",
    "image_reference",
    "VideoGenerator",
    "video_reference",
    "VolumeGenerator",
    "volume_reference",
    "select_device",
]
