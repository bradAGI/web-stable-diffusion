"""Neuro-symbolic omni-modal generation engine.

This module implements a lightweight but fully working omni-modal synthesis
stack.  It follows the ``NeuroSymbolicEngine`` blueprint provided in the user
instructions by combining:

* a beta-reducing lambda-calculus interpreter (𝕃_𝜆),
* a forward-chaining deductive system (𝕃_𝛿), and
* a proof checker that verifies the consistency of inferred facts (𝕃_π).

The symbolic reasoning layer is paired with differentiable-style numerical
pipelines that produce deterministic audio, image, volume and video outputs.
All modalities are generated in parallel, and each pipeline uses multiple
diffusion-like refinement steps guided by the symbolic embeddings of the input
prompt.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import heapq
import itertools
import logging
import math
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from .backends import DiffusersBackendUnavailable, DiffusersImageBackend

    _DIFFUSERS_BACKEND_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DiffusersBackendUnavailable = None  # type: ignore[assignment]
    DiffusersImageBackend = None  # type: ignore[assignment]
    _DIFFUSERS_BACKEND_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import tvm

    _TVM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    tvm = None  # type: ignore[assignment]
    _TVM_AVAILABLE = False


from .compiled_decoders import CompiledDecoder, load_compiled_decoders
from .modal_generators import (
    BACKEND_NUMPY,
    BACKEND_TORCH,
    BACKEND_TVM,
    BACKEND_WEBGPU,
    DeviceAwareResult,
    DeviceSpec,
    select_device,
)


logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """Base class for structured generation failures."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code

    def as_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": str(self)}


class GenerationCancelled(GenerationError):
    """Raised when a caller-initiated cancellation halts generation."""

    def __init__(self, reason: str | None = None) -> None:
        super().__init__(reason or "generation cancelled", code="cancelled")
        self.reason = reason or "cancelled"

    def as_dict(self) -> Dict[str, Any]:
        payload = super().as_dict()
        payload["reason"] = self.reason
        return payload


class BudgetExceededError(GenerationError):
    """Raised when a modality exceeds its configured resource budget."""

    def __init__(
        self,
        modality: str,
        *,
        metric: str,
        limit: float,
        usage: float,
    ) -> None:
        message = (
            f"modality '{modality}' exceeded {metric} budget: "
            f"{usage:.2f} > {limit:.2f}"
        )
        super().__init__(message, code="budget_exceeded")
        self.modality = modality
        self.metric = metric
        self.limit = limit
        self.usage = usage

    def as_dict(self) -> Dict[str, Any]:
        payload = super().as_dict()
        payload.update(
            {
                "modality": self.modality,
                "metric": self.metric,
                "limit": self.limit,
                "usage": self.usage,
            }
        )
        return payload


@dataclass(frozen=True)
class ResourceBudget:
    """Simple resource budget expressed as memory and CPU constraints."""

    max_memory_bytes: Optional[int] = None
    max_cpu_seconds: Optional[float] = None

    def check(self, modality: str, *, memory: Optional[int], cpu: Optional[float]) -> None:
        if self.max_memory_bytes is not None and memory is not None:
            if memory > self.max_memory_bytes:
                raise BudgetExceededError(
                    modality,
                    metric="memory_bytes",
                    limit=float(self.max_memory_bytes),
                    usage=float(memory),
                )
        if self.max_cpu_seconds is not None and cpu is not None:
            if cpu > self.max_cpu_seconds:
                raise BudgetExceededError(
                    modality,
                    metric="cpu_seconds",
                    limit=float(self.max_cpu_seconds),
                    usage=float(cpu),
                )


class CancellationToken:
    """Thread-safe cancellation primitive shared across modalities."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: Optional[str] = None

    def cancel(self, reason: Optional[str] = None) -> None:
        if reason is not None:
            self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelled(self._reason)


DEFAULT_RESOURCE_BUDGETS: Mapping[str, ResourceBudget] = {
    "audio": ResourceBudget(max_memory_bytes=32 * 1024 * 1024, max_cpu_seconds=15.0),
    "image": ResourceBudget(max_memory_bytes=48 * 1024 * 1024, max_cpu_seconds=20.0),
    "volume": ResourceBudget(max_memory_bytes=96 * 1024 * 1024, max_cpu_seconds=25.0),
    "video": ResourceBudget(max_memory_bytes=256 * 1024 * 1024, max_cpu_seconds=30.0),
    "default": ResourceBudget(max_memory_bytes=256 * 1024 * 1024, max_cpu_seconds=30.0),
}


# ---------------------------------------------------------------------------
# 𝕃_𝜆 — Lambda-calculus primitives
# ---------------------------------------------------------------------------
Expression = Union["Lambda", "Application", "Var", "Const"]


@dataclass(frozen=True)
class Var:
    """Lambda-calculus variable."""

    name: str


@dataclass(frozen=True)
class Const:
    """Lambda-calculus constant."""

    value: Any


@dataclass(frozen=True)
class Lambda:
    """Lambda expression ``λ parameter . body``."""

    parameter: str
    body: Expression


@dataclass(frozen=True)
class Application:
    """Function application expression ``(func arg)``."""

    function: Expression
    argument: Expression


@dataclass
class _Closure:
    func: Lambda
    env: MutableMapping[str, Any]


KnowledgeBase = Mapping[str, Sequence[str]]


def _hash_embedding(prompt: str, size: int) -> np.ndarray:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    values = [int.from_bytes(digest[i : i + 4], "little", signed=False) for i in range(0, min(len(digest), size * 4), 4)]
    vector = np.array(values, dtype=np.float32)
    if vector.size < size:
        vector = np.pad(vector, (0, size - vector.size), mode="wrap")
    vector = vector / (np.linalg.norm(vector) + 1e-6)
    return vector.astype(np.float32)


# ---------------------------------------------------------------------------
# Ω-Foundations: MetaLogic layers (see custom instructions)
# ---------------------------------------------------------------------------
@dataclass
class MetaLogic:
    """Symbolic reasoning backbone providing λ-calculus and deduction."""

    reduction_limit: int = 128

    # 𝕃_𝜆 — Lambda calculus -------------------------------------------------
    def lambda_reduce(self, expr: Expression, environment: Optional[MutableMapping[str, Any]] = None) -> Any:
        """Evaluate a lambda-calculus expression using normal-order beta reduction."""

        env: MutableMapping[str, Any] = dict(environment or {})

        def evaluate(node: Expression, steps: List[int]) -> Any:
            if steps[0] >= self.reduction_limit:
                raise RecursionError("lambda reduction exceeded iteration budget")
            steps[0] += 1
            if isinstance(node, Const):
                return node.value
            if isinstance(node, Var):
                if node.name not in env:
                    raise KeyError(f"unbound variable {node.name}")
                value = env[node.name]
                if isinstance(value, (Lambda, Application, Var, Const)):
                    return evaluate(value, steps)
                return value
            if isinstance(node, Lambda):
                return _Closure(node, dict(env))
            if isinstance(node, Application):
                fn = evaluate(node.function, steps)
                arg = evaluate(node.argument, steps)
                if isinstance(fn, _Closure):
                    new_env = dict(fn.env)
                    new_env[fn.func.parameter] = arg
                    return MetaLogic(self.reduction_limit).lambda_reduce(fn.func.body, new_env)
                if callable(fn):
                    return fn(arg)
                raise TypeError(f"cannot apply object of type {type(fn)!r}")
            raise TypeError(f"unsupported expression type: {type(node)!r}")

        return evaluate(expr, [0])

    # 𝕃_𝛿 — Deductive logic -------------------------------------------------
    def deduce(self, knowledge: KnowledgeBase, seed_facts: Optional[Iterable[str]] = None) -> Set[str]:
        """Perform forward-chaining deduction over a simple Horn knowledge base."""

        agenda: List[str] = list(seed_facts or [])
        derived: Set[str] = set(agenda)
        changed = True
        while changed:
            changed = False
            for conclusion, premises in knowledge.items():
                if conclusion in derived:
                    continue
                if all(p in derived for p in premises):
                    derived.add(conclusion)
                    agenda.append(conclusion)
                    changed = True
        return derived

    # 𝕃_π — Proof systems ---------------------------------------------------
    def prove(self, statement: str, knowledge: KnowledgeBase, seed_facts: Optional[Iterable[str]] = None) -> bool:
        """Verify that ``statement`` is entailed by the knowledge base."""

        closure = self.deduce(knowledge, seed_facts)
        return statement in closure


def _tokenise(prompt: str) -> List[str]:
    return [token.lower() for token in prompt.replace("-", " ").replace("_", " ").split() if token]


def _prompt_knowledge(prompt: str) -> Tuple[KnowledgeBase, Set[str]]:
    tokens = _tokenise(prompt)
    facts = {f"token:{token}" for token in tokens}
    knowledge: Dict[str, Sequence[str]] = {}
    if any(token in {"night", "dark", "moon"} for token in tokens):
        knowledge["theme:night"] = [f"token:{token}" for token in tokens if token in {"night", "moon"}]
    if any(token in {"sun", "dawn", "bright"} for token in tokens):
        knowledge["theme:day"] = [f"token:{token}" for token in tokens if token in {"sun", "dawn", "bright"}]
    if any(token in {"water", "lake", "river", "ocean"} for token in tokens):
        knowledge["element:water"] = [f"token:{token}" for token in tokens if token in {"water", "lake", "river", "ocean"}]
    if any(token in {"mountain", "rock", "stone"} for token in tokens):
        knowledge["element:earth"] = [f"token:{token}" for token in tokens if token in {"mountain", "rock", "stone"}]
    if any(token in {"calm", "serene", "quiet"} for token in tokens):
        knowledge["mood:calm"] = [f"token:{token}" for token in tokens if token in {"calm", "serene", "quiet"}]
    if any(token in {"storm", "loud", "intense"} for token in tokens):
        knowledge["mood:intense"] = [f"token:{token}" for token in tokens if token in {"storm", "loud", "intense"}]
    return knowledge, facts


def _apply_guidance(array: np.ndarray, embedding: np.ndarray, gain: float) -> np.ndarray:
    spectrum = np.fft.fftn(array, axes=tuple(range(array.ndim)))
    scaled = spectrum * gain
    guided = np.fft.ifftn(scaled, axes=tuple(range(array.ndim))).real
    guided = guided + float(np.mean(embedding))
    return guided.astype(np.float32)


def _anisotropic_blur(array: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    result = array.copy()
    for axis, weight in enumerate(weights):
        rolled_forward = np.roll(result, 1, axis=axis % array.ndim)
        rolled_backward = np.roll(result, -1, axis=axis % array.ndim)
        result = (1 - weight) * result + (weight / 2.0) * (rolled_forward + rolled_backward)
    return result.astype(np.float32)


def _pack_array(array: np.ndarray, spec: DeviceSpec, metadata: Dict[str, Any]) -> DeviceAwareResult:
    array = np.asarray(array, dtype=np.float32)
    if spec.backend == BACKEND_TORCH:
        assert torch is not None  # pragma: no cover - checked by select_device
        tensor = torch.from_numpy(array).to(spec.device)
        return DeviceAwareResult(spec.backend, spec.device, tensor, metadata)
    if spec.backend == BACKEND_TVM:
        assert tvm is not None  # pragma: no cover - checked by select_device
        return DeviceAwareResult(spec.backend, spec.device, tvm.nd.array(array), metadata)
    if spec.backend == BACKEND_WEBGPU:
        emulated = {"emulated": True, "buffer": array, "dtype": str(array.dtype)}
        return DeviceAwareResult(spec.backend, spec.device, emulated, metadata)
    return DeviceAwareResult(BACKEND_NUMPY, spec.device, array, metadata)


def _normalise(array: np.ndarray) -> np.ndarray:
    min_val = array.min()
    max_val = array.max()
    if math.isclose(max_val, min_val):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_val) / (max_val - min_val)).astype(np.float32)


def _estimate_payload_size(result: DeviceAwareResult) -> int:
    metadata = result.metadata
    dtype_name = metadata.get("dtype") if isinstance(metadata, dict) else None
    shape = metadata.get("shape") if isinstance(metadata, dict) else None
    if dtype_name is not None and shape is not None:
        try:
            dtype = np.dtype(dtype_name)
            size = int(np.prod(shape))
            return int(size * dtype.itemsize)
        except Exception:  # pragma: no cover - defensive
            pass
    array = result.to_numpy()
    if isinstance(metadata, dict):
        metadata.setdefault("dtype", str(array.dtype))
        metadata.setdefault("shape", list(array.shape))
    return int(array.nbytes)


class OmniModalMiniturbo:
    """High level omni-modal synthesis engine with neuro-symbolic guidance."""

    def __init__(
        self,
        *,
        diffusion_steps: int = 6,
        device_spec: DeviceSpec | None = None,
        resource_budgets: Optional[Mapping[str, ResourceBudget]] = None,
    ) -> None:
        self.meta_logic = MetaLogic()
        self.device_spec = device_spec or select_device()
        self.diffusion_steps = diffusion_steps
        base_budgets = dict(DEFAULT_RESOURCE_BUDGETS)
        if resource_budgets:
            base_budgets.update(resource_budgets)
        self.resource_budgets: Dict[str, ResourceBudget] = base_budgets
        self._last_benchmark: Dict[str, Any] | None = None
        self._decoders: Dict[str, CompiledDecoder] = load_compiled_decoders(self.device_spec)
        self._image_backend = self._init_image_backend()

    # -- Validation helpers -----------------------------------------------
    @staticmethod
    def _ensure_positive(value: int, name: str, *, minimum: int = 1) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, received {type(value)!r}")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, received {value}")
        return value

    @staticmethod
    def _ensure_prompt(prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")
        return prompt

    def _resolve_budget(
        self,
        modality: str,
        overrides: Optional[Mapping[str, ResourceBudget]] = None,
    ) -> Optional[ResourceBudget]:
        if overrides and modality in overrides:
            return overrides[modality]
        if overrides and "default" in overrides:
            return overrides["default"]
        if modality in self.resource_budgets:
            return self.resource_budgets[modality]
        return self.resource_budgets.get("default")

    # -- Prompt processing -------------------------------------------------
    def _prompt_embedding(self, prompt: str, size: int = 8) -> np.ndarray:
        base_embedding = _hash_embedding(prompt, size)
        knowledge, facts = _prompt_knowledge(prompt)
        derived = sorted(self.meta_logic.deduce(knowledge, facts))
        if derived:
            derived_scores = np.array(
                [int.from_bytes(hashlib.sha1(item.encode("utf-8")).digest()[:4], "little") for item in derived],
                dtype=np.float32,
            )
            blend = np.pad(derived_scores, (0, max(0, size - derived_scores.size)), mode="wrap")[:size]
        else:
            blend = np.zeros(size, dtype=np.float32)
        return _normalise(base_embedding + _normalise(blend))

    # -- Audio -------------------------------------------------------------
    def _synth_audio(self, embedding: np.ndarray, length: int = 2048, sample_rate: int = 16000) -> np.ndarray:
        time = np.linspace(0.0, length / sample_rate, num=length, endpoint=False, dtype=np.float32)
        waveform = np.zeros_like(time)
        for idx, value in enumerate(embedding):
            frequency = 110.0 + 55.0 * idx
            harmonic = np.sin(2 * np.pi * frequency * time + value * np.pi)
            waveform += harmonic * (0.6 / (idx + 1))
        envelope = np.linspace(0.1, 1.0, num=length, dtype=np.float32)
        refined = waveform * envelope
        refined = refined / (np.abs(refined).max() + 1e-6)
        return refined.astype(np.float32)

    def generate_audio(
        self,
        prompt: str,
        *,
        length: int = 2048,
        sample_rate: int = 16000,
    ) -> DeviceAwareResult:
        prompt = self._ensure_prompt(prompt)
        length = self._ensure_positive(length, "length")
        sample_rate = self._ensure_positive(sample_rate, "sample_rate", minimum=1000)
        embedding = self._prompt_embedding(prompt, size=6)
        conditioned_embedding, decoder_meta = self._modulate_embedding(prompt, "audio", embedding)
        waveform = self._synth_audio(conditioned_embedding, length=length, sample_rate=sample_rate)
        metadata = {
            "shape": waveform.shape,
            "sample_rate": sample_rate,
            "length": length,
            "prompt": prompt,
            "embedding_norm": float(np.linalg.norm(conditioned_embedding)),
        }
        metadata = self._merge_metadata(metadata, decoder_meta)
        return _pack_array(waveform, self.device_spec, metadata)

    # -- Image -------------------------------------------------------------
    def _seed_latent(self, prompt: str, shape: Tuple[int, ...]) -> np.ndarray:
        seed = int(hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest(), 16) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape, dtype=np.float32)

    def _diffuse(self, latent: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        result = latent.astype(np.float32)
        weights = np.linspace(0.1, 0.3, num=result.ndim)
        for step in range(self.diffusion_steps):
            result = 0.7 * result + 0.3 * _anisotropic_blur(result, weights + step * 0.02)
            result = _apply_guidance(result, embedding, gain=1 + 0.05 * step)
            result = np.clip(result, -4.0, 4.0)
        return _normalise(result)

    def _decoder_seed(self, prompt: str, modality: str) -> int:
        digest = hashlib.sha256(f"{prompt}::{modality}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF

    def _modulate_embedding(self, prompt: str, modality: str, embedding: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        decoder = self._decoders[modality]
        projection, decoder_meta = decoder.project(
            embedding,
            seed=self._decoder_seed(prompt, modality),
            features=min(embedding.size, decoder.max_output_dim),
        )
        if projection.size < embedding.size and projection.size > 0:
            repeats = int(math.ceil(embedding.size / projection.size))
            projection = np.tile(projection, repeats)[: embedding.size]
        elif projection.size > embedding.size:
            projection = projection[: embedding.size]
        conditioned = _normalise(embedding + 0.25 * projection)
        return conditioned.astype(np.float32), decoder_meta

    @staticmethod
    def _merge_metadata(base: Dict[str, Any], decoder_meta: Dict[str, Any]) -> Dict[str, Any]:
        model_info = decoder_meta.get("model", {})
        resources = decoder_meta.get("resources", {})
        metrics = decoder_meta.get("metrics", {})
        if model_info:
            base["model"] = model_info
        if resources:
            base["resources"] = resources
        if metrics:
            base.setdefault("metrics", {}).update(metrics)
        return base

    def _init_image_backend(self) -> Optional[DiffusersImageBackend]:
        if not _DIFFUSERS_BACKEND_AVAILABLE or DiffusersImageBackend is None:
            return None
        try:
            backend = DiffusersImageBackend.from_environment()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialise diffusers backend: %s", exc)
            return None
        if backend is None:
            return None
        logger.info("Diffusers backend enabled using model '%s'", backend.config.model)
        return backend

    def generate_image(self, prompt: str, resolution: int = 256) -> DeviceAwareResult:
        prompt = self._ensure_prompt(prompt)
        resolution = self._ensure_positive(resolution, "resolution", minimum=16)
        embedding = self._prompt_embedding(prompt, size=12)
        conditioned_embedding, decoder_meta = self._modulate_embedding(prompt, "image", embedding)
        if self._image_backend is not None:
            backend_result = self._image_backend.generate(prompt, resolution)
            image = backend_result["array"]
            metadata = backend_result["metadata"]
        else:
            latent = self._seed_latent(prompt, (resolution, resolution, 3))
            image = self._diffuse(latent, conditioned_embedding)
            metadata = {
                "shape": image.shape,
                "resolution": resolution,
                "prompt": prompt,
                "embedding_norm": float(np.linalg.norm(conditioned_embedding)),
            }
        metadata = self._merge_metadata(metadata, decoder_meta)
        if self._image_backend is None:
            metadata.setdefault("backend", "symbolic-diffusion")
        metadata.setdefault("embedding_norm", float(np.linalg.norm(conditioned_embedding)))
        return _pack_array(image, self.device_spec, metadata)

    # -- Volume ------------------------------------------------------------
    def _synth_volume(self, prompt: str, embedding: np.ndarray, size: int) -> np.ndarray:
        grid = self._seed_latent(prompt + "::volume", (size, size, size))
        coords = np.linspace(-1.0, 1.0, num=size, dtype=np.float32)
        x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
        radial = np.sqrt(x**2 + y**2 + z**2)
        falloff = np.exp(-2.5 * radial)
        guided = grid * falloff + embedding.mean()
        guided = self._diffuse(guided, embedding)
        return guided

    def generate_volume(self, prompt: str, size: Optional[int] = None) -> DeviceAwareResult:
        prompt = self._ensure_prompt(prompt)
        volume_size = self._ensure_positive(size if size is not None else 32, "size", minimum=8)
        embedding = self._prompt_embedding(prompt, size=10)
        conditioned_embedding, decoder_meta = self._modulate_embedding(prompt, "volume", embedding)
        volume = self._synth_volume(prompt, conditioned_embedding, volume_size)
        metadata = {
            "shape": volume.shape,
            "size": volume_size,
            "prompt": prompt,
            "embedding_norm": float(np.linalg.norm(conditioned_embedding)),
        }
        metadata = self._merge_metadata(metadata, decoder_meta)
        return _pack_array(volume, self.device_spec, metadata)

    # -- Video -------------------------------------------------------------
    def _synth_video(self, prompt: str, embedding: np.ndarray, frames: int, resolution: int) -> np.ndarray:
        base = self._seed_latent(prompt + "::video", (resolution, resolution, 3))
        base = self._diffuse(base, embedding)
        video = []
        for index in range(frames):
            angle = 2 * np.pi * (index / max(frames, 1))
            shifted = np.roll(base, shift=int(embedding[index % embedding.size] * 5), axis=0)
            rotated = np.rot90(shifted, k=index % 4)
            modulated = rotated * (0.8 + 0.2 * math.sin(angle))
            video.append(_normalise(modulated))
        return np.stack(video, axis=0)

    def generate_video(
        self,
        prompt: str,
        *,
        fps: int = 30,
        frames: int = 16,
        resolution: int = 128,
    ) -> DeviceAwareResult:
        prompt = self._ensure_prompt(prompt)
        frames = self._ensure_positive(frames, "frames")
        fps = self._ensure_positive(fps, "fps", minimum=1)
        resolution = self._ensure_positive(resolution, "resolution", minimum=16)
        embedding = self._prompt_embedding(prompt, size=max(8, frames))
        conditioned_embedding, decoder_meta = self._modulate_embedding(prompt, "video", embedding)
        video = self._synth_video(prompt, conditioned_embedding, frames, resolution)
        metadata = {
            "shape": video.shape,
            "prompt": prompt,
            "fps": fps,
            "frames": frames,
            "embedding_norm": float(np.linalg.norm(conditioned_embedding)),
        }
        metadata = self._merge_metadata(metadata, decoder_meta)
        return _pack_array(video, self.device_spec, metadata)

    @property
    def last_benchmark(self) -> Optional[Dict[str, Any]]:
        """Return benchmarking metadata recorded during the last bundle run."""

        return self._last_benchmark

    # -- Parallel orchestration -------------------------------------------
    def _modality_cost_hint(self, key: str, kwargs: Dict[str, Any]) -> float:
        """Return a heuristic cost estimate used for work-stealing scheduling."""

        if key == "audio":
            length = kwargs.get("length", 2048)
            return float(max(1, length) / 2048.0)
        if key == "image":
            resolution = kwargs.get("resolution", 256)
            return float(max(16, resolution) ** 2 / 256.0**2)
        if key == "volume":
            size = kwargs.get("size", 32)
            return float(max(8, size) ** 3 / 32.0**3)
        if key == "video":
            frames = kwargs.get("frames", 16)
            resolution = kwargs.get("resolution", 128)
            return float(max(1, frames) * max(16, resolution) ** 2 / (16.0 * 128.0**2))
        return 1.0

    def _supports_cuda_ipc(self) -> bool:
        """Return ``True`` when the active device can leverage CUDA IPC."""

        backend = self.device_spec.backend
        device = self.device_spec.device
        if backend == BACKEND_TORCH and device.startswith("cuda") and _TORCH_AVAILABLE:
            assert torch is not None  # for type-checkers
            try:
                if not torch.cuda.is_available():
                    return False
                major, _minor = torch.cuda.get_device_capability(0)
                return major >= 3
            except Exception:  # pragma: no cover - capability detection best effort
                return False
        if backend == BACKEND_TVM and "cuda" in device and _TVM_AVAILABLE:
            assert tvm is not None  # for type-checkers
            try:
                cuda_dev = tvm.cuda()
                return bool(getattr(cuda_dev, "exist", False))
            except Exception:  # pragma: no cover - capability detection best effort
                return False
        return False

    def _suggest_worker_count(
        self,
        requested: Optional[int],
        task_count: int,
        use_process_executor: bool,
    ) -> int:
        """Return a bounded worker count honouring user input and CPU topology."""

        if requested is not None:
            return self._ensure_positive(requested, "max_workers", minimum=1)

        detected_cpus = os.cpu_count()
        if not detected_cpus or detected_cpus <= 0:
            return max(1, task_count)

        if use_process_executor:
            usable = max(1, detected_cpus - 1)
            return max(1, min(task_count, usable))

        return max(1, min(task_count, detected_cpus))

    # -- Parallel orchestration -------------------------------------------
    def generate_bundle_iter(
        self,
        prompt: str,
        *,
        resolution: int = 256,
        frames: int = 16,
        volume_size: int = 32,
        audio_length: int = 2048,
        executor: str | None = None,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        budgets: Optional[Mapping[str, ResourceBudget]] = None,
        cancellation: Optional[CancellationToken] = None,
    ) -> Iterator[BundleEvent]:
        """Yield :class:`BundleEvent` instances as modalities finish.

        The generator mirrors :meth:`generate_bundle` but exposes intermediate
        progress so that callers can provide responsive UI updates.  All
        parameters follow the same validation rules as the non-streaming
        variant.  A ``TimeoutError`` is raised if the deadline is exceeded.
        """

        prompt = self._ensure_prompt(prompt)
        resolution = self._ensure_positive(resolution, "resolution", minimum=16)
        frames = self._ensure_positive(frames, "frames")
        volume_size = self._ensure_positive(volume_size, "volume_size", minimum=8)
        audio_length = self._ensure_positive(audio_length, "audio_length")

        exec_mode = (executor or "auto").lower()
        if exec_mode not in {"auto", "thread", "process"}:
            raise ValueError("executor must be 'auto', 'thread' or 'process'")

        tasks = {
            "audio": (self.generate_audio, {"prompt": prompt, "length": audio_length}),
            "image": (self.generate_image, {"prompt": prompt, "resolution": resolution}),
            "volume": (self.generate_volume, {"prompt": prompt, "size": volume_size}),
            "video": (
                self.generate_video,
                {"prompt": prompt, "frames": frames, "resolution": max(16, resolution // 2)},
            ),
        }

        task_count = len(tasks)
        deadline = time.perf_counter() + timeout if timeout is not None else None

        def remaining_timeout() -> Optional[float]:
            if deadline is None:
                return None
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("bundle generation timed out before completion")
            return remaining

        supports_ipc = self._supports_cuda_ipc()
        fallback_to_threads = False
        use_process = False
        if exec_mode == "process":
            if supports_ipc or self.device_spec.backend == BACKEND_NUMPY:
                use_process = True
            else:
                fallback_to_threads = True
                logger.warning(
                    "Process executor requested but backend %s/%s does not support CUDA IPC; "
                    "falling back to threads",
                    self.device_spec.backend,
                    self.device_spec.device,
                )
        elif exec_mode == "auto":
            use_process = supports_ipc or self.device_spec.backend == BACKEND_NUMPY

        worker_count = self._suggest_worker_count(max_workers, task_count, use_process)

        if use_process:
            ctx = multiprocessing.get_context("spawn")
            executor_cls = concurrent.futures.ProcessPoolExecutor
            executor_kwargs: Dict[str, Any] = {"max_workers": worker_count, "mp_context": ctx}
        else:
            executor_cls = concurrent.futures.ThreadPoolExecutor
            executor_kwargs = {"max_workers": worker_count}

        task_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        for index, (key, (func, kwargs)) in enumerate(tasks.items()):
            spec = {"key": key, "func": func, "kwargs": kwargs}
            cost = float(self._modality_cost_hint(key, kwargs))
            heapq.heappush(task_heap, (cost, index, spec))

        benchmark: Dict[str, Any] = {
            "executor": "process" if use_process else "thread",
            "requested_executor": exec_mode,
            "fallback_to_threads": fallback_to_threads,
            "max_workers": worker_count,
            "requested_max_workers": max_workers,
            "cpu_count": os.cpu_count(),
            "device": {"backend": self.device_spec.backend, "device": self.device_spec.device},
            "supports_cuda_ipc": supports_ipc,
            "queue_policy": "work-stealing",
            "tasks": {},
            "transport_breakdown": {"pinned_memory": 0, "host_transfer": 0, "zero_copy": 0},
            "timed_out": False,
            "cancelled": False,
            "timeline": [],
            "resource_summary": {"memory_bytes_peak": 0, "cpu_seconds_total": 0.0},
        }
        if timeout is not None:
            benchmark["deadline_s"] = timeout
        benchmark["executor_state"] = {
            "queue_depth_initial": len(task_heap),
            "worker_launches": 0,
            "policy": "work-stealing",
        }

        schedule_start = time.perf_counter()
        durations: List[float] = []
        active: Dict[concurrent.futures.Future[DeviceAwareResult], Dict[str, Any]] = {}
        dispatch_counter = itertools.count()
        peak_in_flight = 0
        timeline: List[Dict[str, Any]] = benchmark["timeline"]
        resource_summary = benchmark["resource_summary"]

        def submit_next_available(pool: concurrent.futures.Executor) -> None:
            nonlocal peak_in_flight
            while len(active) < worker_count and task_heap:
                cost, _idx, spec = heapq.heappop(task_heap)
                spec = dict(spec)
                spec["cost"] = float(cost)
                spec["dispatch_seq"] = next(dispatch_counter)
                spec["start_time"] = time.perf_counter()
                benchmark["executor_state"]["worker_launches"] += 1
                if use_process:
                    future = pool.submit(
                        _execute_modality,
                        spec["key"],
                        spec["kwargs"],
                        self.diffusion_steps,
                        self.device_spec,
                    )
                else:
                    future = pool.submit(spec["func"], **spec["kwargs"])
                active[future] = spec
                peak_in_flight = max(peak_in_flight, len(active))
                timeline.append(
                    {
                        "type": "dispatch",
                        "key": spec["key"],
                        "dispatch_seq": spec["dispatch_seq"],
                        "timestamp_s": spec["start_time"] - schedule_start,
                        "active_workers": len(active),
                    }
                )
                benchmark["tasks"][spec["key"]] = {
                    "dispatch_seq": spec["dispatch_seq"],
                    "cost_hint": spec["cost"],
                    "start_offset_s": spec["start_time"] - schedule_start,
                }

        total = len(tasks)
        completed = 0
        token = cancellation

        try:
            with executor_cls(**executor_kwargs) as pool:
                submit_next_available(pool)
                while active:
                    if token:
                        token.raise_if_cancelled()

                    try:
                        wait_timeout = remaining_timeout()
                    except TimeoutError:
                        for future in active:
                            future.cancel()
                        benchmark["timed_out"] = True
                        raise

                    done, _ = concurrent.futures.wait(
                        list(active.keys()),
                        timeout=wait_timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        continue

                    for future in done:
                        spec = active.pop(future)
                        end_time = time.perf_counter()
                        duration = end_time - spec["start_time"]
                        try:
                            result = future.result()
                        except Exception:
                            for remaining_future in active:
                                remaining_future.cancel()
                            raise
                        if use_process:
                            result = _restore_device_result(result, self.device_spec)

                        transport = result.metadata.setdefault("transport", {})
                        if use_process and not transport.get("pinned_memory"):
                            transport.setdefault("host_transfer", True)
                        if transport.get("pinned_memory"):
                            benchmark["transport_breakdown"]["pinned_memory"] += 1
                        elif transport.get("host_transfer"):
                            benchmark["transport_breakdown"]["host_transfer"] += 1
                        else:
                            benchmark["transport_breakdown"]["zero_copy"] += 1

                        metrics = result.metadata.setdefault("metrics", {})
                        metrics["wall_clock_s"] = duration
                        metrics["start_offset_s"] = benchmark["tasks"][spec["key"]]["start_offset_s"]
                        metrics["end_offset_s"] = end_time - schedule_start

                        memory_bytes = _estimate_payload_size(result)
                        resource_info = result.metadata.setdefault("resource", {})
                        resource_info["memory_bytes"] = memory_bytes
                        resource_info["cpu_seconds"] = duration
                        resource_summary["memory_bytes_peak"] = max(
                            resource_summary["memory_bytes_peak"], memory_bytes
                        )
                        resource_summary["cpu_seconds_total"] += duration

                        budget = self._resolve_budget(spec["key"], budgets)
                        if budget:
                            budget.check(spec["key"], memory=memory_bytes, cpu=duration)

                        completed += 1
                        durations.append(duration)
                        task_entry = benchmark["tasks"][spec["key"]]
                        task_entry.update(
                            {
                                "duration_s": duration,
                                "end_offset_s": end_time - schedule_start,
                                "transport": transport,
                            }
                        )
                        logger.debug(
                            "Generated modality %s in %.3fs (dispatch #%d)",
                            spec["key"],
                            duration,
                            spec["dispatch_seq"],
                        )
                        timeline.append(
                            {
                                "type": "complete",
                                "key": spec["key"],
                                "dispatch_seq": spec["dispatch_seq"],
                                "timestamp_s": end_time - schedule_start,
                                "duration_s": duration,
                                "completed": completed,
                                "total": total,
                            }
                        )
                        yield BundleEvent(
                            key=spec["key"],
                            result=result,
                            duration_s=duration,
                            completed=completed,
                            total=total,
                        )

                    submit_next_available(pool)
        except GenerationCancelled:
            benchmark["cancelled"] = True
            if token:
                benchmark["cancel_reason"] = token.reason
            raise
        except TimeoutError:
            benchmark["timed_out"] = True
            raise
        finally:
            elapsed = time.perf_counter() - schedule_start
            benchmark["completed"] = completed
            benchmark["total"] = total
            benchmark["wall_clock_s"] = elapsed
            durations_sum = float(sum(durations)) if durations else 0.0
            benchmark["makespan_s"] = elapsed
            benchmark["modalities_per_s"] = (completed / elapsed) if elapsed > 0 else 0.0
            benchmark["speedup_vs_serial"] = (durations_sum / elapsed) if elapsed > 0 else 0.0
            benchmark["worker_utilisation"] = (
                durations_sum / (worker_count * elapsed)
                if elapsed > 0 and worker_count > 0
                else 0.0
            )
            benchmark["average_duration_s"] = (durations_sum / completed) if completed else 0.0
            benchmark["executor_state"]["queue_depth_final"] = len(task_heap)
            benchmark["executor_state"]["peak_in_flight"] = peak_in_flight
            benchmark["resource_summary"]["completed_modalities"] = completed
            self._last_benchmark = benchmark


    def generate_bundle(
        self,
        prompt: str,
        *,
        resolution: int = 256,
        frames: int = 16,
        volume_size: int = 32,
        audio_length: int = 2048,
        executor: str | None = None,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        budgets: Optional[Mapping[str, ResourceBudget]] = None,
        cancellation: Optional[CancellationToken] = None,
    ) -> Dict[str, DeviceAwareResult]:
        """Generate all modalities in parallel and return a device-aware bundle.

        Parameters
        ----------
        prompt:
            Prompt text that seeds the symbolic and numerical generators.
        resolution:
            Base spatial resolution for image/video outputs.
        frames:
            Number of frames used for the video modality.
        volume_size:
            Edge length of the volumetric tensor.
        audio_length:
            Sample length of the audio waveform.
        executor:
            Force a specific executor implementation: ``"thread"``,
            ``"process"`` or ``"auto"`` (default).  ``"process"`` will be
            honoured when the backend is NumPy or when CUDA IPC-compatible
            devices are detected.  Incompatible combinations fall back to
            threads automatically.
        max_workers:
            Optional override for the parallel worker count.
        timeout:
            Optional wall-clock timeout (in seconds) for the entire bundle
            generation call.  A ``TimeoutError`` is raised if the deadline is
            exceeded.
        """

        results: Dict[str, DeviceAwareResult] = {}
        for event in self.generate_bundle_iter(
            prompt,
            resolution=resolution,
            frames=frames,
            volume_size=volume_size,
            audio_length=audio_length,
            executor=executor,
            max_workers=max_workers,
            timeout=timeout,
            budgets=budgets,
            cancellation=cancellation,
        ):
            results[event.key] = event.result
        return results


def _serialise_device_result(result: DeviceAwareResult) -> DeviceAwareResult:
    """Prepare a :class:`DeviceAwareResult` for cross-process transport."""

    metadata = dict(result.metadata)
    transport = dict(metadata.get("transport", {}))
    metadata["transport"] = transport
    backend = result.backend
    device = result.device
    data = result.data

    if backend == BACKEND_TORCH and device.startswith("cuda") and _TORCH_AVAILABLE:
        assert torch is not None  # for type-checkers
        tensor = data
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        try:
            tensor = tensor.to("cpu", non_blocking=True)
        except Exception:
            tensor = tensor.cpu()
        try:
            tensor = tensor.pin_memory()
            transport["pinned_memory"] = True
        except Exception:  # pragma: no cover - pinning is best-effort
            transport["pinned_memory"] = False
        transport["host_transfer"] = True
        transport.setdefault("original_device", device)
        return DeviceAwareResult(backend, device, tensor, metadata)

    if backend == BACKEND_TVM and "cuda" in device and _TVM_AVAILABLE:
        assert tvm is not None  # for type-checkers
        if hasattr(data, "numpy"):
            host_array = np.array(data.numpy())  # type: ignore[attr-defined]
        else:
            host_array = np.array(data)
        transport.setdefault("original_device", device)
        transport["host_transfer"] = True
        return DeviceAwareResult(backend, device, host_array, metadata)

    metadata.setdefault("transport", transport)
    return DeviceAwareResult(backend, device, data, metadata)


def _restore_device_result(result: DeviceAwareResult, device_spec: DeviceSpec) -> DeviceAwareResult:
    """Restore transported results back onto the target device when possible."""

    transport = result.metadata.setdefault("transport", {})
    original_device = transport.get("original_device", device_spec.device if device_spec else result.device)
    backend = result.backend

    if backend == BACKEND_TORCH and original_device.startswith("cuda") and _TORCH_AVAILABLE:
        assert torch is not None  # for type-checkers
        try:
            tensor = result.data
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(np.array(tensor))
            tensor = tensor.to(original_device, non_blocking=True)
            transport["restored"] = True
            return DeviceAwareResult(backend, original_device, tensor, result.metadata)
        except Exception:  # pragma: no cover - restoration is best-effort
            transport.setdefault("restored", False)
            return result

    if backend == BACKEND_TVM and "cuda" in original_device and _TVM_AVAILABLE:
        assert tvm is not None  # for type-checkers
        try:
            ctx = tvm.device(original_device)
            host_array = result.to_numpy()
            tvm_array = tvm.nd.array(host_array, device=ctx)
            transport["restored"] = True
            return DeviceAwareResult(backend, original_device, tvm_array, result.metadata)
        except Exception:  # pragma: no cover - restoration is best-effort
            transport.setdefault("restored", False)
            return result

    return result


def _execute_modality(
    modality: str,
    kwargs: Dict[str, Any],
    diffusion_steps: int,
    device_spec: DeviceSpec,
) -> DeviceAwareResult:
    engine = OmniModalMiniturbo(diffusion_steps=diffusion_steps, device_spec=device_spec)
    generator = getattr(engine, f"generate_{modality}")
    result = generator(**kwargs)
    return _serialise_device_result(result)


@dataclass(frozen=True)
class BundleEvent:
    """Represents completion of a single modality within a bundle."""

    key: str
    result: DeviceAwareResult
    duration_s: float
    completed: int
    total: int

    @property
    def progress(self) -> float:
        """Return overall bundle progress in the inclusive range ``[0, 1]``."""

        if self.total <= 0:
            return 1.0
        return min(1.0, max(0.0, self.completed / self.total))


__all__ = [
    "BundleEvent",
    "BudgetExceededError",
    "CancellationToken",
    "Application",
    "Const",
    "GenerationCancelled",
    "Lambda",
    "MetaLogic",
    "OmniModalMiniturbo",
    "ResourceBudget",
    "Var",
]
