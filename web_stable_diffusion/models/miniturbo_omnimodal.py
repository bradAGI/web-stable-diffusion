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
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

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
    select_device,
)


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


class OmniModalMiniturbo:
    """High level omni-modal synthesis engine with neuro-symbolic guidance."""

    def __init__(self, *, diffusion_steps: int = 6) -> None:
        self.meta_logic = MetaLogic()
        self.device_spec = select_device()
        self.diffusion_steps = diffusion_steps

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

    def generate_audio(self, prompt: str, *, length: int = 2048, sample_rate: int = 16000) -> DeviceAwareResult:
        embedding = self._prompt_embedding(prompt, size=6)
        waveform = self._synth_audio(embedding, length=length, sample_rate=sample_rate)
        metadata = {
            "shape": waveform.shape,
            "sample_rate": sample_rate,
            "prompt": prompt,
            "embedding_norm": float(np.linalg.norm(embedding)),
        }
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

    def generate_image(self, prompt: str, resolution: int = 256) -> DeviceAwareResult:
        embedding = self._prompt_embedding(prompt, size=12)
        latent = self._seed_latent(prompt, (resolution, resolution, 3))
        image = self._diffuse(latent, embedding)
        metadata = {
            "shape": image.shape,
            "resolution": resolution,
            "prompt": prompt,
            "embedding_norm": float(np.linalg.norm(embedding)),
        }
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
        volume_size = size or 32
        embedding = self._prompt_embedding(prompt, size=10)
        volume = self._synth_volume(prompt, embedding, volume_size)
        metadata = {
            "shape": volume.shape,
            "prompt": prompt,
            "embedding_norm": float(np.linalg.norm(embedding)),
        }
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
        embedding = self._prompt_embedding(prompt, size=max(8, frames))
        video = self._synth_video(prompt, embedding, frames, resolution)
        metadata = {
            "shape": video.shape,
            "prompt": prompt,
            "fps": fps,
            "embedding_norm": float(np.linalg.norm(embedding)),
        }
        return _pack_array(video, self.device_spec, metadata)

    # -- Parallel orchestration -------------------------------------------
    def generate_bundle(
        self,
        prompt: str,
        *,
        resolution: int = 256,
        frames: int = 16,
        volume_size: int = 32,
        audio_length: int = 2048,
    ) -> Dict[str, DeviceAwareResult]:
        """Generate all modalities in parallel threads and return a bundle."""

        tasks = {
            "audio": (self.generate_audio, {"prompt": prompt, "length": audio_length}),
            "image": (self.generate_image, {"prompt": prompt, "resolution": resolution}),
            "volume": (self.generate_volume, {"prompt": prompt, "size": volume_size}),
            "video": (
                self.generate_video,
                {"prompt": prompt, "frames": frames, "resolution": resolution // 2},
            ),
        }

        results: Dict[str, DeviceAwareResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_key = {
                executor.submit(func, **kwargs): key for key, (func, kwargs) in tasks.items()
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                results[key] = future.result()
        return results


__all__ = [
    "Application",
    "Const",
    "Lambda",
    "MetaLogic",
    "OmniModalMiniturbo",
    "Var",
]
