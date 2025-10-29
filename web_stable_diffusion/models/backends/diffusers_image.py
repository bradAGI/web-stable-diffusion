"""Diffusers-powered image backend for the omni-modal engine."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from diffusers import StableDiffusionPipeline

    _DIFFUSERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StableDiffusionPipeline = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    _DIFFUSERS_AVAILABLE = False


logger = logging.getLogger(__name__)


class DiffusersBackendUnavailable(RuntimeError):
    """Raised when the diffusers integration cannot be initialised."""


@dataclass
class DiffusersConfig:
    """Configuration describing how to initialise the pipeline."""

    model: str
    torch_dtype: str = "auto"
    revision: Optional[str] = None
    variant: Optional[str] = None
    enable_xformers: bool = True
    guidance_scale: float = 7.5
    num_inference_steps: int = 30

    @classmethod
    def from_environment(cls) -> Optional["DiffusersConfig"]:
        model = os.getenv("OMNIMODAL_DIFFUSERS_MODEL")
        if not model:
            return None
        dtype = os.getenv("OMNIMODAL_DIFFUSERS_DTYPE", "auto")
        revision = os.getenv("OMNIMODAL_DIFFUSERS_REVISION") or None
        variant = os.getenv("OMNIMODAL_DIFFUSERS_VARIANT") or None
        guidance = float(os.getenv("OMNIMODAL_DIFFUSERS_GUIDANCE", "7.5"))
        steps = int(os.getenv("OMNIMODAL_DIFFUSERS_STEPS", "30"))
        enable_xformers = os.getenv("OMNIMODAL_DIFFUSERS_ENABLE_XFORMERS", "1") not in {"0", "false", "False"}
        return cls(
            model=model,
            torch_dtype=dtype,
            revision=revision,
            variant=variant,
            enable_xformers=enable_xformers,
            guidance_scale=guidance,
            num_inference_steps=steps,
        )

    def dtype(self) -> Optional[torch.dtype]:  # type: ignore[override]
        if torch is None:
            return None
        if self.torch_dtype in {"auto", "Auto"}:
            return None
        try:
            return getattr(torch, self.torch_dtype)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise DiffusersBackendUnavailable(
                f"Unsupported torch dtype '{self.torch_dtype}'"
            ) from exc


class DiffusersImageBackend:
    """Wrapper around :class:`StableDiffusionPipeline` for image synthesis."""

    def __init__(self, pipeline: StableDiffusionPipeline, config: DiffusersConfig) -> None:
        if not _DIFFUSERS_AVAILABLE:
            raise DiffusersBackendUnavailable(
                "diffusers and torch must be installed to use this backend"
            )
        self._pipeline = pipeline
        self._config = config
        logger.info(
            "Initialised diffusers pipeline '%s' with guidance=%s steps=%s",
            config.model,
            config.guidance_scale,
            config.num_inference_steps,
        )

    @classmethod
    def from_environment(cls) -> Optional["DiffusersImageBackend"]:
        if not _DIFFUSERS_AVAILABLE:
            return None
        config = DiffusersConfig.from_environment()
        if config is None:
            return None
        return cls(_initialise_pipeline(config), config)

    @property
    def config(self) -> DiffusersConfig:
        return self._config

    def generate(self, prompt: str, resolution: int) -> Dict[str, Any]:
        """Generate an image for the prompt at the requested resolution."""

        options: Dict[str, Any] = {
            "height": resolution,
            "width": resolution,
            "num_inference_steps": self._config.num_inference_steps,
            "guidance_scale": self._config.guidance_scale,
        }
        logger.debug("Running diffusers pipeline with options %s", options)
        result = self._pipeline(prompt, **options)
        image = result.images[0]
        array = np.asarray(image).astype(np.float32) / 255.0
        metadata = {
            "shape": array.shape,
            "resolution": resolution,
            "prompt": prompt,
            "backend": "diffusers",
            "model": {
                "id": self._config.model,
                "revision": self._config.revision,
                "variant": self._config.variant,
                "guidance_scale": self._config.guidance_scale,
                "steps": self._config.num_inference_steps,
            },
        }
        return {"array": array, "metadata": metadata}


@lru_cache(maxsize=1)
def _initialise_pipeline(config: DiffusersConfig) -> StableDiffusionPipeline:
    if not _DIFFUSERS_AVAILABLE:
        raise DiffusersBackendUnavailable(
            "diffusers and torch must be installed to use this backend"
        )
    dtype = config.dtype()
    logger.info("Loading diffusers pipeline '%s' (dtype=%s)", config.model, dtype)
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model,
        torch_dtype=dtype,
        revision=config.revision,
        variant=config.variant,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    if config.enable_xformers and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        try:  # pragma: no cover - optional optimisation
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning("Failed to enable xformers attention: %s", exc)
    pipeline.safety_checker = None  # type: ignore[attr-defined]
    return pipeline


__all__ = ["DiffusersImageBackend", "DiffusersBackendUnavailable"]
