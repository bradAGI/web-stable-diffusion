"""Diffusers-powered image backend for the omni-modal engine."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from diffusers import (
        DPMSolverMultistepScheduler,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )

    _DIFFUSERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StableDiffusionPipeline = None  # type: ignore[assignment]
    StableDiffusionXLPipeline = None  # type: ignore[assignment]
    StableDiffusionImg2ImgPipeline = None  # type: ignore[assignment]
    StableDiffusionXLImg2ImgPipeline = None  # type: ignore[assignment]
    DPMSolverMultistepScheduler = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    _DIFFUSERS_AVAILABLE = False


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Known SDXL model identifiers (prefix match)
_SDXL_MODEL_PREFIXES = (
    "stabilityai/stable-diffusion-xl",
    "stabilityai/sdxl",
)


def _is_sdxl_model(model_id: str) -> bool:
    """Auto-detect whether a model ID refers to an SDXL pipeline."""
    lower = model_id.lower()
    for prefix in _SDXL_MODEL_PREFIXES:
        if lower.startswith(prefix):
            return True
    if "sdxl" in lower or "xl" in lower.split("/")[-1]:
        return True
    return False


class DiffusersBackendUnavailable(RuntimeError):
    """Raised when the diffusers integration cannot be initialised."""


@dataclass
class DiffusersConfig:
    """Configuration describing how to initialise the pipeline."""

    model: str
    torch_dtype: str = "float16"
    revision: Optional[str] = None
    variant: Optional[str] = None
    enable_xformers: bool = True
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    negative_prompt: Optional[str] = None
    lora_weights: Optional[str] = None  # HuggingFace model ID or local path

    @classmethod
    def from_environment(cls) -> Optional["DiffusersConfig"]:
        model = os.getenv("OMNIMODAL_DIFFUSERS_MODEL")
        if not model:
            return None
        dtype = os.getenv("OMNIMODAL_DIFFUSERS_DTYPE", "float16")
        revision = os.getenv("OMNIMODAL_DIFFUSERS_REVISION") or None
        variant = os.getenv("OMNIMODAL_DIFFUSERS_VARIANT") or None
        guidance = float(os.getenv("OMNIMODAL_DIFFUSERS_GUIDANCE", "7.5"))
        steps = int(os.getenv("OMNIMODAL_DIFFUSERS_STEPS", "25"))
        enable_xformers = os.getenv("OMNIMODAL_DIFFUSERS_ENABLE_XFORMERS", "1") not in {"0", "false", "False"}
        negative_prompt = os.getenv("OMNIMODAL_DIFFUSERS_NEGATIVE_PROMPT") or None
        lora_weights = os.getenv("OMNIMODAL_DIFFUSERS_LORA") or None
        return cls(
            model=model,
            torch_dtype=dtype,
            revision=revision,
            variant=variant,
            enable_xformers=enable_xformers,
            guidance_scale=guidance,
            num_inference_steps=steps,
            negative_prompt=negative_prompt,
            lora_weights=lora_weights,
        )

    @classmethod
    def default(cls) -> "DiffusersConfig":
        """Return a config using the default SDXL model."""
        return cls(model=DEFAULT_MODEL)

    def dtype(self) -> Optional["torch.dtype"]:
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


# Module-level pipeline cache (replaces @lru_cache since dataclass is not hashable)
_cached_pipelines: Dict[str, Any] = {}
_cached_pipeline_model: Optional[str] = None


class DiffusersImageBackend:
    """Wrapper around Stable Diffusion / SDXL pipelines for image synthesis."""

    def __init__(self, pipelines: Dict[str, Any], config: DiffusersConfig) -> None:
        if not _DIFFUSERS_AVAILABLE:
            raise DiffusersBackendUnavailable(
                "diffusers and torch must be installed to use this backend"
            )
        self._pipeline = pipelines["txt2img"]
        self._img2img_pipeline = pipelines["img2img"]
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
        return cls(_initialise_pipelines(config), config)

    @classmethod
    def from_default(cls) -> Optional["DiffusersImageBackend"]:
        """Create a backend using the default SDXL model."""
        if not _DIFFUSERS_AVAILABLE:
            return None
        config = DiffusersConfig.default()
        return cls(_initialise_pipelines(config), config)

    @property
    def config(self) -> DiffusersConfig:
        return self._config

    def generate(
        self,
        prompt: str,
        resolution: int,
        *,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, Any], None]] = None,
    ) -> Dict[str, Any]:
        """Generate an image for the prompt at the requested resolution."""

        neg = negative_prompt or self._config.negative_prompt
        options: Dict[str, Any] = {
            "height": resolution,
            "width": resolution,
            "num_inference_steps": self._config.num_inference_steps,
            "guidance_scale": self._config.guidance_scale,
        }
        if neg:
            options["negative_prompt"] = neg
        if progress_callback is not None:
            options["callback_on_step_end"] = progress_callback
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
        if neg:
            metadata["negative_prompt"] = neg
        return {"array": array, "metadata": metadata}

    def img2img(
        self,
        prompt: str,
        input_image: np.ndarray,
        *,
        strength: float = 0.75,
        negative_prompt: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, Any], None]] = None,
    ) -> Dict[str, Any]:
        """Generate an image from a prompt + input image."""
        from PIL import Image as PILImage

        # Convert numpy array to PIL Image
        if input_image.dtype in (np.float32, np.float64):
            input_image = (np.clip(input_image, 0, 1) * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(input_image)

        neg = negative_prompt or self._config.negative_prompt
        options: Dict[str, Any] = {
            "image": pil_image,
            "strength": strength,
            "num_inference_steps": self._config.num_inference_steps,
            "guidance_scale": self._config.guidance_scale,
        }
        if neg:
            options["negative_prompt"] = neg
        if progress_callback:
            options["callback_on_step_end"] = progress_callback

        result = self._img2img_pipeline(prompt, **options)
        image = result.images[0]
        array = np.asarray(image).astype(np.float32) / 255.0
        metadata = {
            "shape": array.shape,
            "prompt": prompt,
            "backend": "diffusers",
            "mode": "img2img",
            "strength": strength,
            "model": {"id": self._config.model},
        }
        if neg:
            metadata["negative_prompt"] = neg
        return {"array": array, "metadata": metadata}


def _initialise_pipelines(config: DiffusersConfig) -> Dict[str, Any]:
    global _cached_pipelines, _cached_pipeline_model

    if not _DIFFUSERS_AVAILABLE:
        raise DiffusersBackendUnavailable(
            "diffusers and torch must be installed to use this backend"
        )

    # Return cached pipelines if same model
    if _cached_pipelines and _cached_pipeline_model == config.model:
        return _cached_pipelines

    dtype = config.dtype()
    # Auto-select pipeline class based on model ID
    use_sdxl = _is_sdxl_model(config.model)
    pipeline_cls = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline
    img2img_cls = StableDiffusionXLImg2ImgPipeline if use_sdxl else StableDiffusionImg2ImgPipeline
    logger.info(
        "Loading %s pipeline '%s' (dtype=%s)",
        "SDXL" if use_sdxl else "SD 1.5",
        config.model,
        dtype,
    )

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
    }
    if config.revision:
        load_kwargs["revision"] = config.revision
    if config.variant:
        load_kwargs["variant"] = config.variant

    pipeline = pipeline_cls.from_pretrained(config.model, **load_kwargs)

    # Use DPMSolver++ scheduler for faster inference
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    if config.enable_xformers and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        try:  # pragma: no cover - optional optimisation
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning("Failed to enable xformers attention: %s", exc)

    if not use_sdxl:
        pipeline.safety_checker = None  # type: ignore[attr-defined]

    # Load LoRA weights if configured
    if config.lora_weights:
        pipeline.load_lora_weights(config.lora_weights)
        logger.info("Loaded LoRA weights from '%s'", config.lora_weights)

    # Optional torch.compile for UNet acceleration
    if os.getenv("OMNIMODAL_TORCH_COMPILE", "0") in {"1", "true", "True"}:
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
            logger.info("Applied torch.compile to UNet")
        except Exception as exc:
            logger.warning("torch.compile failed: %s", exc)

    # Build img2img pipeline from the same components
    img2img_pipeline = img2img_cls(**pipeline.components)

    pipelines = {"txt2img": pipeline, "img2img": img2img_pipeline}

    # Cache the pipelines
    _cached_pipelines = pipelines
    _cached_pipeline_model = config.model

    return pipelines


__all__ = ["DiffusersImageBackend", "DiffusersBackendUnavailable", "DiffusersConfig", "DEFAULT_MODEL"]
