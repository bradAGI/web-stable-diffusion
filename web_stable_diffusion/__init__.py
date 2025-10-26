"""Top-level package exports for :mod:`web_stable_diffusion`."""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["runtime", "trace", "utils", "OmniModalMiniturbo", "MetaLogic"]


def __getattr__(name: str) -> Any:
    if name in {"runtime", "trace", "utils"}:
        return import_module(f"web_stable_diffusion.{name}")
    if name in {"OmniModalMiniturbo", "MetaLogic"}:
        module = import_module("web_stable_diffusion.models.miniturbo_omnimodal")
        return getattr(module, name)
    raise AttributeError(f"module 'web_stable_diffusion' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from . import runtime, trace, utils  # noqa: F401  (re-export for type checking)
    from .models.miniturbo_omnimodal import MetaLogic, OmniModalMiniturbo
