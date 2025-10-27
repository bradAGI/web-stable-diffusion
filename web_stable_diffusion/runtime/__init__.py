"""Runtime utilities for the omni-modal engine."""

from __future__ import annotations

import logging

__all__ = ["scheduler_runtime"]

try:  # pragma: no cover - optional dependency import guard
    from .scheduler_runtime import *  # type: ignore[F401,F403]
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when TVM is absent
    logging.getLogger(__name__).debug("Scheduler runtime unavailable: %s", exc)
