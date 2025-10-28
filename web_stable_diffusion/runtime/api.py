"""FastAPI application providing streaming omni-modal generation."""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from web_stable_diffusion.models.miniturbo_omnimodal import (
    BudgetExceededError,
    BundleEvent,
    CancellationToken,
    GenerationCancelled,
    OmniModalMiniturbo,
    ResourceBudget,
)


logger = logging.getLogger(__name__)


def _normalise_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _event_payload(event: BundleEvent, array: np.ndarray) -> Dict[str, Any]:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, payload=array)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "modality",
        "modality": event.key,
        "duration_s": event.duration_s,
        "completed": event.completed,
        "total": event.total,
        "progress": event.progress,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "backend": event.result.backend,
        "device": event.result.device,
        "metadata": _normalise_metadata(event.result.metadata),
        "payload": encoded,
    }


class BudgetSpec(BaseModel):
    memory_mb: Optional[float] = Field(None, ge=0)
    memory_bytes: Optional[int] = Field(None, ge=0)
    cpu_s: Optional[float] = Field(None, ge=0)

    def to_resource_budget(self) -> ResourceBudget:
        memory = self.memory_bytes
        if memory is None and self.memory_mb is not None:
            memory = int(self.memory_mb * 1024 * 1024)
        return ResourceBudget(
            max_memory_bytes=memory,
            max_cpu_seconds=self.cpu_s,
        )


class CancellationRegistry:
    """Tracks cancellation tokens shared across streaming responses."""

    def __init__(self) -> None:
        self._tokens: Dict[str, CancellationToken] = {}
        self._lock = threading.Lock()

    def create(self) -> Tuple[str, CancellationToken]:
        token_id = uuid.uuid4().hex
        token = CancellationToken()
        with self._lock:
            self._tokens[token_id] = token
        return token_id, token

    def get(self, token_id: str) -> Optional[CancellationToken]:
        with self._lock:
            return self._tokens.get(token_id)

    def release(self, token_id: str) -> None:
        with self._lock:
            self._tokens.pop(token_id, None)

    def cancel(self, token_id: str, reason: Optional[str] = None) -> CancellationToken:
        token = self.get(token_id)
        if token is None:
            raise KeyError(token_id)
        token.cancel(reason)
        return token


class EngineManager:
    """Coordinates engine health checks and self-healing restarts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._healthy = False
        self._last_failure: Optional[str] = None

    def ensure_ready(self) -> None:
        try:
            engine = OmniModalMiniturbo()
            engine.generate_image("healthcheck", resolution=32)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self._healthy = False
                self._last_failure = str(exc)
            logger.error("Engine health check failed: %s", exc)
            raise
        else:
            with self._lock:
                self._healthy = True
                self._last_failure = None

    def create_engine(self) -> OmniModalMiniturbo:
        with self._lock:
            healthy = self._healthy
        if not healthy:
            self.ensure_ready()
        return OmniModalMiniturbo()

    def mark_unhealthy(self, reason: str) -> None:
        with self._lock:
            self._healthy = False
            self._last_failure = reason

    def mark_healthy(self) -> None:
        with self._lock:
            self._healthy = True
            self._last_failure = None

    def health_status(self) -> Dict[str, Any]:
        with self._lock:
            return {"healthy": self._healthy, "last_failure": self._last_failure}


cancellation_registry = CancellationRegistry()
engine_manager = EngineManager()


def _error_response(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, (GenerationCancelled, BudgetExceededError)):
        payload = exc.as_dict()
    elif isinstance(exc, TimeoutError):
        payload = {"code": "timeout", "message": str(exc)}
    else:
        payload = {"code": "runtime_error", "message": str(exc)}
    payload.setdefault("code", "runtime_error")
    payload["type"] = "error"
    return payload


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    resolution: int = Field(256, ge=16, le=1024)
    frames: int = Field(16, ge=1, le=256)
    volume_size: int = Field(32, ge=8, le=256)
    audio_length: int = Field(2048, ge=16, le=65536)
    executor: str = Field("auto")
    max_workers: Optional[int] = Field(None, ge=1)
    timeout: Optional[float] = Field(None, gt=0)
    budgets: Optional[Dict[str, BudgetSpec]] = None

    @field_validator("executor")
    @classmethod
    def _validate_executor(cls, value: str) -> str:
        lowered = value.lower()
        if lowered not in {"auto", "thread", "process"}:
            raise ValueError("executor must be 'auto', 'thread', or 'process'")
        return lowered

    def generation_kwargs(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "resolution": self.resolution,
            "frames": self.frames,
            "volume_size": self.volume_size,
            "audio_length": self.audio_length,
            "executor": self.executor,
            "max_workers": self.max_workers,
            "timeout": self.timeout,
        }

    def budget_overrides(self) -> Dict[str, ResourceBudget]:
        if not self.budgets:
            return {}
        return {key: spec.to_resource_budget() for key, spec in self.budgets.items()}


def _stream_bundle_events(
    request: GenerateRequest,
    token_id: str,
    token: CancellationToken,
    budgets: Optional[Mapping[str, ResourceBudget]] = None,
) -> Iterator[str]:
    engine = engine_manager.create_engine()
    total_wall = 0.0
    artifacts: Dict[str, Any] = {}
    start = time.perf_counter()
    manifest: Optional[Dict[str, Any]] = None

    try:
        yield json.dumps({"type": "info", "cancel_token": token_id}) + "\n"
        for event in engine.generate_bundle_iter(
            **request.generation_kwargs(),
            budgets=budgets,
            cancellation=token,
        ):
            array = event.result.to_numpy()
            payload = _event_payload(event, array)
            artifacts[event.key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "backend": event.result.backend,
                "device": event.result.device,
                "metadata": _normalise_metadata(event.result.metadata),
            }
            total_wall += event.duration_s
            yield json.dumps(payload) + "\n"
    except (GenerationCancelled, BudgetExceededError, TimeoutError) as exc:
        logger.warning("Streaming generation aborted: %s", exc)
        yield json.dumps(_error_response(exc)) + "\n"
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Streaming generation failed: %s", exc, exc_info=True)
        engine_manager.mark_unhealthy(str(exc))
        yield json.dumps(_error_response(exc)) + "\n"
        return
    else:
        engine_manager.mark_healthy()
        manifest = {
            "artifacts": artifacts,
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cancel_token": token_id,
                "budgets": {
                    key: {
                        "memory_bytes": budget.max_memory_bytes,
                        "cpu_seconds": budget.max_cpu_seconds,
                    }
                    for key, budget in (budgets or {}).items()
                },
                "metrics": {
                    "wall_clock_s_total": total_wall,
                    "wall_clock_s_elapsed": time.perf_counter() - start,
                },
            },
        }

        benchmark = getattr(engine, "last_benchmark", None)
        if benchmark:
            manifest["metadata"]["metrics"]["scalability"] = _normalise_metadata(benchmark)

        yield json.dumps({"type": "complete", "manifest": manifest}) + "\n"
    finally:
        cancellation_registry.release(token_id)


def create_app() -> FastAPI:
    app = FastAPI(
        title="OmniModal Streaming API",
        description="Stream omni-modal artefacts generated by the NeuroSymbolic engine.",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            engine_manager.ensure_ready()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Startup health check failed: %s", exc)

    @app.post("/generate", response_class=StreamingResponse)
    def generate(request: GenerateRequest) -> StreamingResponse:
        token_id, token = cancellation_registry.create()
        budgets = request.budget_overrides()
        stream = _stream_bundle_events(request, token_id, token, budgets or None)
        return StreamingResponse(stream, media_type="application/jsonl")

    @app.post("/cancel/{token_id}")
    def cancel_generation(token_id: str) -> Dict[str, Any]:
        try:
            cancellation_registry.cancel(token_id, reason="api-request")
        except KeyError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail="unknown cancellation token") from exc
        return {"status": "cancelled", "token": token_id}

    @app.get("/healthz")
    def healthcheck() -> Dict[str, Any]:
        status = engine_manager.health_status()
        if not status["healthy"]:
            try:
                engine_manager.ensure_ready()
            except Exception as exc:
                status = engine_manager.health_status()
                raise HTTPException(
                    status_code=503,
                    detail={**status, "error": str(exc)},
                ) from exc
            status = engine_manager.health_status()
        return {"status": "ok", **status}

    return app


app = create_app()


__all__ = ["app", "create_app", "GenerateRequest"]
