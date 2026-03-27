"""FastAPI application providing streaming omni-modal generation."""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
import wave
from collections import OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, Header, HTTPException, Request
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
from web_stable_diffusion.runtime.benchmarking import summarise_benchmark


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


def _encode_image_payload(array: np.ndarray) -> Tuple[str, str]:
    if array.ndim == 2:
        rgb = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        if array.shape[-1] == 1:
            rgb = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 3:
            rgb = array
        else:
            rgb = array[..., :3]
    else:
        raise ValueError(f"Unsupported image tensor shape {array.shape}")
    normalised = np.clip(rgb, 0.0, 1.0)
    uint8 = (normalised * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(uint8).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/png"


def _encode_audio_payload(array: np.ndarray, sample_rate: int = 22_050) -> Tuple[str, str, Iterable[Dict[str, float]]]:
    flattened = np.asarray(array).astype(np.float32).reshape(-1)
    peak = float(np.max(np.abs(flattened)) or 1.0)
    normalised = np.clip(flattened / peak, -1.0, 1.0)
    pcm = (normalised * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    segment_count = max(4, min(16, len(pcm) // 256))
    samples_per_segment = max(1, len(pcm) // segment_count)
    slices = []
    for index in range(segment_count):
        start_idx = index * samples_per_segment
        end_idx = min(len(pcm), (index + 1) * samples_per_segment)
        if end_idx <= start_idx:
            continue
        window = normalised[start_idx:end_idx]
        rms = float(np.sqrt(np.mean(window**2))) if window.size else 0.0
        slices.append(
            {
                "index": index,
                "start": start_idx / float(sample_rate),
                "end": end_idx / float(sample_rate),
                "volume": float(min(1.0, rms * 1.5)),
            }
        )

    return base64.b64encode(buffer.getvalue()).decode("ascii"), "audio/wav", slices


def _encode_volume_payload(array: np.ndarray) -> Dict[str, Any]:
    tensor = np.asarray(array).astype(np.float32)
    projections = {}
    axes = {"xy": 0, "xz": 1, "yz": 2}
    for name, axis in axes.items():
        projection = np.max(tensor, axis=axis)
        normalised = np.clip(projection / (projection.max() or 1.0), 0.0, 1.0)
        uint8 = (normalised * 255.0).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(uint8).save(buffer, format="PNG")
        projections[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "modality",
        "modality": "volume",
        "encoding": "base64",
        "mime_type": "image/png",
        "projections": projections,
    }


def _encode_video_payload(array: np.ndarray, *, duration_ms: int = 100) -> Tuple[str, str]:
    frames = np.asarray(array).astype(np.float32)
    if frames.ndim != 4:
        raise ValueError(f"Unsupported video tensor shape {frames.shape}")
    frame_images = []
    for frame in frames:
        data = np.clip(frame, 0.0, 1.0)
        uint8 = (data * 255.0).astype(np.uint8)
        frame_images.append(Image.fromarray(uint8))
    buffer = io.BytesIO()
    frame_images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frame_images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/gif"


def _event_payload(event: BundleEvent, array: np.ndarray) -> Dict[str, Any]:
    base = {
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
    }

    if event.key == "image":
        payload, mime = _encode_image_payload(array)
        base.update({"encoding": "base64", "mime_type": mime, "payload": payload})
    elif event.key == "audio":
        payload, mime, slices = _encode_audio_payload(array)
        base.update(
            {
                "encoding": "base64",
                "mime_type": mime,
                "payload": payload,
                "slices": _normalise_metadata(list(slices)),
            }
        )
    elif event.key == "video":
        payload, mime = _encode_video_payload(array)
        base.update({"encoding": "base64", "mime_type": mime, "payload": payload})
    elif event.key == "volume":
        base.update(_encode_volume_payload(array))
    else:
        base.update({"encoding": "json", "payload": array.tolist()})
    return base


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


class PersistentManifestRegistry:
    """SQLite-backed manifest store with an in-memory LRU cache."""

    def __init__(self, path: Path, max_entries: int = 512) -> None:
        self._lock = threading.Lock()
        self._path = Path(path)
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._max_entries = max_entries
        self._init_db()

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manifests (
                    token TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_json TEXT,
                    manifest_json TEXT,
                    error_json TEXT
                )
                """
            )

    def _update_cache(self, entry: Dict[str, Any]) -> None:
        token = entry["token"]
        self._cache[token] = entry
        self._cache.move_to_end(token)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    @staticmethod
    def _loads(value: Optional[str]) -> Dict[str, Any]:
        if not value:
            return {}
        return json.loads(value)

    def _row_to_entry(self, row: sqlite3.Row) -> Dict[str, Any]:
        entry = {
            "token": row["token"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "request": self._loads(row["request_json"]),
        }
        manifest = self._loads(row["manifest_json"])
        if manifest:
            entry["manifest"] = manifest
        error = self._loads(row["error_json"])
        if error:
            entry["error"] = error
        return entry

    def reconfigure(self, path: Path) -> None:
        new_path = Path(path)
        with self._lock:
            if new_path == self._path:
                return
            self._path = new_path
            self._cache.clear()
            self._init_db()

    def start(self, token_id: str, request: Mapping[str, Any]) -> None:
        timestamp = self._timestamp()
        entry = {
            "token": token_id,
            "status": "pending",
            "created_at": timestamp,
            "updated_at": timestamp,
            "request": dict(request),
        }
        payload = json.dumps(entry["request"])
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO manifests(token, status, created_at, updated_at, request_json, manifest_json, error_json)
                VALUES(?, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(token) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    request_json=excluded.request_json,
                    manifest_json=NULL,
                    error_json=NULL
                """,
                (token_id, entry["status"], timestamp, timestamp, payload),
            )
            self._update_cache(entry)

    def record_complete(self, token_id: str, manifest: Mapping[str, Any]) -> None:
        timestamp = self._timestamp()
        manifest_json = json.dumps(dict(manifest))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO manifests(token, status, created_at, updated_at, request_json, manifest_json, error_json)
                VALUES(?, 'complete', ?, ?, '{}', ?, NULL)
                ON CONFLICT(token) DO UPDATE SET
                    status='complete',
                    updated_at=excluded.updated_at,
                    manifest_json=excluded.manifest_json,
                    error_json=NULL
                """,
                (token_id, timestamp, timestamp, manifest_json),
            )
            row = conn.execute(
                "SELECT * FROM manifests WHERE token = ?",
                (token_id,),
            ).fetchone()
            assert row is not None
            entry = self._row_to_entry(row)
            self._update_cache(entry)

    def record_error(self, token_id: str, payload: Mapping[str, Any]) -> None:
        timestamp = self._timestamp()
        error_json = json.dumps(dict(payload))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO manifests(token, status, created_at, updated_at, request_json, manifest_json, error_json)
                VALUES(?, 'error', ?, ?, '{}', NULL, ?)
                ON CONFLICT(token) DO UPDATE SET
                    status='error',
                    updated_at=excluded.updated_at,
                    error_json=excluded.error_json,
                    manifest_json=NULL
                """,
                (token_id, timestamp, timestamp, error_json),
            )
            row = conn.execute(
                "SELECT * FROM manifests WHERE token = ?",
                (token_id,),
            ).fetchone()
            assert row is not None
            entry = self._row_to_entry(row)
            self._update_cache(entry)

    def get(self, token_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cached = self._cache.get(token_id)
            if cached is not None:
                return deepcopy(cached)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM manifests WHERE token = ?",
                (token_id,),
            ).fetchone()
        if row is None:
            return None
        entry = self._row_to_entry(row)
        with self._lock:
            self._update_cache(entry)
            return deepcopy(entry)

    def delete(self, token_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM manifests WHERE token = ?", (token_id,))
            deleted = cursor.rowcount > 0
            self._cache.pop(token_id, None)
            return deleted

    def list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM manifests ORDER BY updated_at DESC"
        params: Tuple[Any, ...] = ()
        if limit is not None and limit >= 0:
            sql += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_entry(row) for row in rows]


def _default_state_dir() -> Path:
    override = os.getenv("OMNIMODAL_STATE_DIR")
    if override:
        return Path(override)
    return Path.cwd() / "log_db"


def _manifest_db_path() -> Path:
    override = os.getenv("OMNIMODAL_MANIFEST_DB")
    if override:
        return Path(override)
    return _default_state_dir() / "manifests.db"


@dataclass
class SecurityConfig:
    api_keys: List[str]
    rate_limit: int
    rate_period: float

    @classmethod
    def from_environment(cls) -> "SecurityConfig":
        raw_keys = os.getenv("OMNIMODAL_API_KEYS", "")
        api_keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        rate_limit = int(os.getenv("OMNIMODAL_RATE_LIMIT", "60"))
        rate_period = float(os.getenv("OMNIMODAL_RATE_PERIOD", "60"))
        return cls(api_keys=api_keys, rate_limit=rate_limit, rate_period=rate_period)


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__("rate limit exceeded")
        self.retry_after = retry_after


class RateLimiter:
    """Simple fixed-window rate limiter keyed by caller identity."""

    def __init__(self, limit: int, period: float) -> None:
        self.limit = limit
        self.period = period
        self._lock = threading.Lock()
        self._records: Dict[str, deque[float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def check(self, identity: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            history = self._records.setdefault(identity, deque())
            while history and now - history[0] >= self.period:
                history.popleft()
            if len(history) >= self.limit:
                retry_after = max(0.0, self.period - (now - history[0]))
                raise RateLimitExceeded(retry_after)
            history.append(now)


def build_api_key_dependency(config: SecurityConfig):
    async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if not config.api_keys:
            return
        if not x_api_key or x_api_key not in config.api_keys:
            raise HTTPException(status_code=401, detail={"message": "invalid api key"})

    return require_api_key


def build_rate_limit_dependency(limiter: RateLimiter):
    async def enforce_rate_limit(
        request: Request,
        x_api_key: Optional[str] = Header(default=None),
    ) -> None:
        identity = x_api_key or (request.client.host if request.client else "anonymous")
        try:
            limiter.check(identity)
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail={"message": "rate limit exceeded", "retry_after": round(exc.retry_after, 3)},
            ) from exc

    return enforce_rate_limit


cancellation_registry = CancellationRegistry()
engine_manager = EngineManager()
manifest_registry = PersistentManifestRegistry(_manifest_db_path())


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


class Img2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)
    strength: float = Field(0.75, ge=0.0, le=1.0)
    negative_prompt: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    resolution: int = Field(512, ge=16, le=1024)
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
        manifest_registry.start(token_id, request.model_dump(mode="json"))
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
        error_payload = _error_response(exc)
        manifest_registry.record_error(token_id, error_payload)
        yield json.dumps(error_payload) + "\n"
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Streaming generation failed: %s", exc, exc_info=True)
        engine_manager.mark_unhealthy(str(exc))
        error_payload = _error_response(exc)
        manifest_registry.record_error(token_id, error_payload)
        yield json.dumps(error_payload) + "\n"
        return
    else:
        engine_manager.mark_healthy()
        elapsed = time.perf_counter() - start
        metrics: Dict[str, Any] = {
            "wall_clock_s_total": total_wall,
            "wall_clock_s_elapsed": elapsed,
        }
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
                "metrics": metrics,
            },
        }

        benchmark = getattr(engine, "last_benchmark", None)
        if benchmark:
            metrics["scheduler"] = summarise_benchmark(benchmark)
            metrics["scalability"] = _normalise_metadata(benchmark)

        manifest_registry.record_complete(token_id, manifest)
        yield json.dumps({"type": "complete", "manifest": manifest}) + "\n"
    finally:
        cancellation_registry.release(token_id)


def create_app() -> FastAPI:
    app = FastAPI(
        title="OmniModal Streaming API",
        description="Stream omni-modal artefacts generated by the NeuroSymbolic engine.",
        version="0.1.0",
    )

    manifest_registry.reconfigure(_manifest_db_path())
    security_config = SecurityConfig.from_environment()
    rate_limiter = RateLimiter(security_config.rate_limit, security_config.rate_period)
    api_key_dependency = build_api_key_dependency(security_config)
    rate_limit_dependency = build_rate_limit_dependency(rate_limiter)

    def _secured_dependencies() -> List[Depends]:
        return [Depends(api_key_dependency), Depends(rate_limit_dependency)]

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            engine_manager.ensure_ready()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Startup health check failed: %s", exc)

    @app.post("/generate", response_class=StreamingResponse, dependencies=_secured_dependencies())
    def generate(request: GenerateRequest) -> StreamingResponse:
        token_id, token = cancellation_registry.create()
        budgets = request.budget_overrides()
        stream = _stream_bundle_events(request, token_id, token, budgets or None)
        return StreamingResponse(stream, media_type="application/jsonl")

    @app.post("/img2img", dependencies=_secured_dependencies())
    async def img2img(request: Img2ImgRequest) -> Dict[str, Any]:
        from web_stable_diffusion.models.backends.diffusers_image import DiffusersImageBackend

        backend = DiffusersImageBackend.from_environment() or DiffusersImageBackend.from_default()
        if backend is None:
            raise HTTPException(status_code=503, detail="Diffusers backend unavailable")

        try:
            raw_bytes = base64.b64decode(request.image_base64)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid base64 image: {exc}") from exc

        input_image = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))

        result = backend.img2img(
            prompt=request.prompt,
            input_image=input_image,
            strength=request.strength,
            negative_prompt=request.negative_prompt,
        )

        output_b64, mime = _encode_image_payload(result["array"])
        return {
            "image_base64": output_b64,
            "mime_type": mime,
            "metadata": _normalise_metadata(result["metadata"]),
        }

    @app.post("/cancel/{token_id}", dependencies=_secured_dependencies())
    def cancel_generation(token_id: str) -> Dict[str, Any]:
        try:
            cancellation_registry.cancel(token_id, reason="api-request")
        except KeyError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail="unknown cancellation token") from exc
        return {"status": "cancelled", "token": token_id}

    @app.get("/manifests/{token_id}", dependencies=_secured_dependencies())
    def get_manifest(token_id: str) -> Dict[str, Any]:
        record = manifest_registry.get(token_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown manifest token")
        return record

    @app.delete("/manifests/{token_id}", dependencies=_secured_dependencies())
    def delete_manifest(token_id: str) -> Dict[str, Any]:
        removed = manifest_registry.delete(token_id)
        if not removed:
            raise HTTPException(status_code=404, detail="unknown manifest token")
        return {"status": "deleted", "token": token_id}

    @app.get("/manifests", dependencies=_secured_dependencies())
    def list_manifests(limit: int = 20) -> Dict[str, Any]:
        if limit < 0:
            raise HTTPException(status_code=422, detail={"message": "limit must be non-negative"})
        return {"items": manifest_registry.list(limit=limit)}

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
