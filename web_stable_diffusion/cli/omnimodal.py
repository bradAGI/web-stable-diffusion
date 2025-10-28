"""CLI entry point for the neuro-symbolic omni-modal generator."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import signal
import sys
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

from web_stable_diffusion.models.miniturbo_omnimodal import (
    BudgetExceededError,
    CancellationToken,
    GenerationCancelled,
    OmniModalMiniturbo,
    ResourceBudget,
)


def _serialise_payload(array: np.ndarray, output_path: pathlib.Path) -> str:
    """Persist ``array`` to ``output_path`` using ``.npz`` compression."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, payload=array)
    return str(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Prompt text that conditions every modality")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.cwd() / "omnimodal_outputs",
        help="Directory where the generated artefacts will be written",
    )
    parser.add_argument("--resolution", type=int, default=256, help="Square resolution for the generated image")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames for generated video")
    parser.add_argument("--volume-size", type=int, default=32, help="Edge length of the volumetric tensor")
    parser.add_argument("--audio-length", type=int, default=2048, help="Sample length for generated audio waveform")
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Filename of the JSON manifest relative to the output directory",
    )
    parser.add_argument(
        "--executor",
        choices=["auto", "thread", "process"],
        default="auto",
        help="Execution backend used for parallel generation",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional override for the executor worker count",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for bundle generation",
    )
    parser.add_argument(
        "--budget-file",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file defining per-modality resource budgets",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity for the CLI run",
    )
    return parser


def _bundle_to_manifest(
    bundle: Dict[str, Any],
    base_dir: pathlib.Path,
    *,
    executor: str,
    device_backend: str,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "artifacts": {},
        "metadata": {
            "base_dir": str(base_dir),
            "executor": executor,
            "device_backend": device_backend,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    total_duration = 0.0
    for key, result in bundle.items():
        array = result.to_numpy()
        artifact_path = base_dir / f"{key}.npz"
        file_path = _serialise_payload(array, artifact_path)
        entry: Dict[str, Any] = {
            "file": file_path,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "backend": result.backend,
            "device": result.device,
            "metadata": result.metadata,
        }
        metrics = result.metadata.get("metrics") if isinstance(result.metadata, dict) else None
        if metrics and "wall_clock_s" in metrics:
            total_duration += float(metrics["wall_clock_s"])
        manifest["artifacts"][key] = entry
    if total_duration:
        manifest["metadata"]["metrics"] = {"wall_clock_s_total": total_duration}
    return manifest


def _load_budget_file(path: pathlib.Path) -> Dict[str, ResourceBudget]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"failed to parse budget file: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("budget file must contain a JSON object")

    budgets: Dict[str, ResourceBudget] = {}
    for modality, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"budget for '{modality}' must be an object")
        memory_bytes: Optional[int] = None
        if "memory_bytes" in spec and spec["memory_bytes"] is not None:
            memory_bytes = int(spec["memory_bytes"])
        elif "memory_mb" in spec and spec["memory_mb"] is not None:
            memory_bytes = int(float(spec["memory_mb"]) * 1024 * 1024)
        cpu_seconds: Optional[float] = None
        if "cpu_s" in spec and spec["cpu_s"] is not None:
            cpu_seconds = float(spec["cpu_s"])
        budgets[modality] = ResourceBudget(
            max_memory_bytes=memory_bytes,
            max_cpu_seconds=cpu_seconds,
        )
    return budgets


def _error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, (GenerationCancelled, BudgetExceededError)):
        return {"error": exc.as_dict()}
    if isinstance(exc, TimeoutError):
        return {"error": {"code": "timeout", "message": str(exc)}}
    return {"error": {"code": "runtime_error", "message": str(exc)}}


def _install_signal_cancellation(token: CancellationToken, logger: logging.Logger) -> Mapping[int, Any]:
    previous_handlers: Dict[int, Any] = {}

    def _handle_signal(signum: int, _frame: Any) -> None:
        if not token.cancelled:
            logger.warning("Received signal %s, cancelling generation", signum)
            token.cancel(reason=f"signal:{signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except Exception:  # pragma: no cover - platform specific
            previous_handlers[sig] = None
    return previous_handlers


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for sig, handler in previous.items():
        if handler is None:
            continue
        try:
            signal.signal(sig, handler)
        except Exception:  # pragma: no cover - platform specific
            pass


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    if args.max_workers is not None and args.max_workers <= 0:
        parser.error("--max-workers must be a positive integer")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    engine = OmniModalMiniturbo()
    budgets: Optional[Dict[str, ResourceBudget]] = None
    if args.budget_file is not None:
        try:
            budgets = _load_budget_file(args.budget_file)
        except ValueError as exc:
            parser.error(str(exc))

    token = CancellationToken()
    previous_handlers = _install_signal_cancellation(token, logger)
    try:
        bundle = engine.generate_bundle(
            prompt=args.prompt,
            resolution=args.resolution,
            frames=args.frames,
            volume_size=args.volume_size,
            audio_length=args.audio_length,
            executor=args.executor,
            max_workers=args.max_workers,
            timeout=args.timeout,
            budgets=budgets,
            cancellation=token,
        )
    except GenerationCancelled as exc:
        logger.warning("Generation cancelled: %s", exc.reason)
        print(json.dumps(_error_payload(exc)), file=sys.stderr)
        return 130
    except BudgetExceededError as exc:
        logger.error("Resource budget exceeded: %s", exc, exc_info=True)
        print(json.dumps(_error_payload(exc)), file=sys.stderr)
        return 2
    except TimeoutError as exc:
        logger.error("Generation timed out: %s", exc, exc_info=True)
        print(json.dumps(_error_payload(exc)), file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover - error paths exercised in tests via str(exc)
        logger.error("Failed to generate omni-modal bundle: %s", exc, exc_info=True)
        print(json.dumps(_error_payload(exc)), file=sys.stderr)
        return 1
    finally:
        _restore_signal_handlers(previous_handlers)

    output_dir: pathlib.Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _bundle_to_manifest(
        bundle,
        output_dir,
        executor=args.executor,
        device_backend=engine.device_spec.backend,
    )
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps({"manifest": str(manifest_path), "artifacts": list(manifest["artifacts"].keys())}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
