"""CLI entry point for the neuro-symbolic omni-modal generator."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Any, Dict

import numpy as np

from web_stable_diffusion.models.miniturbo_omnimodal import OmniModalMiniturbo


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
        )
    except Exception as exc:  # pragma: no cover - error paths exercised in tests via str(exc)
        logger.error("Failed to generate omni-modal bundle: %s", exc, exc_info=True)
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

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
