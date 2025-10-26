"""CLI entry point for the neuro-symbolic omni-modal generator."""
from __future__ import annotations

import argparse
import json
import pathlib
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
    return parser


def _bundle_to_manifest(bundle: Dict[str, Any], base_dir: pathlib.Path) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"artifacts": {}, "metadata": {"base_dir": str(base_dir)}}
    for key, result in bundle.items():
        array = result.to_numpy()
        artifact_path = base_dir / f"{key}.npz"
        file_path = _serialise_payload(array, artifact_path)
        manifest["artifacts"][key] = {
            "file": file_path,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "backend": result.backend,
            "device": result.device,
            "metadata": result.metadata,
        }
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    engine = OmniModalMiniturbo()
    bundle = engine.generate_bundle(
        prompt=args.prompt,
        resolution=args.resolution,
        frames=args.frames,
        volume_size=args.volume_size,
        audio_length=args.audio_length,
    )

    output_dir: pathlib.Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _bundle_to_manifest(bundle, output_dir)
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps({"manifest": str(manifest_path), "artifacts": list(manifest["artifacts"].keys())}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
