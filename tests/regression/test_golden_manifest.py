from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from web_stable_diffusion.models.miniturbo_omnimodal import OmniModalMiniturbo
from web_stable_diffusion.models.modal_generators import BACKEND_NUMPY, DeviceSpec


GOLDEN_MANIFEST = Path(__file__).resolve().parents[1] / "fixtures" / "golden_manifests" / "celestial_gardens.json"


@pytest.mark.parametrize("manifest_path", [GOLDEN_MANIFEST])
def test_bundle_matches_golden_manifest_within_tolerance(manifest_path: Path) -> None:
    """Ensure real model outputs stay close to the golden manifest."""

    payload = json.loads(manifest_path.read_text())
    tolerances = payload["tolerances"]["statistics"]

    engine = OmniModalMiniturbo(
        device_spec=DeviceSpec(BACKEND_NUMPY, "cpu"),
        diffusion_steps=payload["configuration"]["diffusion_steps"],
    )
    bundle = engine.generate_bundle(
        prompt=payload["prompt"],
        resolution=payload["configuration"]["resolution"],
        frames=payload["configuration"]["frames"],
        volume_size=payload["configuration"]["volume_size"],
        audio_length=payload["configuration"]["audio_length"],
        executor="thread",
        max_workers=payload["configuration"]["max_workers"],
    )

    for key, golden_artifact in payload["artifacts"].items():
        assert key in bundle, f"Missing modality {key} in generated bundle"
        result = bundle[key]
        array = result.to_numpy()

        assert list(array.shape) == golden_artifact["shape"], f"Shape regression for {key}"
        assert str(array.dtype) == golden_artifact["dtype"], f"dtype regression for {key}"

        stats = {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

        for metric, expected in golden_artifact["statistics"].items():
            tolerance = tolerances[metric]
            assert np.isclose(
                stats[metric],
                expected,
                atol=tolerance,
                rtol=0.0,
            ), f"{key} {metric} drifted by {abs(stats[metric] - expected)}"
