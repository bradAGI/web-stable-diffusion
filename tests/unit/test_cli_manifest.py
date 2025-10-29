from __future__ import annotations

from pathlib import Path

import numpy as np

from web_stable_diffusion.cli.omnimodal import _bundle_to_manifest
from web_stable_diffusion.models.modal_generators import DeviceAwareResult


def test_bundle_to_manifest_embeds_scheduler_summary(tmp_path: Path) -> None:
    bundle = {
        "image": DeviceAwareResult(
            backend="numpy",
            device="cpu",
            data=np.zeros((2, 2, 3), dtype=np.float32),
            metadata={"metrics": {"wall_clock_s": 0.5}},
        )
    }
    benchmark = {
        "executor": "thread",
        "max_workers": 1,
        "requested_executor": "auto",
        "requested_max_workers": None,
        "completed": 1,
        "total": 1,
        "timeline": [],
        "tasks": {"image": {"dispatch_seq": 0}},
    }

    manifest = _bundle_to_manifest(
        bundle,
        tmp_path,
        executor="thread",
        device_backend="numpy",
        benchmark=benchmark,
    )

    artifact_path = tmp_path / "image.npz"
    assert artifact_path.exists()

    metrics = manifest["metadata"]["metrics"]
    assert metrics["wall_clock_s_total"] == 0.5
    scheduler = metrics["scheduler"]
    assert scheduler["executor"] == "thread"
    assert scheduler["worker_count"] == 1
    assert scheduler["modalities_completed"] == 1
