from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from web_stable_diffusion.cli import omnimodal as omnimodal_cli
from web_stable_diffusion.models.miniturbo_omnimodal import OmniModalMiniturbo
from web_stable_diffusion.models.modal_generators import BACKEND_NUMPY, DeviceSpec


def _numpy_engine() -> OmniModalMiniturbo:
    """Utility returning an engine forced onto the NumPy backend."""

    return OmniModalMiniturbo(device_spec=DeviceSpec(BACKEND_NUMPY, "cpu"), diffusion_steps=2)


def test_generate_bundle_populates_metrics_thread_executor():
    engine = _numpy_engine()
    bundle = engine.generate_bundle(
        prompt="luminous waterfalls",
        resolution=32,
        frames=4,
        volume_size=12,
        audio_length=64,
        executor="thread",
        max_workers=2,
    )

    assert set(bundle) == {"audio", "image", "volume", "video"}
    for result in bundle.values():
        metrics = result.metadata.get("metrics", {})
        assert "wall_clock_s" in metrics
        assert metrics["wall_clock_s"] >= 0.0


def test_generate_bundle_process_executor_numpy_backend():
    engine = _numpy_engine()
    bundle = engine.generate_bundle(
        prompt="starlit canyon",
        resolution=24,
        frames=3,
        volume_size=10,
        audio_length=32,
        executor="process",
    )

    assert set(bundle) == {"audio", "image", "volume", "video"}


def test_generate_bundle_honours_timeout(monkeypatch):
    engine = _numpy_engine()

    original_generate_audio = engine.generate_audio

    def slow_generate_audio(*args, **kwargs):
        time.sleep(0.2)
        return original_generate_audio(*args, **kwargs)

    monkeypatch.setattr(engine, "generate_audio", slow_generate_audio)

    with pytest.raises(TimeoutError):
        engine.generate_bundle(
            prompt="glacial sunrise",
            resolution=24,
            frames=2,
            volume_size=8,
            audio_length=32,
            executor="thread",
            max_workers=1,
            timeout=0.05,
        )


def test_generate_bundle_scales_workers_to_cpu(monkeypatch):
    engine = _numpy_engine()

    monkeypatch.setattr("os.cpu_count", lambda: 2)

    engine.generate_bundle(
        prompt="quantum dunes",
        resolution=32,
        frames=4,
        volume_size=12,
        audio_length=64,
        executor="thread",
    )

    benchmark = engine.last_benchmark
    assert benchmark is not None
    assert benchmark["max_workers"] == 2
    assert benchmark["requested_max_workers"] is None
    assert benchmark["cpu_count"] == 2


def test_process_executor_reserves_parent_cpu(monkeypatch):
    engine = _numpy_engine()

    monkeypatch.setattr("os.cpu_count", lambda: 3)

    engine.generate_bundle(
        prompt="stellar echoes",
        resolution=32,
        frames=4,
        volume_size=12,
        audio_length=64,
        executor="process",
    )

    benchmark = engine.last_benchmark
    assert benchmark is not None
    assert benchmark["executor"] == "process"
    assert benchmark["max_workers"] == 2
    assert benchmark["cpu_count"] == 3


def test_cli_generates_manifest(tmp_path: pathlib.Path):
    manifest_dir = tmp_path / "artifacts"
    args = [
        "--prompt",
        "cerulean tides",
        "--output",
        str(manifest_dir),
        "--executor",
        "thread",
        "--resolution",
        "32",
        "--frames",
        "4",
        "--volume-size",
        "8",
        "--audio-length",
        "64",
        "--log-level",
        "ERROR",
    ]

    exit_code = omnimodal_cli.main(args)
    assert exit_code == 0

    manifest_path = manifest_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())

    assert sorted(data["artifacts"]) == ["audio", "image", "video", "volume"]
    assert data["metadata"]["executor"] == "thread"
    assert "metrics" in data["metadata"]
