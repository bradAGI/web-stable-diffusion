from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROMPT = "Parallel dawn over mountains"


def test_cli_generates_bundle(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "web_stable_diffusion.cli.omnimodal",
            "--prompt",
            PROMPT,
            "--output",
            str(tmp_path),
            "--manifest-name",
            manifest.name,
            "--resolution",
            "32",
            "--frames",
            "4",
            "--volume-size",
            "16",
            "--audio-length",
            "128",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert manifest.exists(), result.stdout
    data = json.loads(manifest.read_text())
    assert set(data["artifacts"]) == {"audio", "image", "video", "volume"}
    benchmark = data["metadata"].get("benchmark")
    assert benchmark is not None, "benchmark metadata missing"
    assert benchmark.get("completed") == 4
    timeline = benchmark.get("timeline", [])
    assert any(event.get("type") == "dispatch" for event in timeline)
    assert any(event.get("type") == "complete" for event in timeline)
    resource_summary = benchmark.get("resource_summary", {})
    assert resource_summary.get("completed_modalities") == 4

    audio_npz = Path(data["artifacts"]["audio"]["file"])
    assert audio_npz.exists()
    audio = np.load(audio_npz)["payload"]
    assert audio.shape == (128,)

    image_npz = Path(data["artifacts"]["image"]["file"])
    image = np.load(image_npz)["payload"]
    assert image.shape == (32, 32, 3)

    video_npz = Path(data["artifacts"]["video"]["file"])
    video = np.load(video_npz)["payload"]
    assert video.shape[0] == 4

    volume_npz = Path(data["artifacts"]["volume"]["file"])
    volume = np.load(volume_npz)["payload"]
    assert volume.shape == (16, 16, 16)
