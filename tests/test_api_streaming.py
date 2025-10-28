from __future__ import annotations

import base64
import io
import json
import wave
from typing import Dict, List

import numpy as np
from PIL import Image, ImageSequence
from fastapi.testclient import TestClient

from web_stable_diffusion.runtime.api import create_app


def _decode_png(encoded: str) -> Image.Image:
    payload = base64.b64decode(encoded.encode("ascii"))
    return Image.open(io.BytesIO(payload))


def _decode_wav(encoded: str) -> wave.Wave_read:
    payload = base64.b64decode(encoded.encode("ascii"))
    buffer = io.BytesIO(payload)
    return wave.open(buffer)


def test_streaming_api_emits_all_modalities() -> None:
    client = TestClient(create_app())

    with client.stream(
        "POST",
        "/generate",
        json={"prompt": "orchestral sunrise", "frames": 4, "audio_length": 128},
    ) as response:
        assert response.status_code == 200
        events: List[Dict[str, object]] = [
            json.loads(line) for line in response.iter_lines() if line
        ]

    modality_events = [event for event in events if event["type"] == "modality"]
    assert len(modality_events) == 4
    assert {event["modality"] for event in modality_events} == {
        "audio",
        "image",
        "volume",
        "video",
    }

    for event in modality_events:
        assert event["encoding"] in {"base64", "json"}
        if event["modality"] == "image":
            image = _decode_png(event["payload"])
            assert image.size == tuple(event["shape"][1::-1])
        elif event["modality"] == "audio":
            with _decode_wav(event["payload"]) as wav:
                assert wav.getnchannels() == 1
                assert wav.getsampwidth() == 2
            assert event["slices"]
            durations = [slice_["end"] - slice_["start"] for slice_ in event["slices"]]
            assert all(duration >= 0 for duration in durations)
        elif event["modality"] == "video":
            payload = base64.b64decode(event["payload"].encode("ascii"))
            gif = Image.open(io.BytesIO(payload))
            frames = list(ImageSequence.Iterator(gif))
            assert len(frames) == event["shape"][0]
        elif event["modality"] == "volume":
            projections = event.get("projections") or {}
            assert set(projections.keys()) == {"xy", "xz", "yz"}
            for encoded in projections.values():
                image = _decode_png(encoded)
                assert image.mode in {"L", "RGB"}
        else:
            payload = np.asarray(event["payload"])
            assert list(payload.shape) == event["shape"]
            assert payload.dtype == np.dtype(event["dtype"])
        assert 0.0 < event["progress"] <= 1.0

    complete = events[-1]
    assert complete["type"] == "complete"
    manifest = complete["manifest"]
    assert set(manifest["artifacts"].keys()) == {event["modality"] for event in modality_events}
    metrics = manifest["metadata"]["metrics"]
    assert metrics["wall_clock_s_total"] >= 0.0
    assert metrics["wall_clock_s_elapsed"] >= 0.0


def test_streaming_api_rejects_invalid_executor() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/generate",
        json={"prompt": "hello", "executor": "invalid"},
    )
    assert response.status_code == 422
