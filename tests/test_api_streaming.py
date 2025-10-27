from __future__ import annotations

import base64
import io
import json
from typing import Dict, List

import numpy as np
from fastapi.testclient import TestClient

from web_stable_diffusion.runtime.api import create_app


def _decode_payload(encoded: str) -> np.ndarray:
    data = base64.b64decode(encoded.encode("ascii"))
    with np.load(io.BytesIO(data)) as archive:
        return archive["payload"]


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
        payload = _decode_payload(event["payload"])
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
