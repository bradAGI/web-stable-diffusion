from __future__ import annotations

import base64
import io
import json
import wave
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageSequence
from fastapi.testclient import TestClient

from web_stable_diffusion.runtime.api import create_app


def _build_client() -> TestClient:
    client = TestClient(create_app())
    client.headers.update({"X-API-Key": "test-key"})
    return client


def _collect_stream(client: TestClient, payload: Dict[str, object]) -> Tuple[List[Dict[str, object]], str]:
    token_id = ""
    events: List[Dict[str, object]] = []
    with client.stream("POST", "/generate", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "info" and event.get("cancel_token"):
                token_id = event["cancel_token"]
            events.append(event)
    assert token_id, "Streaming API did not return a cancellation token"
    return events, token_id


def _decode_png(encoded: str) -> Image.Image:
    payload = base64.b64decode(encoded.encode("ascii"))
    return Image.open(io.BytesIO(payload))


def _decode_wav(encoded: str) -> wave.Wave_read:
    payload = base64.b64decode(encoded.encode("ascii"))
    buffer = io.BytesIO(payload)
    return wave.open(buffer)


def test_streaming_api_emits_all_modalities(api_security_env: None) -> None:
    with _build_client() as client:
        events, token_id = _collect_stream(
            client,
            {"prompt": "orchestral sunrise", "frames": 4, "audio_length": 128},
        )

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
    scheduler = metrics["scheduler"]
    assert scheduler["modalities_completed"] == len(modality_events)
    assert scheduler["executor"] in {"thread", "process"}
    assert scheduler["timeline_events"] >= len(modality_events)

    manifest_response = client.get(f"/manifests/{token_id}")
    assert manifest_response.status_code == 200
    manifest_entry = manifest_response.json()
    assert manifest_entry["status"] == "complete"
    assert manifest_entry["token"] == token_id
    assert manifest_entry["manifest"]["artifacts"] == manifest["artifacts"]
    assert manifest_entry["request"]["prompt"] == "orchestral sunrise"

    listing = client.get("/manifests", params={"limit": 1})
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert any(item["token"] == token_id for item in items)

    delete_response = client.delete(f"/manifests/{token_id}")
    assert delete_response.status_code == 200
    assert client.get(f"/manifests/{token_id}").status_code == 404


def test_streaming_api_rejects_invalid_executor(api_security_env: None) -> None:
    with _build_client() as client:
        response = client.post(
            "/generate",
            json={"prompt": "hello", "executor": "invalid"},
        )
        assert response.status_code == 422


def test_manifest_registry_records_errors(api_security_env: None) -> None:
    with _build_client() as client:
        events, token_id = _collect_stream(
            client,
            {
                "prompt": "budget bust",
                "frames": 2,
                "audio_length": 32,
                "budgets": {"default": {"memory_bytes": 1}},
            },
        )

    last = events[-1]
    assert last["type"] == "error"
    record_response = client.get(f"/manifests/{token_id}")
    assert record_response.status_code == 200
    record = record_response.json()
    assert record["status"] == "error"
    assert record["error"]["code"] == "budget_exceeded"
    assert record["request"]["prompt"] == "budget bust"
