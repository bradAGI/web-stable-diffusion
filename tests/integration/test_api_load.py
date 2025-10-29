from __future__ import annotations

import concurrent.futures
import json
from typing import Dict, List

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for API tests")

from fastapi.testclient import TestClient

from web_stable_diffusion.runtime.api import create_app


def _collect_events(payload: Dict[str, object]) -> Dict[str, object]:
    with TestClient(create_app()) as client:
        client.headers.update({"X-API-Key": "test-key"})
        with client.stream("POST", "/generate", json=payload) as response:
            assert response.status_code == 200
            events = [json.loads(line) for line in response.iter_lines() if line]
    return {"events": events, "final": events[-1] if events else {}}


def test_concurrent_generation_requests_stay_isolated(api_security_env: None) -> None:
    payload = {
        "prompt": "interstellar gardens",
        "frames": 3,
        "audio_length": 96,
        "executor": "thread",
        "max_workers": 2,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_collect_events, payload) for _ in range(3)]
        results = [future.result() for future in futures]

    for result in results:
        events = result["events"]
        modality_events: List[Dict[str, object]] = [
            event for event in events if event.get("type") == "modality"
        ]
        assert len(modality_events) == 4
        progress_values = [event["progress"] for event in modality_events]
        assert progress_values == sorted(progress_values)
        manifest = result["final"].get("manifest", {})
        assert set(manifest.get("artifacts", {}).keys()) == {"audio", "image", "video", "volume"}
        metrics = manifest.get("metadata", {}).get("metrics", {})
        assert metrics.get("wall_clock_s_total", 0.0) >= 0.0
        assert metrics.get("wall_clock_s_elapsed", 0.0) >= 0.0
        scheduler = metrics["scheduler"]
        assert scheduler["modalities_completed"] == 4
        assert scheduler["worker_count"] >= 1
