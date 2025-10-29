"""Pytest fixtures shared across omni-modal tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def api_security_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Configure environment variables required to hit the secured API."""

    db_path = tmp_path / "manifests.db"
    monkeypatch.setenv("OMNIMODAL_API_KEYS", "test-key")
    monkeypatch.setenv("OMNIMODAL_RATE_LIMIT", "100")
    monkeypatch.setenv("OMNIMODAL_RATE_PERIOD", "60")
    monkeypatch.setenv("OMNIMODAL_MANIFEST_DB", str(db_path))
