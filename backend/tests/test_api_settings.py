"""Tests for /api/settings endpoints (token + cache management)."""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from neural_mri.core.scan_cache import ScanCache
from neural_mri.main import app


@pytest.fixture(autouse=True)
def _override_settings():
    """Override settings dependencies and reset global token state."""
    from neural_mri.api.routes_settings import get_scan_cache, get_settings

    cache = ScanCache(max_entries=5)
    settings = types.SimpleNamespace(max_cache_entries=5)
    app.dependency_overrides[get_scan_cache] = lambda: cache
    app.dependency_overrides[get_settings] = lambda: settings
    yield cache
    app.dependency_overrides.clear()
    # Reset global token state
    import neural_mri.api.routes_settings as rs

    rs._runtime_token = None


@pytest.fixture
def client():
    return TestClient(app)


# ── Token endpoints ──────────────────────────────────────────────


@patch("neural_mri.api.routes_settings._validate_token", return_value=True)
def test_post_token_valid(mock_validate, client):
    resp = client.post("/api/settings/token", json={"token": "hf_test123"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_set"] is True
    assert data["is_valid"] is True
    assert data["source"] == "runtime"
    mock_validate.assert_called_once_with("hf_test123")


@patch("neural_mri.api.routes_settings._validate_token", return_value=False)
def test_post_token_invalid(mock_validate, client):
    resp = client.post("/api/settings/token", json={"token": "bad_token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_set"] is True
    assert data["is_valid"] is False
    assert data["source"] == "runtime"


@patch("neural_mri.api.routes_settings._validate_token", return_value=True)
def test_delete_token(mock_validate, client):
    # First set a token
    client.post("/api/settings/token", json={"token": "hf_test"})
    # Then delete it
    resp = client.delete("/api/settings/token")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_set"] is False
    assert data["is_valid"] is None
    assert data["source"] == "none"


def test_get_token_status_no_token(client):
    resp = client.get("/api/settings/token/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_set"] is False
    assert data["source"] == "none"


@patch("neural_mri.api.routes_settings._validate_token", return_value=True)
def test_get_token_status_after_set(mock_validate, client):
    client.post("/api/settings/token", json={"token": "hf_valid"})
    resp = client.get("/api/settings/token/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_set"] is True
    assert data["source"] == "runtime"


# ── Cache endpoints ──────────────────────────────────────────────


def test_get_cache_status_empty(client):
    resp = client.get("/api/settings/cache")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entry_count"] == 0
    assert data["max_entries"] == 5


def test_get_cache_status_with_entries(client, _override_settings):
    cache = _override_settings
    cache.put("model1", "T1", "", {"data": 1})
    cache.put("model1", "T2", "", {"data": 2})
    resp = client.get("/api/settings/cache")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entry_count"] == 2


def test_delete_cache_clears(client, _override_settings):
    cache = _override_settings
    cache.put("model1", "T1", "", {"data": 1})
    assert len(cache._store) == 1

    resp = client.delete("/api/settings/cache")
    assert resp.status_code == 200
    assert resp.json() == {"status": "cleared"}

    # Verify cache is empty
    resp2 = client.get("/api/settings/cache")
    assert resp2.json()["entry_count"] == 0
