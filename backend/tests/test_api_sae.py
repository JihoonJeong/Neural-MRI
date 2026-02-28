"""API tests for /api/sae endpoints."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from neural_mri.api.routes_sae import get_model_manager, get_sae_manager, get_scan_cache
from neural_mri.core.scan_cache import ScanCache
from neural_mri.main import app


@pytest.fixture
def _override_deps(mock_model_manager, mock_sae_manager):
    app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
    app.dependency_overrides[get_sae_manager] = lambda: mock_sae_manager
    app.dependency_overrides[get_scan_cache] = lambda: ScanCache(max_entries=5)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def _override_no_model():
    mm = MagicMock()
    mm.is_loaded = False
    mm.model_id = None
    app.dependency_overrides[get_model_manager] = lambda: mm
    yield
    app.dependency_overrides.clear()


async def test_sae_info_model_loaded(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/sae/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["model_id"] == "gpt2"


async def test_sae_info_no_model(_override_no_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/sae/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is False


async def test_sae_support_returns_dict(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/sae/support")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpt2" in data
    assert isinstance(data["gpt2"], bool)


async def test_sae_scan_no_model_400(_override_no_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/sae/scan", json={"prompt": "test", "layer_idx": 0})
    assert resp.status_code == 400
