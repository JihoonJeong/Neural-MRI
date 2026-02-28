"""API tests for /api/scan endpoints."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from neural_mri.main import app


def _get_deps():
    from neural_mri.api.routes_scan import get_model_manager, get_scan_cache
    return get_model_manager, get_scan_cache


@pytest.fixture
def _override_deps(mock_model_manager):
    from neural_mri.core.scan_cache import ScanCache
    get_mm, get_cache = _get_deps()
    app.dependency_overrides[get_mm] = lambda: mock_model_manager
    app.dependency_overrides[get_cache] = lambda: ScanCache(max_entries=5)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def _override_no_model():
    mm = MagicMock()
    mm.is_loaded = False
    get_mm, _ = _get_deps()
    app.dependency_overrides[get_mm] = lambda: mm
    yield
    app.dependency_overrides.clear()


async def test_scan_structural_200(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/scan/structural", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "layers" in data
    assert "connections" in data


async def test_scan_structural_no_model_400(_override_no_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/scan/structural", json={})
    assert resp.status_code == 400


async def test_scan_weights_200(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/scan/weights", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "layers" in data


async def test_scan_activation_200(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/scan/activation", json={"prompt": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "tokens" in data
    assert "layers" in data
