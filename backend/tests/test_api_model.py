"""API tests for /api/model endpoints."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from neural_mri.main import app


def _get_model_manager_dep():
    from neural_mri.api.routes_model import get_model_manager

    return get_model_manager


@pytest.fixture
def _override_deps(mock_model_manager):
    dep = _get_model_manager_dep()
    app.dependency_overrides[dep] = lambda: mock_model_manager
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def _override_no_model():
    mm = MagicMock()
    mm.is_loaded = False
    mm.model_id = None
    dep = _get_model_manager_dep()
    app.dependency_overrides[dep] = lambda: mm
    yield
    app.dependency_overrides.clear()


async def test_model_list_returns_array(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/model/list")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 8


async def test_model_info_loaded(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "gpt2"


async def test_model_info_not_loaded(_override_no_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/model/info")
    assert resp.status_code == 404 or resp.status_code == 400


async def test_root_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
