"""API tests for /api/battery endpoints."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from neural_mri.api.routes_battery import get_model_manager, get_sae_manager
from neural_mri.main import app
from neural_mri.schemas.battery import (
    ActivationSummary,
    BatteryResult,
    TestPrediction,
    TestResult,
)


def _make_battery_result(**overrides):
    defaults = dict(
        model_id="gpt2",
        total_tests=7,
        passed=4,
        failed=3,
        results=[
            TestResult(
                test_id="t1",
                category="factual_recall",
                name="Test",
                prompt="p",
                passed=True,
                top_k=[TestPrediction(token="x", prob=0.5)],
                actual_token="x",
                actual_prob=0.5,
                activation_summary=ActivationSummary(
                    peak_layers=[0],
                    peak_activation=1.0,
                    active_layer_count=1,
                ),
                interpretation="ok",
            ),
        ],
        summary="4/7 tests passed.",
    )
    defaults.update(overrides)
    return BatteryResult(**defaults)


@pytest.fixture
def _override_deps(mock_model_manager, mock_sae_manager):
    app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
    app.dependency_overrides[get_sae_manager] = lambda: mock_sae_manager
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def _override_no_model():
    mm = MagicMock()
    mm.is_loaded = False
    app.dependency_overrides[get_model_manager] = lambda: mm
    yield
    app.dependency_overrides.clear()


async def test_get_tests_returns_list(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/battery/tests")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 7


async def test_run_battery_no_model_returns_400(_override_no_model):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/battery/run", json={})
    assert resp.status_code == 400


async def test_run_battery_default(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/battery/run", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "total_tests" in data


async def test_run_battery_with_categories(_override_deps):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/battery/run", json={"categories": ["factual_recall"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tests"] == 2
