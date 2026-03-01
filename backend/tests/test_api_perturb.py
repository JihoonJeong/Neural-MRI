"""Integration tests for /api/perturb endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from neural_mri.core.model_manager import ModelManager
from neural_mri.main import app


@pytest.fixture
def client(mock_model_manager):
    from neural_mri.api.routes_perturb import get_model_manager

    app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_model():
    """Client with no model loaded."""
    from neural_mri.api.routes_perturb import get_model_manager

    mm = MagicMock(spec=ModelManager)
    mm.is_loaded = False
    app.dependency_overrides[get_model_manager] = lambda: mm
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── Zero Out ──────────────────────────────────────────────────────


def test_perturb_zero_200(client):
    resp = client.post(
        "/api/perturb/zero",
        json={
            "component": "blocks.0.attn",
            "prompt": "The capital of France is",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "gpt2"
    assert data["component"] == "blocks.0.attn"
    assert data["perturbation_type"] == "zero_out"
    assert "original" in data
    assert "perturbed" in data
    assert "logit_diff" in data
    assert "kl_divergence" in data
    assert "top_k_original" in data
    assert len(data["top_k_original"]) == 5


# ── Amplify ───────────────────────────────────────────────────────


def test_perturb_amplify_200(client):
    resp = client.post(
        "/api/perturb/amplify",
        json={
            "component": "blocks.0.mlp",
            "factor": 3.0,
            "prompt": "Hello world",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["perturbation_type"] == "amplify"
    assert data["component"] == "blocks.0.mlp"


# ── Ablate ────────────────────────────────────────────────────────


def test_perturb_ablate_200(client):
    resp = client.post(
        "/api/perturb/ablate",
        json={
            "component": "blocks.1.attn",
            "prompt": "test input",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["perturbation_type"] == "ablate"


# ── Patch ─────────────────────────────────────────────────────────


def test_perturb_patch_200(client):
    resp = client.post(
        "/api/perturb/patch",
        json={
            "clean_prompt": "The capital of France is",
            "corrupt_prompt": "The capital of Germany is",
            "component": "blocks.0.attn",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["component"] == "blocks.0.attn"
    assert "recovery_score" in data
    assert 0 <= data["recovery_score"] <= 1
    assert "clean_prediction" in data
    assert "corrupt_prediction" in data
    assert "patched_prediction" in data


# ── Causal Trace ──────────────────────────────────────────────────


def test_perturb_causal_trace_200(client):
    resp = client.post(
        "/api/perturb/causal-trace",
        json={
            "clean_prompt": "The capital of France is",
            "corrupt_prompt": "The capital of Germany is",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_layers"] == 2
    assert len(data["cells"]) == 5  # embed + 2*attn + 2*mlp
    assert "clean_prediction" in data
    assert "corrupt_prediction" in data


def test_perturb_causal_trace_cell_types(client):
    resp = client.post(
        "/api/perturb/causal-trace",
        json={
            "clean_prompt": "A",
            "corrupt_prompt": "B",
        },
    )
    data = resp.json()
    types = [c["component_type"] for c in data["cells"]]
    assert types.count("embed") == 1
    assert types.count("attn") == 2
    assert types.count("mlp") == 2


# ── Reset ─────────────────────────────────────────────────────────


def test_perturb_reset_200(client):
    resp = client.post("/api/perturb/reset")
    assert resp.status_code == 200
    assert resp.json() == {"status": "reset"}


# ── Error Cases ───────────────────────────────────────────────────


def test_perturb_no_model_400(client_no_model):
    resp = client_no_model.post(
        "/api/perturb/zero",
        json={
            "component": "blocks.0.attn",
            "prompt": "test",
        },
    )
    assert resp.status_code == 400
    assert "No model loaded" in resp.json()["detail"]


# ── Field validations ─────────────────────────────────────────────


def test_perturb_kl_nonnegative(client):
    """KL divergence should be non-negative (with some float tolerance)."""
    resp = client.post(
        "/api/perturb/zero",
        json={
            "component": "blocks.0.attn",
            "prompt": "test",
        },
    )
    data = resp.json()
    # KL divergence >= 0 (allow small float imprecision)
    assert data["kl_divergence"] >= -0.01


def test_perturb_patch_recovery_bounded(client):
    resp = client.post(
        "/api/perturb/patch",
        json={
            "clean_prompt": "A B C",
            "corrupt_prompt": "D E F",
            "component": "blocks.0.mlp",
        },
    )
    data = resp.json()
    assert 0 <= data["recovery_score"] <= 1
