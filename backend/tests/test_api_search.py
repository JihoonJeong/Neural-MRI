"""Tests for GET /api/model/search endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from neural_mri.core.model_search import SearchResult
from neural_mri.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _make_result(model_id: str = "user/gpt2", tl_compat: bool = True) -> SearchResult:
    return SearchResult(
        model_id=model_id,
        author="user",
        downloads=1000,
        likes=10,
        pipeline_tag="text-generation",
        gated=False,
        tl_compat=tl_compat,
        architectures=["GPT2LMHeadModel"],
    )


@patch("neural_mri.api.routes_search.search_models")
def test_search_returns_results(mock_search, client):
    mock_search.return_value = [_make_result("user/gpt2"), _make_result("user/gpt2-xl")]
    resp = client.get("/api/model/search?q=gpt2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["model_id"] == "user/gpt2"
    assert "downloads" in data[0]
    assert "tl_compat" in data[0]
    assert "architectures" in data[0]


def test_search_empty_query_422(client):
    resp = client.get("/api/model/search?q=")
    assert resp.status_code == 422


@patch("neural_mri.api.routes_search.search_models")
def test_search_tl_only_param(mock_search, client):
    mock_search.return_value = []
    client.get("/api/model/search?q=gpt&tl_only=true")
    mock_search.assert_called_once()
    kwargs = mock_search.call_args.kwargs
    assert kwargs.get("filter_tl_compat") is True


@patch("neural_mri.api.routes_search.search_models")
def test_search_limit_param(mock_search, client):
    mock_search.return_value = []
    client.get("/api/model/search?q=gpt&limit=5")
    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args
    assert call_kwargs.kwargs.get("limit") == 5 or call_kwargs[1].get("limit") == 5


@patch("neural_mri.api.routes_search.search_models")
def test_search_hub_failure_502(mock_search, client):
    mock_search.side_effect = Exception("Hub down")
    resp = client.get("/api/model/search?q=gpt")
    assert resp.status_code == 502
    assert "Hub" in resp.json()["detail"]
