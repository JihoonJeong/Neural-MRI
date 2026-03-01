"""Tests for core/model_search.py — TL compatibility, search cache, search function."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from neural_mri.core.model_search import (
    SearchResult,
    _search_cache,
    _SearchCache,
    check_tl_compatibility,
    search_models,
)

# ── TL Compatibility ─────────────────────────────────────────────


def test_tl_compat_known_architecture():
    assert check_tl_compatibility(["GPT2LMHeadModel"], "any-model") is True


def test_tl_compat_unknown_architecture():
    assert check_tl_compatibility(["CustomTransformerModel"], "any-model") is False


def test_tl_compat_mixed_architectures():
    # One known + one unknown → True (any match)
    assert check_tl_compatibility(["CustomModel", "LlamaForCausalLM"], "x") is True


def test_tl_compat_no_arch_known_prefix():
    assert check_tl_compatibility([], "gpt2-small") is True


def test_tl_compat_no_arch_known_prefix_case_insensitive():
    assert check_tl_compatibility([], "EleutherAI/Pythia-70m") is True


def test_tl_compat_no_arch_unknown_prefix():
    assert check_tl_compatibility([], "totally-custom-model") is None


# ── Search Cache ──────────────────────────────────────────────────


def test_search_cache_hit():
    cache = _SearchCache()
    results = [SearchResult("m1", None, 100, 10, None, False, True, [])]
    cache.put("key", results)
    assert cache.get("key") is results


def test_search_cache_miss():
    cache = _SearchCache()
    assert cache.get("nonexistent") is None


def test_search_cache_ttl_expiry():
    cache = _SearchCache(_ttl=0.01)
    cache.put("key", [])
    time.sleep(0.02)
    assert cache.get("key") is None


def test_search_cache_eviction():
    cache = _SearchCache()
    for i in range(51):
        cache.put(f"key_{i}", [])
    assert len(cache._cache) == 50


# ── search_models ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_search_cache():
    """Clear module-level search cache between tests."""
    _search_cache._cache.clear()
    yield
    _search_cache._cache.clear()


def _make_mock_model(model_id: str, downloads: int = 1000, architectures: list[str] | None = None):
    """Create a mock HF model object."""
    m = MagicMock()
    m.id = model_id
    m.author = model_id.split("/")[0] if "/" in model_id else None
    m.downloads = downloads
    m.likes = 10
    m.pipeline_tag = "text-generation"
    m.gated = False
    if architectures is not None:
        m.config = {"architectures": architectures}
    else:
        m.config = None
    return m


@patch("huggingface_hub.HfApi")
def test_search_models_returns_results(mock_hf_cls):
    mock_api = MagicMock()
    mock_hf_cls.return_value = mock_api
    mock_api.list_models.return_value = [
        _make_mock_model("user/gpt2-custom", 5000, ["GPT2LMHeadModel"]),
        _make_mock_model("user/llama-tiny", 3000, ["LlamaForCausalLM"]),
    ]

    results = search_models("gpt", limit=10)
    assert len(results) == 2
    assert results[0].model_id == "user/gpt2-custom"
    assert results[0].tl_compat is True
    assert results[0].downloads == 5000


@patch("huggingface_hub.HfApi")
def test_search_models_tl_filter(mock_hf_cls):
    mock_api = MagicMock()
    mock_hf_cls.return_value = mock_api
    mock_api.list_models.return_value = [
        _make_mock_model("user/gpt2-custom", 5000, ["GPT2LMHeadModel"]),
        _make_mock_model("user/custom-model", 3000, ["CustomArch"]),
    ]

    results = search_models("model", limit=10, filter_tl_compat=True)
    assert len(results) == 1
    assert results[0].model_id == "user/gpt2-custom"


@patch("huggingface_hub.HfApi")
def test_search_models_caches_result(mock_hf_cls):
    mock_api = MagicMock()
    mock_hf_cls.return_value = mock_api
    mock_api.list_models.return_value = [
        _make_mock_model("user/m1", 100, ["GPT2LMHeadModel"]),
    ]

    # First call
    results1 = search_models("cached_query", limit=5)
    # Second call — should hit cache
    results2 = search_models("cached_query", limit=5)

    assert results1 == results2
    # HfApi should only be instantiated once (first call)
    assert mock_hf_cls.call_count == 1


@patch("huggingface_hub.HfApi")
def test_search_models_hub_failure_raises(mock_hf_cls):
    mock_api = MagicMock()
    mock_hf_cls.return_value = mock_api
    mock_api.list_models.side_effect = Exception("Network error")

    with pytest.raises(Exception, match="Network error"):
        search_models("fail_query", limit=5)
