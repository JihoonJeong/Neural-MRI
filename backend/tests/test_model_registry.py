"""Tests for the model registry."""

import pytest

from neural_mri.core.model_registry import (
    MODEL_REGISTRY,
    _recent_models,
    add_recent_model,
    get_model_info,
    list_models,
)


@pytest.fixture(autouse=True)
def _clean_recent():
    """Reset recent models between tests."""
    _recent_models.clear()
    yield
    _recent_models.clear()


def test_list_models_returns_8():
    assert len(list_models()) == 8


def test_list_models_marks_loaded():
    models = list_models("gpt2")
    loaded = [m for m in models if m["is_loaded"]]
    assert len(loaded) == 1
    assert loaded[0]["model_id"] == "gpt2"


def test_list_models_none_loaded():
    models = list_models(None)
    assert all(not m["is_loaded"] for m in models)


def test_get_model_info_gpt2():
    info = get_model_info("gpt2")
    assert info is not None
    assert info["params"] == "124M"
    assert info["family"] == "gpt2"


def test_get_model_info_unknown():
    assert get_model_info("nonexistent") is None


# ── add_recent_model tests ───────────────────────────────────────


def test_add_recent_model_appears_in_list():
    add_recent_model("custom/my-model", n_params=500_000_000)
    models = list_models()
    model_ids = [m["model_id"] for m in models]
    assert "custom/my-model" in model_ids
    assert len(models) == len(MODEL_REGISTRY) + 1


def test_add_recent_model_params_formatting_millions():
    add_recent_model("small/model", n_params=500_000_000)
    models = list_models()
    entry = next(m for m in models if m["model_id"] == "small/model")
    assert entry["params"] == "500M"


def test_add_recent_model_params_formatting_billions():
    add_recent_model("big/model", n_params=1_500_000_000)
    models = list_models()
    entry = next(m for m in models if m["model_id"] == "big/model")
    assert entry["params"] == "1.5B"


def test_add_recent_model_no_duplicate():
    add_recent_model("custom/dup", n_params=100_000_000)
    add_recent_model("custom/dup", n_params=100_000_000)
    models = list_models()
    dup_count = sum(1 for m in models if m["model_id"] == "custom/dup")
    assert dup_count == 1


def test_add_recent_model_skips_registry():
    initial_count = len(list_models())
    add_recent_model("gpt2", n_params=124_000_000)
    assert len(list_models()) == initial_count


def test_list_models_source_field():
    add_recent_model("custom/dynamic", n_params=100_000_000)
    models = list_models()
    for m in models:
        if m["model_id"] in MODEL_REGISTRY:
            assert m["source"] == "registry"
        elif m["model_id"] == "custom/dynamic":
            assert m["source"] == "dynamic"


def test_add_recent_model_display_name():
    add_recent_model("org/my-cool-model", n_params=100_000_000)
    models = list_models()
    entry = next(m for m in models if m["model_id"] == "org/my-cool-model")
    assert entry["display_name"] == "my-cool-model"
