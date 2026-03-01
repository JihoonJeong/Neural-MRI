"""Tests for the model registry."""

from neural_mri.core.model_registry import get_model_info, list_models


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
