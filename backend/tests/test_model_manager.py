"""Tests for ModelManager."""

from unittest.mock import MagicMock

from neural_mri.core.model_manager import ModelManager, _parse_param_str


def test_initial_state_not_loaded():
    mm = ModelManager()
    assert mm.is_loaded is False
    assert mm.model_id is None


def test_get_model_raises_when_no_model():
    mm = ModelManager()
    try:
        mm.get_model()
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "No model" in str(e)


def test_get_model_info_raises_when_no_model():
    mm = ModelManager()
    try:
        mm.get_model_info()
        assert False, "Should have raised"
    except RuntimeError:
        pass


def test_resolve_device_cpu():
    assert ModelManager._resolve_device("cpu") == "cpu"


def test_parse_param_str():
    assert _parse_param_str("124M") == 124_000_000
    assert _parse_param_str("1.4B") == 1_400_000_000
    assert _parse_param_str("355M") == 355_000_000
    assert _parse_param_str("100") == 100


def test_unload_clears_state():
    mm = ModelManager()
    mm._model = MagicMock()
    mm._model_id = "gpt2"
    mm.unload_model()
    assert mm.is_loaded is False
    assert mm.model_id is None
