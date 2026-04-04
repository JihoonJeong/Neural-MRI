"""Tests for SAEManager — provider-based SAE loading."""

from unittest.mock import MagicMock, patch

import pytest

from neural_mri.core.sae_manager import SAEManager


@pytest.fixture
def _mock_provider():
    mock_prov = MagicMock()
    mock_prov.d_sae = 24576
    mock_prov.hook_name = "blocks.5.hook_resid_pre"
    mock_prov.provider_name = "MockProvider"
    with patch("neural_mri.core.sae_manager.load_sae_provider") as mock_load:
        mock_load.return_value = mock_prov
        yield mock_load, mock_prov


def test_initial_state():
    mgr = SAEManager()
    assert mgr._provider is None
    assert mgr._model_id is None


def test_get_provider_loads(_mock_provider):
    mock_load, mock_prov = _mock_provider
    mgr = SAEManager()
    result = mgr.get_provider("gpt2", 5, "cpu")
    assert result is mock_prov
    mock_load.assert_called_once()


def test_get_provider_caches(_mock_provider):
    mock_load, mock_prov = _mock_provider
    mgr = SAEManager()
    mgr.get_provider("gpt2", 5, "cpu")
    mgr.get_provider("gpt2", 5, "cpu")
    assert mock_load.call_count == 1


def test_get_provider_unsupported_model():
    mgr = SAEManager()
    with pytest.raises(ValueError, match="No SAE available"):
        mgr.get_provider("gpt2-medium", 5, "cpu")


def test_get_provider_invalid_layer():
    mgr = SAEManager()
    with pytest.raises(ValueError, match="Layer 99"):
        mgr.get_provider("gpt2", 99, "cpu")


def test_get_sae_alias(_mock_provider):
    """get_sae is a backward-compat alias for get_provider."""
    mock_load, mock_prov = _mock_provider
    mgr = SAEManager()
    result = mgr.get_sae("gpt2", 5, "cpu")
    assert result is mock_prov
