"""Tests for SAEManager — mock SAE.from_pretrained."""

from unittest.mock import MagicMock, patch

import pytest

from neural_mri.core.sae_manager import SAEManager


@pytest.fixture
def _mock_from_pretrained():
    mock_sae = MagicMock()
    mock_sae.cfg = MagicMock()
    mock_sae.cfg.d_sae = 24576
    with patch("neural_mri.core.sae_manager.SAE") as MockSAE:
        MockSAE.from_pretrained.return_value = mock_sae
        yield MockSAE, mock_sae


def test_initial_state():
    mgr = SAEManager()
    assert mgr._sae is None
    assert mgr._model_id is None


def test_get_sae_loads(_mock_from_pretrained):
    MockSAE, mock_sae = _mock_from_pretrained
    mgr = SAEManager()
    result = mgr.get_sae("gpt2", 5, "cpu")
    assert result is mock_sae
    MockSAE.from_pretrained.assert_called_once()


def test_get_sae_caches(_mock_from_pretrained):
    MockSAE, mock_sae = _mock_from_pretrained
    mgr = SAEManager()
    mgr.get_sae("gpt2", 5, "cpu")
    mgr.get_sae("gpt2", 5, "cpu")
    assert MockSAE.from_pretrained.call_count == 1


def test_get_sae_unsupported_model():
    mgr = SAEManager()
    with pytest.raises(ValueError, match="No SAE available"):
        mgr.get_sae("gpt2-medium", 5, "cpu")


def test_get_sae_invalid_layer():
    mgr = SAEManager()
    with pytest.raises(ValueError, match="Layer 99"):
        mgr.get_sae("gpt2", 99, "cpu")
