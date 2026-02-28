"""Tests for the SAE registry."""

from neural_mri.core.sae_registry import get_sae_info, list_sae_support


def test_get_sae_info_gpt2():
    info = get_sae_info("gpt2")
    assert info is not None
    assert info["release"] == "gpt2-small-res-jb"
    assert 0 in info["layers"]
    assert info["d_sae"] == 24576


def test_get_sae_info_gemma():
    info = get_sae_info("google/gemma-2-2b")
    assert info is not None
    assert info["release"] == "gemma-scope-2b-pt-res-canonical"
    assert len(info["layers"]) == 26


def test_get_sae_info_unknown_returns_none():
    assert get_sae_info("gpt2-medium") is None
    assert get_sae_info("nonexistent") is None


def test_list_sae_support_has_all_registry_models():
    support = list_sae_support()
    assert "gpt2" in support
    assert "gpt2-medium" in support
    assert len(support) == 5


def test_list_sae_support_gpt2_true():
    assert list_sae_support()["gpt2"] is True


def test_list_sae_support_gpt2_medium_false():
    assert list_sae_support()["gpt2-medium"] is False
