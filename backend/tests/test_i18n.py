"""Tests for the i18n translation system."""

from neural_mri.i18n import T


def test_t_en_simple_key():
    assert T("en", "report.arch_title") == "Architecture"


def test_t_ko_simple_key():
    result = T("ko", "report.arch_title")
    assert result != "report.arch_title"  # should not be the key itself
    assert result != "Architecture"  # should be Korean


def test_t_en_with_kwargs():
    result = T("en", "report.arch_layers", n=12)
    assert "12" in result
    assert "layer" in result.lower()


def test_t_unknown_locale_falls_back_to_en():
    assert T("fr", "report.arch_title") == "Architecture"


def test_t_unknown_key_returns_key():
    assert T("en", "nonexistent.key") == "nonexistent.key"
