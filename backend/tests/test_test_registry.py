"""Tests for the functional test registry."""

from neural_mri.core.test_registry import (
    get_all_tests,
    get_available_categories,
    get_tests_by_categories,
)


def test_get_all_tests_returns_7():
    assert len(get_all_tests()) == 7


def test_all_tests_have_required_fields():
    for tc in get_all_tests():
        assert tc.test_id
        assert tc.category
        assert tc.name
        assert tc.prompt


def test_get_tests_by_categories_factual():
    results = get_tests_by_categories(["factual_recall"])
    assert len(results) == 2
    assert all(r.category == "factual_recall" for r in results)


def test_get_tests_by_categories_empty():
    assert get_tests_by_categories(["nonexistent"]) == []


def test_get_tests_by_multiple_categories():
    results = get_tests_by_categories(["factual_recall", "negation"])
    assert len(results) == 3


def test_get_available_categories_returns_6():
    cats = get_available_categories()
    assert len(cats) == 6
    assert cats == sorted(cats)


def test_test_ids_are_unique():
    ids = [tc.test_id for tc in get_all_tests()]
    assert len(ids) == len(set(ids))
