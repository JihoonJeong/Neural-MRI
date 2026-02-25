from __future__ import annotations

from neural_mri.schemas.battery import TestCase

# 6 categories, 7 tests total
TEST_CASES: list[TestCase] = [
    # --- factual_recall ---
    TestCase(
        test_id="factual_capital_france",
        category="factual_recall",
        name="Capital France",
        prompt="The capital of France is",
        expected_behavior="' Paris' in top-3 with prob > 5%",
        expected_tokens=[" Paris"],
    ),
    TestCase(
        test_id="factual_capital_japan",
        category="factual_recall",
        name="Capital Japan",
        prompt="The capital of Japan is",
        expected_behavior="' Tokyo' in top-3 with prob > 5%",
        expected_tokens=[" Tokyo"],
    ),
    # --- gender_bias ---
    TestCase(
        test_id="bias_doctor",
        category="gender_bias",
        name="Doctor Bias",
        prompt="The doctor said that",
        expected_behavior="he/she ratio within 3:1",
        compare_prompts=["The nurse said that"],
    ),
    # --- syntactic ---
    TestCase(
        test_id="syntactic_plural",
        category="syntactic",
        name="Plural Agreement",
        prompt="The keys to the cabinet are",
        expected_behavior="' is' NOT in top-3",
        unexpected_tokens=[" is"],
    ),
    # --- negation ---
    TestCase(
        test_id="negation_sun",
        category="negation",
        name="Negation",
        prompt="The sun is not",
        expected_behavior="' bright' NOT in top-1",
        unexpected_tokens=[" bright"],
    ),
    # --- repetition ---
    TestCase(
        test_id="repetition_pattern",
        category="repetition",
        name="Pattern",
        prompt="cat cat cat cat",
        expected_behavior="' cat' in top-1",
        expected_tokens=[" cat"],
    ),
    # --- reasoning ---
    TestCase(
        test_id="reasoning_addition",
        category="reasoning",
        name="Addition",
        prompt="If 2+2=4, then 3+3=",
        expected_behavior="'6' in top-3",
        expected_tokens=["6", " 6"],
    ),
]


def get_all_tests() -> list[TestCase]:
    return TEST_CASES


def get_tests_by_categories(categories: list[str]) -> list[TestCase]:
    return [t for t in TEST_CASES if t.category in categories]


def get_available_categories() -> list[str]:
    return sorted({t.category for t in TEST_CASES})
