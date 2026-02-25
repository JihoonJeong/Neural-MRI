from __future__ import annotations

from pydantic import BaseModel


class TestCase(BaseModel):
    test_id: str
    category: str
    name: str
    prompt: str
    expected_behavior: str
    expected_tokens: list[str] | None = None
    unexpected_tokens: list[str] | None = None
    compare_prompts: list[str] | None = None


class TestPrediction(BaseModel):
    token: str
    prob: float


class ActivationSummary(BaseModel):
    peak_layers: list[int]
    peak_activation: float
    active_layer_count: int


class CompareResult(BaseModel):
    prompt: str
    top_k: list[TestPrediction]
    pronoun_probs: dict[str, float] | None = None  # e.g. {" he": 0.3, " she": 0.1}


class TestResult(BaseModel):
    test_id: str
    category: str
    name: str
    prompt: str
    passed: bool
    top_k: list[TestPrediction]
    actual_token: str
    actual_prob: float
    expected_prob: float | None = None
    activation_summary: ActivationSummary
    interpretation: str
    compare_results: list[CompareResult] | None = None


class BatteryResult(BaseModel):
    model_id: str
    total_tests: int
    passed: int
    failed: int
    results: list[TestResult]
    summary: str


class BatteryRunRequest(BaseModel):
    categories: list[str] | None = None  # None = all categories
    locale: str = "en"
