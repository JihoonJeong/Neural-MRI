"""Tests for BatteryEngine — mock model + torch tensors."""

from neural_mri.core.battery_engine import BatteryEngine
from neural_mri.schemas.battery import SAEFeatureBrief


def test_run_battery_all_categories(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)
    result = engine.run_battery()
    assert result.total_tests == 7
    assert result.passed + result.failed == 7
    assert result.model_id == "gpt2"


def test_run_battery_single_category(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)
    result = engine.run_battery(categories=["factual_recall"])
    assert result.total_tests == 2


def test_run_battery_empty_category(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)
    result = engine.run_battery(categories=["nonexistent"])
    assert result.total_tests == 0


def test_battery_result_structure(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)
    result = engine.run_battery()
    assert result.summary
    for r in result.results:
        assert r.test_id
        assert r.category
        assert r.prompt
        assert r.top_k
        assert r.activation_summary is not None
        assert r.interpretation


def test_battery_summary_locale_en(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)
    result = engine.run_battery(locale="en")
    assert "test" in result.summary.lower()


def test_battery_summary_locale_ko(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)
    result = engine.run_battery(locale="ko")
    # Should contain the count
    assert str(result.total_tests) in result.summary


def test_battery_no_sae_by_default(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)
    result = engine.run_battery()
    assert result.sae_summary is None
    assert all(r.sae_features is None for r in result.results)


def test_build_cross_test_summary(mock_model_manager):
    engine = BatteryEngine(mock_model_manager)

    # Simulate features: feature #10 appears in two tests
    features_a = [
        SAEFeatureBrief(feature_idx=10, activation=5.0, activation_normalized=1.0),
        SAEFeatureBrief(feature_idx=20, activation=3.0, activation_normalized=0.6),
    ]
    features_b = [
        SAEFeatureBrief(feature_idx=10, activation=4.0, activation_normalized=0.8),
        SAEFeatureBrief(feature_idx=30, activation=2.0, activation_normalized=0.4),
    ]

    from neural_mri.schemas.battery import ActivationSummary, TestResult

    results = [
        TestResult(
            test_id="test_a",
            category="factual_recall",
            name="A",
            prompt="p",
            passed=True,
            top_k=[],
            actual_token="x",
            actual_prob=0.5,
            activation_summary=ActivationSummary(
                peak_layers=[0],
                peak_activation=1.0,
                active_layer_count=1,
            ),
            interpretation="ok",
        ),
        TestResult(
            test_id="test_b",
            category="syntactic",
            name="B",
            prompt="p",
            passed=True,
            top_k=[],
            actual_token="y",
            actual_prob=0.5,
            activation_summary=ActivationSummary(
                peak_layers=[1],
                peak_activation=1.0,
                active_layer_count=1,
            ),
            interpretation="ok",
        ),
    ]

    sae_info = {
        "d_sae": 24576,
        "neuronpedia_url_template": "https://neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_idx}",
    }

    summary = engine._build_cross_test_summary(
        {"test_a": features_a, "test_b": features_b},
        results,
        layer_idx=5,
        sae_info=sae_info,
        loc="en",
    )

    assert summary.layer_idx == 5
    assert summary.d_sae == 24576
    assert summary.total_unique_features == 3  # 10, 20, 30
    # Feature #10 should be in cross_test_features (2 tests)
    cross_idxs = [cf.feature_idx for cf in summary.cross_test_features]
    assert 10 in cross_idxs
    assert summary.interpretation
