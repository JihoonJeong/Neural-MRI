"""Tests for AnalysisEngine — mock model scans."""

from neural_mri.core.analysis_engine import AnalysisEngine


def test_scan_structural_layer_count(mock_model_manager):
    engine = AnalysisEngine(mock_model_manager)
    data = engine.scan_structural()
    # embed + 2*(attn+mlp) + unembed = 1 + 4 + 1 = 6
    assert len(data.layers) == 6
    assert data.layers[0].layer_type == "embedding"
    assert data.layers[-1].layer_type == "output"


def test_scan_structural_connections(mock_model_manager):
    engine = AnalysisEngine(mock_model_manager)
    data = engine.scan_structural()
    assert len(data.connections) == len(data.layers) - 1


def test_scan_structural_metadata(mock_model_manager):
    engine = AnalysisEngine(mock_model_manager)
    data = engine.scan_structural()
    assert "device" in data.metadata
    assert "compute_time_ms" in data.metadata


def test_scan_weights_returns_stats(mock_model_manager):
    engine = AnalysisEngine(mock_model_manager)
    data = engine.scan_weights()
    assert len(data.layers) > 0
    for layer in data.layers:
        assert layer.l2_norm > 0
        assert len(layer.histogram) == 20


def test_scan_activation_returns_normalized(mock_model_manager):
    engine = AnalysisEngine(mock_model_manager)
    from neural_mri.schemas.scan import ActivationScanRequest

    data = engine.scan_activation(ActivationScanRequest(prompt="test"))
    for layer in data.layers:
        for val in layer.activations:
            assert 0.0 <= val <= 1.0


def test_scan_anomaly_layer_count(mock_model_manager):
    engine = AnalysisEngine(mock_model_manager)
    from neural_mri.schemas.scan import AnomalyScanRequest

    data = engine.scan_anomaly(AnomalyScanRequest(prompt="test"))
    assert len(data.layers) == 2  # n_layers=2
