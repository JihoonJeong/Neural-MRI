"""Tests for ReportEngine static analysis methods."""

from neural_mri.core.report_engine import ReportEngine
from neural_mri.schemas.report import ReportFinding
from neural_mri.schemas.scan import (
    ActivationData,
    AnomalyData,
    CircuitData,
    ComponentImportance,
    ConnectionInfo,
    LayerActivation,
    LayerAnomaly,
    LayerStructure,
    LayerWeightStats,
    PathwayConnection,
    StructuralData,
    WeightData,
)

# ── Helpers ──


def _structural(n_layers=12):
    layers = [
        LayerStructure(
            layer_id="embed",
            layer_type="embedding",
            param_count=1000,
            shape_info={},
        ),
    ]
    for i in range(n_layers):
        layers.append(
            LayerStructure(
                layer_id=f"blocks.{i}.attn",
                layer_type="attention",
                layer_index=i,
                param_count=500,
                shape_info={},
            )
        )
        layers.append(
            LayerStructure(
                layer_id=f"blocks.{i}.mlp",
                layer_type="mlp",
                layer_index=i,
                param_count=500,
                shape_info={},
            )
        )
    layers.append(
        LayerStructure(
            layer_id="unembed",
            layer_type="output",
            param_count=1000,
            shape_info={},
        )
    )
    conns = [
        ConnectionInfo(
            from_id=layers[i].layer_id,
            to_id=layers[i + 1].layer_id,
            type="sequential",
        )
        for i in range(len(layers) - 1)
    ]
    return StructuralData(
        model_id="gpt2",
        total_params=124_000_000,
        layers=layers,
        connections=conns,
        metadata={},
    )


def _weight_data(l2_values, std=0.01):
    layers = [
        LayerWeightStats(
            layer_id=f"blocks.{i}.attn",
            component="W_Q",
            mean=0.0,
            std=std,
            min_val=-1.0,
            max_val=1.0,
            l2_norm=v,
            shape=[8, 8],
            num_outliers=0,
            histogram=[0.0] * 20,
        )
        for i, v in enumerate(l2_values)
    ]
    return WeightData(model_id="gpt2", layers=layers, metadata={})


def _activation_data(last_tok_vals):
    layers = [
        LayerActivation(layer_id=f"blocks.{i}", activations=[0.1, 0.2, 0.3, v])
        for i, v in enumerate(last_tok_vals)
    ]
    return ActivationData(model_id="gpt2", tokens=["a", "b", "c", "d"], layers=layers, metadata={})


def _circuit_data(pathway_count, total_count):
    comps = [
        ComponentImportance(layer_id=f"c{i}", importance=0.5, is_pathway=i < pathway_count)
        for i in range(total_count)
    ]
    conns = [
        PathwayConnection(
            from_id=f"c{i}",
            to_id=f"c{i + 1}",
            strength=0.5,
            is_pathway=i < pathway_count,
        )
        for i in range(total_count - 1)
    ]
    return CircuitData(
        model_id="gpt2",
        tokens=["a", "b"],
        target_token_idx=1,
        connections=conns,
        components=comps,
        attention_heads=[],
        metadata={},
    )


def _anomaly_data(scores):
    layers = [
        LayerAnomaly(layer_id=f"blocks.{i}", anomaly_scores=s, kl_scores=s, entropy_scores=s)
        for i, s in enumerate(scores)
    ]
    return AnomalyData(model_id="gpt2", tokens=["a", "b"], layers=layers, metadata={})


# ── T1 ──


def test_t1_normal_12_layers():
    findings = ReportEngine._analyze_t1(_structural(12), 124_000_000, "en")
    assert len(findings) == 1
    assert findings[0].severity == "normal"


def test_t1_shallow_3_layers():
    findings = ReportEngine._analyze_t1(_structural(3), 10_000_000, "en")
    assert findings[0].severity == "notable"
    assert any("Shallow" in d or "shallow" in d.lower() for d in findings[0].details)


def test_t1_deep_48_layers():
    findings = ReportEngine._analyze_t1(_structural(48), 1_000_000_000, "en")
    assert findings[0].severity == "notable"


# ── T2 ──


def test_t2_normal_no_outliers():
    findings = ReportEngine._analyze_t2(_weight_data([1.0, 1.1, 0.9, 1.0]), "en")
    assert findings[0].severity == "normal"


def test_t2_outlier_notable():
    findings = ReportEngine._analyze_t2(_weight_data([1.0, 1.0, 1.0, 1.0, 1.0, 50.0]), "en")
    assert findings[0].severity in ("notable", "warning")


def test_t2_near_zero_warning():
    findings = ReportEngine._analyze_t2(_weight_data([1.0, 1.0, 0.001]), "en")
    assert findings[0].severity == "warning"


def test_t2_empty_layers():
    data = WeightData(model_id="gpt2", layers=[], metadata={})
    findings = ReportEngine._analyze_t2(data, "en")
    assert findings[0].severity == "normal"


# ── fMRI ──


def test_fmri_no_high_activation():
    findings = ReportEngine._analyze_fmri(_activation_data([0.3, 0.4, 0.5]), "en")
    assert findings[0].severity == "normal"


def test_fmri_high_activation_notable():
    findings = ReportEngine._analyze_fmri(_activation_data([0.3, 0.8, 0.9]), "en")
    assert findings[0].severity == "notable"


# ── DTI ──


def test_dti_normal_density():
    findings = ReportEngine._analyze_dti(_circuit_data(2, 8), "en")
    assert findings[0].severity == "normal"


def test_dti_low_density():
    findings = ReportEngine._analyze_dti(_circuit_data(1, 10), "en")
    assert findings[0].severity == "notable"


# ── FLAIR ──


def test_flair_clean():
    findings = ReportEngine._analyze_flair(_anomaly_data([[0.1, 0.2], [0.3, 0.4]]), "en")
    assert findings[0].severity == "normal"


def test_flair_warning():
    findings = ReportEngine._analyze_flair(_anomaly_data([[0.1, 0.9], [0.85, 0.5]]), "en")
    assert findings[0].severity == "warning"


# ── Battery ──


def test_battery_all_passed():
    data = {"total_tests": 7, "passed": 7, "failed": 0, "results": []}
    findings = ReportEngine._analyze_battery(data, "en")
    assert findings[0].severity == "normal"


def test_battery_some_failed():
    data = {
        "total_tests": 7,
        "passed": 5,
        "failed": 2,
        "results": [
            {"name": "Capital France", "passed": False, "category": "factual_recall"},
            {"name": "Doctor Bias", "passed": False, "category": "gender_bias"},
        ],
    }
    findings = ReportEngine._analyze_battery(data, "en")
    assert findings[0].severity == "notable"


def test_battery_half_failed():
    data = {
        "total_tests": 6,
        "passed": 3,
        "failed": 3,
        "results": [{"name": f"test_{i}", "passed": False, "category": "cat"} for i in range(3)],
    }
    findings = ReportEngine._analyze_battery(data, "en")
    assert findings[0].severity == "warning"


# ── Impressions ──


def test_impressions_all_normal():
    findings = [ReportFinding(scan_mode="T1", severity="normal", title="t", details=[])]
    imps = ReportEngine._build_impressions(findings, "en")
    assert any("healthy" in i.text.lower() for i in imps)


def test_impressions_with_warnings():
    findings = [ReportFinding(scan_mode="T2", severity="warning", title="t", details=[])]
    imps = ReportEngine._build_impressions(findings, "en")
    assert any(i.severity == "warning" for i in imps)
