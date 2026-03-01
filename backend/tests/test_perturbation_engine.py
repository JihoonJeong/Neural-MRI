"""Unit tests for PerturbationEngine — hook resolution, perturbation methods."""

from __future__ import annotations

import pytest

from neural_mri.core.perturbation_engine import PerturbationEngine
from neural_mri.schemas.causal_trace import CausalTraceRequest
from neural_mri.schemas.perturb import (
    AblateRequest,
    AmplifyRequest,
    PatchRequest,
    PatchResult,
    PerturbResult,
    ZeroOutRequest,
)

# ── _resolve_hook ─────────────────────────────────────────────────


def test_resolve_hook_embed():
    assert PerturbationEngine._resolve_hook("embed") == "hook_embed"


def test_resolve_hook_attn():
    assert PerturbationEngine._resolve_hook("blocks.3.attn") == "blocks.3.hook_attn_out"


def test_resolve_hook_mlp():
    assert PerturbationEngine._resolve_hook("blocks.5.mlp") == "blocks.5.hook_mlp_out"


def test_resolve_hook_unknown_raises():
    with pytest.raises(ValueError, match="Unknown component"):
        PerturbationEngine._resolve_hook("invalid.component")


# ── Perturbation methods ─────────────────────────────────────────


def test_zero_out_returns_result(mock_model_manager):
    engine = PerturbationEngine(mock_model_manager)
    req = ZeroOutRequest(component="blocks.0.attn", prompt="The capital of France is")
    result = engine.zero_out(req)

    assert isinstance(result, PerturbResult)
    assert result.perturbation_type == "zero_out"
    assert result.model_id == "gpt2"
    assert result.component == "blocks.0.attn"
    assert result.original is not None
    assert result.perturbed is not None
    assert len(result.top_k_original) == 5
    assert len(result.top_k_perturbed) == 5
    assert isinstance(result.logit_diff, float)
    assert isinstance(result.kl_divergence, float)


def test_amplify_returns_result(mock_model_manager):
    engine = PerturbationEngine(mock_model_manager)
    req = AmplifyRequest(component="blocks.0.mlp", factor=3.0, prompt="test prompt")
    result = engine.amplify(req)

    assert isinstance(result, PerturbResult)
    assert result.perturbation_type == "amplify"
    assert result.component == "blocks.0.mlp"


def test_ablate_returns_result(mock_model_manager):
    engine = PerturbationEngine(mock_model_manager)
    req = AblateRequest(component="blocks.1.attn", prompt="test prompt")
    result = engine.ablate(req)

    assert isinstance(result, PerturbResult)
    assert result.perturbation_type == "ablate"
    assert result.component == "blocks.1.attn"


def test_activation_patch_returns_result(mock_model_manager):
    engine = PerturbationEngine(mock_model_manager)
    req = PatchRequest(
        clean_prompt="The capital of France is",
        corrupt_prompt="The capital of Germany is",
        component="blocks.0.attn",
    )
    result = engine.activation_patch(req)

    assert isinstance(result, PatchResult)
    assert result.component == "blocks.0.attn"
    assert 0 <= result.recovery_score <= 1
    assert result.clean_prediction is not None
    assert result.corrupt_prediction is not None
    assert result.patched_prediction is not None


def test_causal_trace_cell_count(mock_model_manager):
    """With N_LAYERS=2, expect 5 cells: embed + 2*attn + 2*mlp."""
    engine = PerturbationEngine(mock_model_manager)
    req = CausalTraceRequest(
        clean_prompt="The capital of France is",
        corrupt_prompt="The capital of Germany is",
    )
    result = engine.causal_trace(req)

    assert result.n_layers == 2
    assert len(result.cells) == 5  # embed + 2*attn + 2*mlp
    # Check component types
    types = [c.component_type for c in result.cells]
    assert types.count("embed") == 1
    assert types.count("attn") == 2
    assert types.count("mlp") == 2


def test_causal_trace_recovery_scores_bounded(mock_model_manager):
    engine = PerturbationEngine(mock_model_manager)
    req = CausalTraceRequest(
        clean_prompt="The capital of France is",
        corrupt_prompt="The capital of Germany is",
    )
    result = engine.causal_trace(req)

    for cell in result.cells:
        assert 0 <= cell.recovery_score <= 1, f"{cell.component}: {cell.recovery_score}"


def test_kl_divergence_is_finite(mock_model_manager):
    engine = PerturbationEngine(mock_model_manager)
    req = ZeroOutRequest(component="blocks.0.attn", prompt="test")
    result = engine.zero_out(req)

    assert result.kl_divergence is not None
    import math
    assert math.isfinite(result.kl_divergence)
