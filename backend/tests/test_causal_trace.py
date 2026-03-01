"""Tests for the causal trace schema and API endpoint."""

from httpx import ASGITransport, AsyncClient

from neural_mri.main import app
from neural_mri.schemas.causal_trace import (
    CausalTraceCell,
    CausalTraceRequest,
    CausalTraceResult,
)


def test_causal_trace_request_defaults():
    req = CausalTraceRequest(
        clean_prompt="The capital of France is",
        corrupt_prompt="The capital of XXXXX is",
    )
    assert req.target_token_idx == -1


def test_causal_trace_cell_fields():
    cell = CausalTraceCell(
        component="blocks.0.attn",
        layer_idx=0,
        component_type="attn",
        recovery_score=0.75,
    )
    assert cell.recovery_score == 0.75
    assert cell.component_type == "attn"


def test_causal_trace_result_fields():
    result = CausalTraceResult(
        model_id="gpt2",
        clean_prompt="The capital of France is",
        corrupt_prompt="The capital of XXXXX is",
        target_token_idx=5,
        clean_prediction=" Paris",
        corrupt_prediction=" the",
        cells=[
            CausalTraceCell(
                component="embed",
                layer_idx=-1,
                component_type="embed",
                recovery_score=0.1,
            ),
            CausalTraceCell(
                component="blocks.0.attn",
                layer_idx=0,
                component_type="attn",
                recovery_score=0.5,
            ),
        ],
        n_layers=12,
        metadata={"compute_time_ms": 100.0},
    )
    assert len(result.cells) == 2
    assert result.n_layers == 12


async def test_causal_trace_no_model():
    """Causal trace endpoint returns 400 when no model is loaded."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/perturb/causal-trace",
            json={"clean_prompt": "hello", "corrupt_prompt": "xxxxx"},
        )
    assert resp.status_code == 400
