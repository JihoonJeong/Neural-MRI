from __future__ import annotations

from pydantic import BaseModel


class CausalTraceRequest(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token_idx: int = -1  # default: last token


class CausalTraceCell(BaseModel):
    component: str  # "embed", "blocks.0.attn", "blocks.0.mlp", ...
    layer_idx: int  # -1 for embed
    component_type: str  # "attn" | "mlp" | "embed"
    recovery_score: float  # 0-1


class CausalTraceResult(BaseModel):
    model_id: str
    clean_prompt: str
    corrupt_prompt: str
    target_token_idx: int
    clean_prediction: str
    corrupt_prediction: str
    cells: list[CausalTraceCell]
    n_layers: int
    metadata: dict
