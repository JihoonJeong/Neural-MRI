from __future__ import annotations

from pydantic import BaseModel


class ModelLoadRequest(BaseModel):
    model_id: str = "gpt2"
    device: str = "auto"


class LayerConfig(BaseModel):
    layer_index: int
    layer_type: str  # "transformer_block"
    components: list[str]  # ["attn", "mlp"]
    num_heads: int
    d_model: int
    d_head: int
    d_mlp: int


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    n_params: int
    n_layers: int
    d_model: int
    d_vocab: int
    n_heads: int
    d_head: int
    d_mlp: int
    max_seq_len: int
    device: str
    layers: list[LayerConfig]
    dtype: str
