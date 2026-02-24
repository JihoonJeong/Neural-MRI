from __future__ import annotations

from pydantic import BaseModel


# --- T1: Structural Scan ---


class StructuralScanRequest(BaseModel):
    pass


class LayerStructure(BaseModel):
    layer_id: str  # "embed", "blocks.0.attn", "blocks.0.mlp", "unembed"
    layer_type: str  # "embedding" | "attention" | "mlp" | "output"
    layer_index: int | None = None
    param_count: int
    shape_info: dict


class ConnectionInfo(BaseModel):
    from_id: str
    to_id: str
    type: str  # "sequential" | "residual"


class StructuralData(BaseModel):
    model_id: str
    scan_mode: str = "T1"
    total_params: int
    layers: list[LayerStructure]
    connections: list[ConnectionInfo]
    metadata: dict


# --- T2: Weight Scan ---


class WeightScanRequest(BaseModel):
    layers: list[str] | None = None  # None = all layers


class LayerWeightStats(BaseModel):
    layer_id: str
    component: str
    mean: float
    std: float
    min_val: float
    max_val: float
    l2_norm: float
    shape: list[int]
    num_outliers: int
    histogram: list[float]  # 20-bin histogram


class WeightData(BaseModel):
    model_id: str
    scan_mode: str = "T2"
    layers: list[LayerWeightStats]
    metadata: dict


# --- fMRI: Activation Scan ---


class ActivationScanRequest(BaseModel):
    prompt: str
    layers: list[str] | None = None  # None = all layers
    aggregation: str = "l2"  # "l2" | "mean"


class LayerActivation(BaseModel):
    layer_id: str
    activations: list[float]  # per-token scalar activation (0-1 normalized)
    per_head: list[list[float]] | None = None  # [n_heads][n_tokens] for attn layers


class ActivationData(BaseModel):
    model_id: str
    scan_mode: str = "fMRI"
    tokens: list[str]  # tokenized prompt strings
    layers: list[LayerActivation]
    metadata: dict


# --- DTI: Circuit Scan ---


class CircuitScanRequest(BaseModel):
    prompt: str
    target_token_idx: int = -1  # which output token to trace (-1 = last)


class PathwayConnection(BaseModel):
    from_id: str
    to_id: str
    strength: float  # 0-1 normalized importance
    is_pathway: bool  # True if strength > threshold


class AttentionHead(BaseModel):
    layer_idx: int
    head_idx: int
    pattern: list[list[float]]  # [seq_len, seq_len] attention weights


class ComponentImportance(BaseModel):
    layer_id: str
    importance: float  # 0-1 normalized
    is_pathway: bool


class CircuitData(BaseModel):
    model_id: str
    scan_mode: str = "DTI"
    tokens: list[str]
    target_token_idx: int
    connections: list[PathwayConnection]
    components: list[ComponentImportance]
    attention_heads: list[AttentionHead]
    metadata: dict


# --- FLAIR: Anomaly Scan ---


class AnomalyScanRequest(BaseModel):
    prompt: str


class LayerAnomaly(BaseModel):
    layer_id: str
    anomaly_scores: list[float]  # per-token anomaly score (0-1)
    kl_scores: list[float]  # per-token KL divergence (0-1 normalized)
    entropy_scores: list[float]  # per-token entropy (0-1 normalized)


class AnomalyData(BaseModel):
    model_id: str
    scan_mode: str = "FLAIR"
    tokens: list[str]
    layers: list[LayerAnomaly]
    metadata: dict
