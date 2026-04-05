from __future__ import annotations

from pydantic import BaseModel, Field

# --- Probe Extraction ---


class ExtractProbesRequest(BaseModel):
    layer_idx: int | None = None  # None = auto (n_layers * 2 // 3)
    mode: str = "comprehension"  # "comprehension" (base+instruct) or "generation" (instruct only)


class EmotionProbe(BaseModel):
    emotion: str
    layer_idx: int
    n_passages: int  # how many passages were averaged


class ExtractProbesResponse(BaseModel):
    model_id: str
    layer_idx: int
    mode: str
    n_emotions: int
    emotions: list[str]
    metadata: dict


# --- Steering ---


class SteerRequest(BaseModel):
    prompt: str
    emotion: str  # which emotion vector to steer with
    strength: float = Field(default=0.05, ge=-1.0, le=1.0)  # fraction of residual stream norm
    layer_range: list[int] | None = None  # None = all layers
    max_new_tokens: int = Field(default=50, ge=1, le=200)


class SteerComparison(BaseModel):
    """Side-by-side: original vs steered output."""

    original_text: str
    steered_text: str
    emotion: str
    strength: float
    layer_range: list[int]


class EmotionActivation(BaseModel):
    emotion: str
    activation: float  # dot product with emotion vector
    activation_normalized: float  # 0-1


class SteerResponse(BaseModel):
    model_id: str
    prompt: str
    comparison: SteerComparison
    # Emotion probe activations on steered vs original (at last token)
    original_emotions: list[EmotionActivation]
    steered_emotions: list[EmotionActivation]
    metadata: dict


# --- Steer + SAE ---


class SteerSAERequest(BaseModel):
    prompt: str
    emotion: str
    strength: float = Field(default=0.05, ge=-1.0, le=1.0)
    sae_layer_idx: int | None = None  # None = auto (probe layer)
    top_k: int = Field(default=20, ge=1, le=100)
    layer_range: list[int] | None = None


class SAEFeatureDiff(BaseModel):
    feature_idx: int
    original_activation: float
    steered_activation: float
    diff: float  # steered - original


class SteerSAEResponse(BaseModel):
    model_id: str
    prompt: str
    emotion: str
    strength: float
    sae_layer_idx: int
    sae_hook_name: str
    d_sae: int
    # Per-token original vs steered top features (at last prompt token)
    original_top_features: list[SAEFeatureDiff]
    steered_top_features: list[SAEFeatureDiff]
    # Features with largest activation change
    top_changed_features: list[SAEFeatureDiff]
    metadata: dict
