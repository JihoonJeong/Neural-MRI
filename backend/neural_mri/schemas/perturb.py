from __future__ import annotations

from pydantic import BaseModel

# --- Request Models ---


class ZeroOutRequest(BaseModel):
    component: str  # e.g. "blocks.3.attn", "blocks.5.mlp"
    prompt: str


class AmplifyRequest(BaseModel):
    component: str
    factor: float = 2.0
    prompt: str


class AblateRequest(BaseModel):
    component: str  # mean ablation target
    prompt: str


class PatchRequest(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    component: str
    target_token_idx: int = -1


# --- Response Models ---


class TokenPrediction(BaseModel):
    token: str
    logit: float
    prob: float


class PerturbResult(BaseModel):
    model_id: str
    component: str
    perturbation_type: str  # "zero_out" | "amplify" | "ablate"
    original: TokenPrediction
    perturbed: TokenPrediction
    top_k_original: list[TokenPrediction]
    top_k_perturbed: list[TokenPrediction]
    logit_diff: float
    kl_divergence: float
    metadata: dict


class PatchResult(BaseModel):
    model_id: str
    component: str
    clean_prompt: str
    corrupt_prompt: str
    clean_prediction: TokenPrediction
    corrupt_prediction: TokenPrediction
    patched_prediction: TokenPrediction
    recovery_score: float  # 0-1, how much patching restored clean behavior
    metadata: dict
