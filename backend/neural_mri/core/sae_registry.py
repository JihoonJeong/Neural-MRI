from __future__ import annotations

SAE_REGISTRY: dict[str, dict] = {
    "gpt2": {
        "release": "gpt2-small-res-jb",
        "sae_id_template": "blocks.{layer}.hook_resid_pre",
        "layers": list(range(12)),
        "d_sae": 24576,
        "neuronpedia_url_template": "https://neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_idx}",
    },
    "google/gemma-2-2b": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "layers": list(range(26)),
        "d_sae": 16384,
        "neuronpedia_url_template": "https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_idx}",
    },
}


def get_sae_info(model_id: str) -> dict | None:
    """Return SAE registry entry for a model, or None if unsupported."""
    return SAE_REGISTRY.get(model_id)


def list_sae_support() -> dict[str, bool]:
    """Return {model_id: has_sae} for all registered models."""
    from neural_mri.core.model_registry import MODEL_REGISTRY

    return {mid: mid in SAE_REGISTRY for mid in MODEL_REGISTRY}
