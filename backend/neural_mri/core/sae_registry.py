from __future__ import annotations

SAE_REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------ #
    # SAELens provider (Joseph Bloom / Neuronpedia)
    # ------------------------------------------------------------------ #
    "gpt2": {
        "provider": "saelens",
        "release": "gpt2-small-res-jb",
        "sae_id_template": "blocks.{layer}.hook_resid_pre",
        "layers": list(range(12)),
        "d_sae": 24576,
        "neuronpedia_url_template": "https://neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_idx}",
    },
    "google/gemma-2-2b": {
        "provider": "saelens",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "layers": list(range(26)),
        "d_sae": 16384,
        "neuronpedia_url_template": "https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_idx}",
    },
    # ------------------------------------------------------------------ #
    # EleutherAI provider (sparsify)
    # ------------------------------------------------------------------ #
    "meta-llama/Llama-3.1-8B": {
        "provider": "eleutherai",
        "hub_id": "EleutherAI/sae-llama-3.1-8b-32x",
        "hookpoint_template": "layers.{layer}.mlp",
        "tl_hook_template": "blocks.{layer}.hook_mlp_out",
        "layers": [23, 29],  # only these layers available in the repo
        "d_sae": None,  # determined at load time (expansion_factor * d_model)
    },
    "meta-llama/Llama-3-8B": {
        "provider": "eleutherai",
        "hub_id": "EleutherAI/sae-llama-3-8b-32x",
        "hookpoint_template": "layers.{layer}",
        "tl_hook_template": "blocks.{layer}.hook_resid_pre",
        "layers": list(range(32)),
        "d_sae": None,
    },
}


def get_sae_info(model_id: str) -> dict | None:
    """Return SAE registry entry for a model, or None if unsupported."""
    return SAE_REGISTRY.get(model_id)


def list_sae_support() -> dict[str, bool]:
    """Return {model_id: has_sae} for all registered models."""
    from neural_mri.core.model_registry import MODEL_REGISTRY

    return {mid: mid in SAE_REGISTRY for mid in MODEL_REGISTRY}
