from __future__ import annotations

MODEL_REGISTRY: dict[str, dict] = {
    "gpt2": {
        "family": "gpt2",
        "display_name": "GPT-2 Small",
        "params": "124M",
        "tl_compat": True,
        "gated": False,
    },
    "gpt2-medium": {
        "family": "gpt2",
        "display_name": "GPT-2 Medium",
        "params": "355M",
        "tl_compat": True,
        "gated": False,
    },
    "EleutherAI/pythia-1.4b": {
        "family": "pythia",
        "display_name": "Pythia 1.4B",
        "params": "1.4B",
        "tl_compat": True,
        "gated": False,
    },
    "google/gemma-2-2b": {
        "family": "gemma",
        "display_name": "Gemma 2 2B",
        "params": "2B",
        "tl_compat": True,
        "gated": True,
    },
    "meta-llama/Llama-3.2-3B": {
        "family": "llama",
        "display_name": "Llama 3.2 3B",
        "params": "3.2B",
        "tl_compat": True,
        "gated": True,
    },
    "Qwen/Qwen2.5-3B": {
        "family": "qwen",
        "display_name": "Qwen 2.5 3B",
        "params": "3B",
        "tl_compat": True,
        "gated": False,
    },
    "mistralai/Mistral-7B-v0.3": {
        "family": "mistral",
        "display_name": "Mistral 7B v0.3",
        "params": "7.2B",
        "tl_compat": True,
        "gated": True,
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "family": "phi",
        "display_name": "Phi-3 Mini 3.8B",
        "params": "3.8B",
        "tl_compat": True,
        "gated": False,
    },
}


def list_models(loaded_model_id: str | None = None) -> list[dict]:
    """Return model list for frontend, with current load status."""
    result = []
    for model_id, meta in MODEL_REGISTRY.items():
        result.append(
            {
                "model_id": model_id,
                "display_name": meta["display_name"],
                "family": meta["family"],
                "params": meta["params"],
                "tl_compat": meta["tl_compat"],
                "gated": meta.get("gated", False),
                "is_loaded": model_id == loaded_model_id,
            }
        )
    return result


def get_model_info(model_id: str) -> dict | None:
    """Return registry metadata for a model, or None if not found."""
    return MODEL_REGISTRY.get(model_id)
