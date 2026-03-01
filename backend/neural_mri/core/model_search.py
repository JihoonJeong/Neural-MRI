from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Known TransformerLens-supported architecture classes.
# Kept in sync with transformer_lens loading.py OFFICIAL_MODEL_NAMES.
TL_SUPPORTED_ARCHITECTURES: set[str] = {
    "GPT2LMHeadModel",
    "GPTNeoForCausalLM",
    "GPTNeoXForCausalLM",
    "GPTJForCausalLM",
    "OPTForCausalLM",
    "LlamaForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Phi3ForCausalLM",
    "PhiForCausalLM",
    "Qwen2ForCausalLM",
    "BloomForCausalLM",
}

# Known model-id substrings that TransformerLens supports
TL_KNOWN_PREFIXES: set[str] = {
    "gpt2",
    "gpt-neo",
    "gpt-j",
    "pythia",
    "opt",
    "llama",
    "gemma",
    "mistral",
    "mixtral",
    "phi",
    "qwen",
    "bloom",
}


@dataclass
class SearchResult:
    model_id: str
    author: str | None
    downloads: int
    likes: int
    pipeline_tag: str | None
    gated: bool | str  # False, True, or "auto"
    tl_compat: bool | None  # None = unknown
    architectures: list[str]


@dataclass
class _SearchCache:
    """Simple TTL cache for search results."""

    _cache: dict[str, tuple[float, list[SearchResult]]] = field(
        default_factory=dict,
    )
    _ttl: float = 300.0  # 5 minutes

    def get(self, key: str) -> list[SearchResult] | None:
        if key in self._cache:
            ts, results = self._cache[key]
            if time.time() - ts < self._ttl:
                return results
            del self._cache[key]
        return None

    def put(self, key: str, results: list[SearchResult]) -> None:
        self._cache[key] = (time.time(), results)
        # Evict oldest if over 50 cached queries
        if len(self._cache) > 50:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]


_search_cache = _SearchCache()


def check_tl_compatibility(
    architectures: list[str],
    model_id: str,
) -> bool | None:
    """
    Best-effort check if a model is TransformerLens compatible.
    Returns True, False, or None (unknown).
    """
    if architectures:
        return any(arch in TL_SUPPORTED_ARCHITECTURES for arch in architectures)

    # Fallback: check model_id prefix against known TL models
    model_id_lower = model_id.lower()
    for prefix in TL_KNOWN_PREFIXES:
        if prefix in model_id_lower:
            return True

    return None


def search_models(
    query: str,
    limit: int = 20,
    filter_tl_compat: bool = False,
) -> list[SearchResult]:
    """Search HuggingFace Hub for models matching query."""
    cache_key = f"{query}::{limit}::{filter_tl_compat}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        logger.info("Model search cache HIT: %s", cache_key)
        return cached

    from huggingface_hub import HfApi

    hf = HfApi()
    results: list[SearchResult] = []

    try:
        fetch_limit = limit * 3 if filter_tl_compat else limit
        models = hf.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=fetch_limit,
            pipeline_tag="text-generation",
        )

        for m in models:
            # Extract architectures from config
            architectures: list[str] = []
            if hasattr(m, "config") and m.config:
                cfg = m.config
                if isinstance(cfg, dict):
                    architectures = cfg.get("architectures", []) or []

            tl_compat = check_tl_compatibility(architectures, m.id)

            if filter_tl_compat and not tl_compat:
                continue

            gated = getattr(m, "gated", False)
            if gated is None:
                gated = False

            results.append(
                SearchResult(
                    model_id=m.id,
                    author=getattr(m, "author", None),
                    downloads=getattr(m, "downloads", 0) or 0,
                    likes=getattr(m, "likes", 0) or 0,
                    pipeline_tag=getattr(m, "pipeline_tag", None),
                    gated=gated,
                    tl_compat=tl_compat,
                    architectures=architectures,
                )
            )

            if len(results) >= limit:
                break

    except Exception as e:
        logger.error("HuggingFace Hub search failed: %s", e)
        raise

    _search_cache.put(cache_key, results)
    return results
