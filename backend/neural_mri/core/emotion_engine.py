"""Emotion vector extraction and steering engine.

Implements the comprehension-based probe extraction method:
1. Forward-pass pre-written emotional passages through the model
2. Extract residual stream activations at last token of each passage
3. Average per-emotion, subtract global mean → emotion vectors
4. Use vectors for projection (heatmap) and steering (causal intervention)

Based on Anthropic "Emotion Concepts" (2026) methodology, adapted for SLMs.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch

from neural_mri.core.model_manager import ModelManager
from neural_mri.schemas.emotion import (
    EmotionActivation,
    EmotionProbe,
    ExtractProbesRequest,
    ExtractProbesResponse,
    SteerComparison,
    SteerRequest,
    SteerResponse,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TEXTS_FILE = _DATA_DIR / "emotion_comprehension_texts.csv"


def _load_comprehension_texts() -> dict[str, list[str]]:
    """Load emotion → [passage, ...] from bundled CSV."""
    texts: dict[str, list[str]] = {}
    with open(_TEXTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            emotion = row[0].strip()
            passage = row[2].strip()
            texts.setdefault(emotion, []).append(passage)
    return texts


class EmotionEngine:
    """Extract emotion probes and steer model behavior with emotion vectors."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager
        # Cached probes: {model_id: {layer_idx: {emotion: vector}}}
        self._probes: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        self._texts: dict[str, list[str]] | None = None

    def _get_texts(self) -> dict[str, list[str]]:
        if self._texts is None:
            self._texts = _load_comprehension_texts()
        return self._texts

    def _resolve_layer(self, req_layer: int | None) -> int:
        """Pick probe layer: user-specified or ~2/3 through the model."""
        model = self._mm.get_model()
        if req_layer is not None:
            return req_layer
        return model.cfg.n_layers * 2 // 3

    def _get_cached_probes(self, model_id: str, layer_idx: int) -> dict[str, torch.Tensor] | None:
        return self._probes.get(model_id, {}).get(layer_idx)

    def _cache_probes(self, model_id: str, layer_idx: int, probes: dict[str, torch.Tensor]) -> None:
        self._probes.setdefault(model_id, {})[layer_idx] = probes

    # ------------------------------------------------------------------ #
    # Probe Extraction
    # ------------------------------------------------------------------ #

    def extract_probes(self, req: ExtractProbesRequest) -> ExtractProbesResponse:
        """Extract emotion vectors via comprehension-mode forward passes."""
        start = time.time()
        model = self._mm.get_model()
        model_id = self._mm.model_id
        device = str(model.cfg.device)
        layer_idx = self._resolve_layer(req.layer_idx)
        hook_name = f"blocks.{layer_idx}.hook_resid_post"

        # Check cache
        cached = self._get_cached_probes(model_id, layer_idx)
        if cached is not None:
            elapsed_ms = (time.time() - start) * 1000
            return ExtractProbesResponse(
                model_id=model_id,
                layer_idx=layer_idx,
                mode=req.mode,
                n_emotions=len(cached),
                emotions=sorted(cached.keys()),
                metadata={
                    "cached": True,
                    "compute_time_ms": round(elapsed_ms, 1),
                },
            )

        texts = self._get_texts()

        # Forward pass each passage, collect last-token activations
        emotion_activations: dict[str, list[torch.Tensor]] = {}

        with torch.no_grad():
            for emotion, passages in texts.items():
                acts_list = []
                for passage in passages:
                    tokens = model.to_tokens(passage)
                    _, cache = model.run_with_cache(tokens)
                    resid = cache[hook_name]  # [1, seq_len, d_model]
                    last_act = resid[0, -1].float()  # [d_model]
                    acts_list.append(last_act)
                emotion_activations[emotion] = acts_list

        # Mean per emotion
        emotion_means: dict[str, torch.Tensor] = {}
        all_vecs: list[torch.Tensor] = []
        for emotion, acts in emotion_activations.items():
            mean_vec = torch.stack(acts).mean(dim=0)
            emotion_means[emotion] = mean_vec
            all_vecs.append(mean_vec)

        # Global mean (across all emotions)
        global_mean = torch.stack(all_vecs).mean(dim=0)

        # Emotion vectors = per-emotion mean - global mean
        probes: dict[str, torch.Tensor] = {}
        for emotion, mean_vec in emotion_means.items():
            probes[emotion] = mean_vec - global_mean

        self._cache_probes(model_id, layer_idx, probes)

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Emotion probes extracted: %d emotions, layer %d, %.1fms",
            len(probes),
            layer_idx,
            elapsed_ms,
        )

        return ExtractProbesResponse(
            model_id=model_id,
            layer_idx=layer_idx,
            mode=req.mode,
            n_emotions=len(probes),
            emotions=sorted(probes.keys()),
            metadata={
                "cached": False,
                "n_passages_per_emotion": {e: len(p) for e, p in texts.items()},
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    # ------------------------------------------------------------------ #
    # Emotion Projection (measure activations against probes)
    # ------------------------------------------------------------------ #

    def _project_activations(
        self,
        activations: torch.Tensor,
        probes: dict[str, torch.Tensor],
    ) -> list[EmotionActivation]:
        """Project activations onto emotion vectors, return sorted by magnitude."""
        results: list[tuple[str, float]] = []
        for emotion, vec in probes.items():
            # Cosine-weighted dot product: normalize the probe vector
            vec_norm = vec / (vec.norm() + 1e-8)
            dot = torch.dot(activations.float(), vec_norm).item()
            results.append((emotion, dot))

        # Normalize to 0-1 range for display
        vals = [r[1] for r in results]
        v_min, v_max = min(vals), max(vals)
        rng = v_max - v_min if v_max > v_min else 1.0

        return sorted(
            [
                EmotionActivation(
                    emotion=e,
                    activation=round(v, 4),
                    activation_normalized=round((v - v_min) / rng, 4),
                )
                for e, v in results
            ],
            key=lambda x: -x.activation,
        )

    # ------------------------------------------------------------------ #
    # Steering
    # ------------------------------------------------------------------ #

    def steer(self, req: SteerRequest) -> SteerResponse:
        """Run prompt with and without emotion vector steering, compare outputs."""
        start = time.time()
        model = self._mm.get_model()
        model_id = self._mm.model_id
        n_layers = model.cfg.n_layers

        # Ensure probes exist
        probe_layer = n_layers * 2 // 3
        probes = self._get_cached_probes(model_id, probe_layer)
        if probes is None:
            self.extract_probes(ExtractProbesRequest(layer_idx=probe_layer))
            probes = self._get_cached_probes(model_id, probe_layer)

        if req.emotion not in probes:
            available = sorted(probes.keys())
            raise ValueError(
                f"Emotion '{req.emotion}' not found. Available: {available}"
            )

        emotion_vec = probes[req.emotion]
        # Normalize to unit vector, then scale by strength * residual stream norm
        steer_dir = emotion_vec / (emotion_vec.norm() + 1e-8)

        # Layer range for steering
        if req.layer_range is not None:
            steer_layers = req.layer_range
        else:
            steer_layers = list(range(n_layers))

        tokens = model.to_tokens(req.prompt)
        probe_hook = f"blocks.{probe_layer}.hook_resid_post"

        # --- (1) Original generation ---
        with torch.no_grad():
            original_output = model.generate(
                tokens,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
            )
            original_text = model.tokenizer.decode(
                original_output[0, tokens.shape[1]:],
                skip_special_tokens=True,
            )

            # Get original emotion activations at last prompt token
            _, orig_cache = model.run_with_cache(tokens)
            orig_act = orig_cache[probe_hook][0, -1].float()
            orig_emotions = self._project_activations(orig_act, probes)

        # --- (2) Steered generation ---
        def steering_hook(value, hook):
            # Scale steering by residual stream norm at each position
            resid_norm = value.norm(dim=-1, keepdim=True).mean()
            scaled_dir = steer_dir.to(value.device, value.dtype) * req.strength * resid_norm
            return value + scaled_dir

        # Build hook list for all target layers
        fwd_hooks = [
            (f"blocks.{layer}.hook_resid_post", steering_hook)
            for layer in steer_layers
        ]

        with torch.no_grad():
            # Steered generation: inject hooks during each forward pass
            steered_output = model.generate(
                tokens,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                fwd_hooks=fwd_hooks,
            )
            steered_text = model.tokenizer.decode(
                steered_output[0, tokens.shape[1]:],
                skip_special_tokens=True,
            )

            # Get steered emotion activations
            steered_logits, steered_cache = model.run_with_cache(
                tokens,
                fwd_hooks=fwd_hooks,
            )
            steered_act = steered_cache[probe_hook][0, -1].float()
            steered_emotions = self._project_activations(steered_act, probes)

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Emotion steering: %s strength=%.3f, %d layers, %.1fms",
            req.emotion,
            req.strength,
            len(steer_layers),
            elapsed_ms,
        )

        return SteerResponse(
            model_id=model_id,
            prompt=req.prompt,
            comparison=SteerComparison(
                original_text=original_text,
                steered_text=steered_text,
                emotion=req.emotion,
                strength=req.strength,
                layer_range=steer_layers,
            ),
            original_emotions=orig_emotions,
            steered_emotions=steered_emotions,
            metadata={
                "probe_layer": probe_layer,
                "n_steer_layers": len(steer_layers),
                "max_new_tokens": req.max_new_tokens,
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    def clear_cache(self, model_id: str | None = None) -> None:
        """Clear cached probes for a model (or all)."""
        if model_id:
            self._probes.pop(model_id, None)
        else:
            self._probes.clear()
