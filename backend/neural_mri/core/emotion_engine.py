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
from neural_mri.core.sae_manager import SAEManager
from neural_mri.core.sae_registry import get_sae_info
from neural_mri.schemas.emotion import (
    EmotionActivation,
    ExtractProbesRequest,
    ExtractProbesResponse,
    LayerEmotionPoint,
    LayerEvolutionRequest,
    LayerEvolutionResponse,
    PCAResponse,
    ProjectRequest,
    ProjectResponse,
    SAEFeatureDiff,
    SteerComparison,
    SteerRequest,
    SteerResponse,
    SteerSAERequest,
    SteerSAEResponse,
    SweepPoint,
    SweepRequest,
    SweepResponse,
    TokenEmotionProfile,
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
    # Token-level Projection
    # ------------------------------------------------------------------ #

    def project(self, req: ProjectRequest) -> ProjectResponse:
        """Project each token's residual stream onto all emotion vectors."""
        start = time.time()
        model = self._mm.get_model()
        model_id = self._mm.model_id
        layer_idx = self._resolve_layer(req.layer_idx)
        hook_name = f"blocks.{layer_idx}.hook_resid_post"

        # Ensure probes
        probes = self._get_cached_probes(model_id, layer_idx)
        if probes is None:
            self.extract_probes(ExtractProbesRequest(layer_idx=layer_idx))
            probes = self._get_cached_probes(model_id, layer_idx)

        # Filter emotions if requested
        emotions = req.emotions or sorted(probes.keys())
        selected = {e: probes[e] for e in emotions if e in probes}

        tokens = model.to_tokens(req.prompt)
        str_tokens = model.to_str_tokens(req.prompt)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            resid = cache[hook_name][0].float()  # [seq_len, d_model]

        # Normalize probe vectors once
        norm_probes = {e: v / (v.norm() + 1e-8) for e, v in selected.items()}

        token_profiles: list[TokenEmotionProfile] = []
        for t_idx in range(resid.shape[0]):
            act = resid[t_idx]
            activations = {e: round(torch.dot(act, nv).item(), 4) for e, nv in norm_probes.items()}
            token_profiles.append(
                TokenEmotionProfile(
                    token_idx=t_idx,
                    token_str=str(str_tokens[t_idx]),
                    activations=activations,
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        return ProjectResponse(
            model_id=model_id,
            prompt=req.prompt,
            layer_idx=layer_idx,
            emotions=list(selected.keys()),
            tokens=token_profiles,
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
        )

    # ------------------------------------------------------------------ #
    # PCA
    # ------------------------------------------------------------------ #

    def pca(self, layer_idx: int | None = None) -> PCAResponse:
        """Compute 2D PCA of emotion vectors (valence/arousal axes)."""
        start = time.time()
        self._mm.get_model()  # ensure loaded
        model_id = self._mm.model_id
        layer_idx = self._resolve_layer(layer_idx)

        probes = self._get_cached_probes(model_id, layer_idx)
        if probes is None:
            self.extract_probes(ExtractProbesRequest(layer_idx=layer_idx))
            probes = self._get_cached_probes(model_id, layer_idx)

        emotions = sorted(probes.keys())
        mat = torch.stack([probes[e] for e in emotions])  # [n_emotions, d_model]

        # Center
        mat_centered = mat - mat.mean(dim=0, keepdim=True)

        # SVD for PCA
        U, S, Vh = torch.linalg.svd(mat_centered, full_matrices=False)
        total_var = (S**2).sum().item()
        pc1_var = round((S[0] ** 2).item() / total_var, 4)
        pc2_var = round((S[1] ** 2).item() / total_var, 4)

        # Project onto first 2 PCs
        proj = mat_centered @ Vh[:2].T  # [n_emotions, 2]

        elapsed_ms = (time.time() - start) * 1000
        return PCAResponse(
            model_id=model_id,
            layer_idx=layer_idx,
            emotions=emotions,
            pc1=[round(v, 4) for v in proj[:, 0].tolist()],
            pc2=[round(v, 4) for v in proj[:, 1].tolist()],
            variance_explained=[pc1_var, pc2_var],
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
        )

    # ------------------------------------------------------------------ #
    # Strength Sweep
    # ------------------------------------------------------------------ #

    def sweep(self, req: SweepRequest) -> SweepResponse:
        """Run steering at multiple strengths, return dose-response curve."""
        start = time.time()
        model = self._mm.get_model()
        model_id = self._mm.model_id
        n_layers = model.cfg.n_layers
        probe_layer = n_layers * 2 // 3

        probes = self._get_cached_probes(model_id, probe_layer)
        if probes is None:
            self.extract_probes(ExtractProbesRequest(layer_idx=probe_layer))
            probes = self._get_cached_probes(model_id, probe_layer)

        if req.emotion not in probes:
            raise ValueError(f"Emotion '{req.emotion}' not found.")

        emotion_vec = probes[req.emotion]
        steer_dir = emotion_vec / (emotion_vec.norm() + 1e-8)
        probe_vec_norm = emotion_vec / (emotion_vec.norm() + 1e-8)
        probe_hook = f"blocks.{probe_layer}.hook_resid_post"
        hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
        tokens = model.to_tokens(req.prompt)

        points: list[SweepPoint] = []

        for strength in req.strengths:

            def make_hook(s):
                def hook_fn(value, hook):
                    resid_norm = value.norm(dim=-1, keepdim=True).mean()
                    return value + steer_dir.to(value.device, value.dtype) * s * resid_norm

                return hook_fn

            with torch.no_grad():
                if abs(strength) < 1e-9:
                    # No steering
                    output = model.generate(
                        tokens, max_new_tokens=req.max_new_tokens, do_sample=False
                    )
                    _, cache = model.run_with_cache(tokens)
                else:
                    # Steered generation
                    hook_fn = make_hook(strength)
                    steered_ids = tokens.clone()
                    for _ in range(req.max_new_tokens):
                        logits = model.run_with_hooks(
                            steered_ids,
                            fwd_hooks=[(n, hook_fn) for n in hook_names],
                        )
                        next_tok = logits[0, -1].argmax(dim=-1, keepdim=True)
                        steered_ids = torch.cat([steered_ids, next_tok.unsqueeze(0)], dim=-1)
                        if next_tok.item() == model.tokenizer.eos_token_id:
                            break
                    output = steered_ids

                    # Activation measurement
                    for n in hook_names:
                        model.add_hook(n, hook_fn)
                    try:
                        _, cache = model.run_with_cache(tokens)
                    finally:
                        model.reset_hooks()

                text = model.tokenizer.decode(
                    output[0, tokens.shape[1] :], skip_special_tokens=True
                )
                act = cache[probe_hook][0, -1].float()
                target_act = torch.dot(act, probe_vec_norm).item()

            points.append(
                SweepPoint(
                    strength=strength,
                    target_emotion_activation=round(target_act, 4),
                    generated_text=text,
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info("Sweep: %s, %d strengths, %.1fms", req.emotion, len(req.strengths), elapsed_ms)

        return SweepResponse(
            model_id=model_id,
            prompt=req.prompt,
            emotion=req.emotion,
            points=points,
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
        )

    # ------------------------------------------------------------------ #
    # Layer Evolution
    # ------------------------------------------------------------------ #

    def layer_evolution(self, req: LayerEvolutionRequest) -> LayerEvolutionResponse:
        """Measure emotion activations across all layers at a specific token."""
        start = time.time()
        model = self._mm.get_model()
        model_id = self._mm.model_id
        n_layers = model.cfg.n_layers

        # Need probes at every layer — extract at default layer first for the vectors
        default_layer = n_layers * 2 // 3
        probes = self._get_cached_probes(model_id, default_layer)
        if probes is None:
            self.extract_probes(ExtractProbesRequest(layer_idx=default_layer))
            probes = self._get_cached_probes(model_id, default_layer)

        emotions = req.emotions or sorted(probes.keys())
        # Note: we reuse the same probe directions across layers. This is an approximation
        # (Anthropic found structure is stable across mid-to-late layers).
        selected = {e: probes[e] / (probes[e].norm() + 1e-8) for e in emotions if e in probes}

        tokens = model.to_tokens(req.prompt)
        str_tokens = model.to_str_tokens(req.prompt)
        seq_len = tokens.shape[1]
        t_idx = req.token_idx if req.token_idx >= 0 else seq_len - 1
        t_idx = min(t_idx, seq_len - 1)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        layers: list[LayerEmotionPoint] = []
        for i in range(n_layers):
            hook = f"blocks.{i}.hook_resid_post"
            act = cache[hook][0, t_idx].float()
            activations = {e: round(torch.dot(act, nv).item(), 4) for e, nv in selected.items()}
            layers.append(LayerEmotionPoint(layer_idx=i, activations=activations))

        elapsed_ms = (time.time() - start) * 1000
        return LayerEvolutionResponse(
            model_id=model_id,
            prompt=req.prompt,
            token_idx=t_idx,
            token_str=str(str_tokens[t_idx]),
            emotions=list(selected.keys()),
            layers=layers,
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
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
            raise ValueError(f"Emotion '{req.emotion}' not found. Available: {available}")

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

        def steering_hook(value, hook):
            resid_norm = value.norm(dim=-1, keepdim=True).mean()
            scaled_dir = steer_dir.to(value.device, value.dtype) * req.strength * resid_norm
            return value + scaled_dir

        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in steer_layers]

        # --- (1) Original generation (greedy, no hooks) ---
        with torch.no_grad():
            original_output = model.generate(
                tokens,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
            )
            original_text = model.tokenizer.decode(
                original_output[0, tokens.shape[1] :],
                skip_special_tokens=True,
            )

            # Original emotion activations at last prompt token
            _, orig_cache = model.run_with_cache(tokens)
            orig_act = orig_cache[probe_hook][0, -1].float()
            orig_emotions = self._project_activations(orig_act, probes)

        # --- (2) Steered generation (manual greedy loop with hooks) ---
        with torch.no_grad():
            steered_ids = tokens.clone()
            for _ in range(req.max_new_tokens):
                steered_logits = model.run_with_hooks(
                    steered_ids,
                    fwd_hooks=[(name, steering_hook) for name in hook_names],
                )
                next_token = steered_logits[0, -1].argmax(dim=-1, keepdim=True)
                steered_ids = torch.cat([steered_ids, next_token.unsqueeze(0)], dim=-1)
                if next_token.item() == model.tokenizer.eos_token_id:
                    break

            steered_text = model.tokenizer.decode(
                steered_ids[0, tokens.shape[1] :],
                skip_special_tokens=True,
            )

            # Steered emotion activations (use add_hook since run_with_cache
            # does not accept fwd_hooks in TransformerLens)
            hook_handles = []
            for name in hook_names:
                hook_handles.append(model.add_hook(name, steering_hook))
            try:
                _, steered_cache = model.run_with_cache(tokens)
            finally:
                model.reset_hooks()
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

    # ------------------------------------------------------------------ #
    # Steer + SAE Combined
    # ------------------------------------------------------------------ #

    def steer_sae(self, req: SteerSAERequest, sae_mgr: SAEManager) -> SteerSAEResponse:
        """Run steering and capture SAE feature changes at the target layer."""
        start = time.time()
        model = self._mm.get_model()
        model_id = self._mm.model_id
        device = str(model.cfg.device)
        n_layers = model.cfg.n_layers

        # Ensure probes exist
        probe_layer = n_layers * 2 // 3
        probes = self._get_cached_probes(model_id, probe_layer)
        if probes is None:
            self.extract_probes(ExtractProbesRequest(layer_idx=probe_layer))
            probes = self._get_cached_probes(model_id, probe_layer)

        if req.emotion not in probes:
            available = sorted(probes.keys())
            raise ValueError(f"Emotion '{req.emotion}' not found. Available: {available}")

        # SAE setup
        sae_info = get_sae_info(model_id)
        if sae_info is None:
            raise ValueError(f"No SAE available for model: {model_id}")

        sae_layer = req.sae_layer_idx if req.sae_layer_idx is not None else probe_layer
        if sae_layer not in sae_info["layers"]:
            sae_layer = sae_info["layers"][len(sae_info["layers"]) // 2]

        provider = sae_mgr.get_provider(model_id, sae_layer, device)
        sae_hook = provider.hook_name

        # Steering setup
        emotion_vec = probes[req.emotion]
        steer_dir = emotion_vec / (emotion_vec.norm() + 1e-8)

        if req.layer_range is not None:
            steer_layers = req.layer_range
        else:
            steer_layers = list(range(n_layers))

        steer_hook_names = [f"blocks.{layer}.hook_resid_post" for layer in steer_layers]

        tokens = model.to_tokens(req.prompt)

        def steering_hook(value, hook):
            resid_norm = value.norm(dim=-1, keepdim=True).mean()
            scaled_dir = steer_dir.to(value.device, value.dtype) * req.strength * resid_norm
            return value + scaled_dir

        with torch.no_grad():
            # --- (1) Original: plain forward pass + SAE encode ---
            _, orig_cache = model.run_with_cache(tokens)
            orig_acts = orig_cache[sae_hook]  # [1, seq_len, d_model]
            orig_enc = provider.encode(orig_acts)

            # --- (2) Steered: add hooks + forward pass + SAE encode ---
            for name in steer_hook_names:
                model.add_hook(name, steering_hook)
            try:
                _, steered_cache = model.run_with_cache(tokens)
            finally:
                model.reset_hooks()
            steered_acts = steered_cache[sae_hook]
            steered_enc = provider.encode(steered_acts)

        # Compare at last prompt token
        last_idx = tokens.shape[1] - 1
        top_k = req.top_k

        # Original top-k
        orig_top_acts = orig_enc.top_acts[0, last_idx, :top_k]
        orig_top_idxs = orig_enc.top_indices[0, last_idx, :top_k]
        steered_top_acts = steered_enc.top_acts[0, last_idx, :top_k]
        steered_top_idxs = steered_enc.top_indices[0, last_idx, :top_k]

        # Build activation maps for both
        orig_map: dict[int, float] = {}
        for idx, act in zip(orig_top_idxs.tolist(), orig_top_acts.tolist()):
            orig_map[idx] = act

        steered_map: dict[int, float] = {}
        for idx, act in zip(steered_top_idxs.tolist(), steered_top_acts.tolist()):
            steered_map[idx] = act

        # All features that appear in either top-k
        all_features = sorted(set(orig_map.keys()) | set(steered_map.keys()))

        # Build diff list
        all_diffs: list[SAEFeatureDiff] = []
        for feat_idx in all_features:
            o = orig_map.get(feat_idx, 0.0)
            s = steered_map.get(feat_idx, 0.0)
            all_diffs.append(
                SAEFeatureDiff(
                    feature_idx=feat_idx,
                    original_activation=round(o, 4),
                    steered_activation=round(s, 4),
                    diff=round(s - o, 4),
                )
            )

        # Original top features (sorted by original activation)
        original_top = sorted(all_diffs, key=lambda x: -x.original_activation)[:top_k]

        # Steered top features (sorted by steered activation)
        steered_top = sorted(all_diffs, key=lambda x: -x.steered_activation)[:top_k]

        # Top changed features (sorted by absolute diff)
        top_changed = sorted(all_diffs, key=lambda x: -abs(x.diff))[:top_k]

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Steer+SAE: %s strength=%.3f, SAE layer %d, %d features compared, %.1fms",
            req.emotion,
            req.strength,
            sae_layer,
            len(all_diffs),
            elapsed_ms,
        )

        return SteerSAEResponse(
            model_id=model_id,
            prompt=req.prompt,
            emotion=req.emotion,
            strength=req.strength,
            sae_layer_idx=sae_layer,
            sae_hook_name=sae_hook,
            d_sae=provider.d_sae,
            original_top_features=original_top,
            steered_top_features=steered_top,
            top_changed_features=top_changed,
            metadata={
                "probe_layer": probe_layer,
                "n_steer_layers": len(steer_layers),
                "n_features_compared": len(all_diffs),
                "top_k": top_k,
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    def clear_cache(self, model_id: str | None = None) -> None:
        """Clear cached probes for a model (or all)."""
        if model_id:
            self._probes.pop(model_id, None)
        else:
            self._probes.clear()
