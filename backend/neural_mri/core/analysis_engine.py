from __future__ import annotations

import logging
import time

import torch

from neural_mri.core.model_manager import ModelManager
from neural_mri.core.sae_manager import SAEManager
from neural_mri.core.sae_registry import get_sae_info
from neural_mri.schemas.scan import (
    ActivationData,
    ActivationScanRequest,
    AnomalyData,
    AnomalyScanRequest,
    AttentionHead,
    CircuitData,
    CircuitScanRequest,
    ComponentImportance,
    ConnectionInfo,
    LayerActivation,
    LayerAnomaly,
    LayerStructure,
    LayerWeightStats,
    PathwayConnection,
    SAEData,
    SAEFeatureInfo,
    SAEScanRequest,
    SAETokenFeatures,
    StructuralData,
    WeightData,
)
from neural_mri.utils.tensor_summary import tensor_stats

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """Performs scan analyses across different MRI modalities."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def scan_structural(self) -> StructuralData:
        """T1 scan: extract static architecture topology from model config."""
        start = time.time()
        model = self._mm.get_model()
        cfg = model.cfg

        layers: list[LayerStructure] = []

        # Embedding layer
        layers.append(
            LayerStructure(
                layer_id="embed",
                layer_type="embedding",
                layer_index=None,
                param_count=cfg.d_vocab * cfg.d_model,
                shape_info={"d_vocab": cfg.d_vocab, "d_model": cfg.d_model},
            )
        )

        # Transformer blocks: each has attn + mlp
        for i in range(cfg.n_layers):
            # Attention: Q, K, V, O projections
            attn_params = cfg.d_model * cfg.d_head * cfg.n_heads * 4
            layers.append(
                LayerStructure(
                    layer_id=f"blocks.{i}.attn",
                    layer_type="attention",
                    layer_index=i,
                    param_count=attn_params,
                    shape_info={
                        "n_heads": cfg.n_heads,
                        "d_head": cfg.d_head,
                        "d_model": cfg.d_model,
                    },
                )
            )

            # MLP: in + out projections
            mlp_params = cfg.d_model * cfg.d_mlp * 2
            layers.append(
                LayerStructure(
                    layer_id=f"blocks.{i}.mlp",
                    layer_type="mlp",
                    layer_index=i,
                    param_count=mlp_params,
                    shape_info={
                        "d_mlp": cfg.d_mlp,
                        "d_model": cfg.d_model,
                    },
                )
            )

        # Output / unembedding layer
        layers.append(
            LayerStructure(
                layer_id="unembed",
                layer_type="output",
                layer_index=None,
                param_count=cfg.d_model * cfg.d_vocab,
                shape_info={"d_model": cfg.d_model, "d_vocab": cfg.d_vocab},
            )
        )

        connections = self._build_connections(layers)
        elapsed_ms = (time.time() - start) * 1000

        return StructuralData(
            model_id=self._mm.model_id,
            total_params=sum(p.numel() for p in model.parameters()),
            layers=layers,
            connections=connections,
            metadata={
                "device": str(cfg.device),
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    @staticmethod
    def _build_connections(layers: list[LayerStructure]) -> list[ConnectionInfo]:
        """Build sequential connections between adjacent layer components."""
        connections: list[ConnectionInfo] = []
        for i in range(len(layers) - 1):
            connections.append(
                ConnectionInfo(
                    from_id=layers[i].layer_id,
                    to_id=layers[i + 1].layer_id,
                    type="sequential",
                )
            )
        return connections

    def scan_weights(self, layer_ids: list[str] | None = None) -> WeightData:
        """T2 scan: extract weight distribution statistics from state_dict."""
        start = time.time()
        model = self._mm.get_model()
        state_dict = model.state_dict()

        results: list[LayerWeightStats] = []
        for name, tensor in state_dict.items():
            # Filter by requested layers
            if layer_ids and not any(lid in name for lid in layer_ids):
                continue

            # Skip 1D tensors (biases, layer norms) — focus on weight matrices
            if tensor.ndim < 2:
                continue

            stats = tensor_stats(tensor)

            # Map parameter name to layer_id (e.g. "blocks.0.attn.W_Q" -> "blocks.0.attn")
            parts = name.split(".")
            component = parts[-1]  # e.g. "W_Q", "W_K", "W_in", "W_out"
            layer_id = ".".join(parts[:-1])  # e.g. "blocks.0.attn"

            results.append(
                LayerWeightStats(
                    layer_id=layer_id,
                    component=component,
                    mean=stats["mean"],
                    std=stats["std"],
                    min_val=stats["min_val"],
                    max_val=stats["max_val"],
                    l2_norm=stats["l2_norm"],
                    shape=stats["shape"],
                    num_outliers=stats["num_outliers"],
                    histogram=stats["histogram"],
                )
            )

        elapsed_ms = (time.time() - start) * 1000

        return WeightData(
            model_id=self._mm.model_id,
            layers=results,
            metadata={
                "num_tensors_scanned": len(results),
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    # ------------------------------------------------------------------ #
    # fMRI: Activation Scan
    # ------------------------------------------------------------------ #

    def scan_activation(self, req: ActivationScanRequest) -> ActivationData:
        """fMRI scan: run prompt through model, extract per-layer activations."""
        start = time.time()
        model = self._mm.get_model()
        cfg = model.cfg

        # Tokenize
        tokens = model.to_tokens(req.prompt)  # [1, seq_len]
        str_tokens = model.to_str_tokens(req.prompt)  # list of strings
        seq_len = tokens.shape[1]

        # Run with cache — single forward pass captures all activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        layers: list[LayerActivation] = []
        all_norms: list[float] = []  # collect all raw norms for global normalization

        # Embedding activation
        embed_act = cache["hook_embed"]  # [1, seq, d_model]
        embed_norms = torch.norm(embed_act[0], dim=-1).tolist()  # [seq]
        all_norms.extend(embed_norms)

        # Per-block activations
        block_attn_norms: list[list[float]] = []
        block_attn_heads: list[list[list[float]] | None] = []
        block_mlp_norms: list[list[float]] = []

        for i in range(cfg.n_layers):
            # Attention per-head output: hook_z shape [1, seq, n_heads, d_head]
            attn_z = cache[f"blocks.{i}.attn.hook_z"]  # [1, seq, heads, d_head]
            # Per-token aggregate: L2 norm across heads and d_head
            attn_norms = torch.norm(attn_z[0], dim=(-1, -2)).tolist()  # [seq]
            all_norms.extend(attn_norms)
            block_attn_norms.append(attn_norms)

            # Per-head activations: L2 norm per head per token
            per_head = torch.norm(attn_z[0], dim=-1)  # [seq, heads]
            head_max = per_head.max().item() or 1.0
            per_head_normalized = (per_head / head_max).T.tolist()  # [heads, seq]
            block_attn_heads.append(per_head_normalized)

            # MLP output
            mlp_out = cache[f"blocks.{i}.hook_mlp_out"]  # [1, seq, d_model]
            mlp_norms = torch.norm(mlp_out[0], dim=-1).tolist()  # [seq]
            all_norms.extend(mlp_norms)
            block_mlp_norms.append(mlp_norms)

        # Unembed: use residual stream at final layer
        resid_final = cache[f"blocks.{cfg.n_layers - 1}.hook_resid_post"]  # [1, seq, d_model]
        unembed_norms = torch.norm(resid_final[0], dim=-1).tolist()
        all_norms.extend(unembed_norms)

        # Global 0-1 normalization
        norm_max = max(all_norms) if all_norms else 1.0
        norm_min = min(all_norms) if all_norms else 0.0
        rng = norm_max - norm_min if norm_max > norm_min else 1.0

        def normalize(vals: list[float]) -> list[float]:
            return [round((v - norm_min) / rng, 4) for v in vals]

        # Build layer activation list (same order as T1 structural)
        layers.append(
            LayerActivation(
                layer_id="embed",
                activations=normalize(embed_norms),
            )
        )
        for i in range(cfg.n_layers):
            layers.append(
                LayerActivation(
                    layer_id=f"blocks.{i}.attn",
                    activations=normalize(block_attn_norms[i]),
                    per_head=block_attn_heads[i],
                )
            )
            layers.append(
                LayerActivation(
                    layer_id=f"blocks.{i}.mlp",
                    activations=normalize(block_mlp_norms[i]),
                )
            )
        layers.append(
            LayerActivation(
                layer_id="unembed",
                activations=normalize(unembed_norms),
            )
        )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "fMRI scan: %d tokens, %d layers, %.1fms",
            seq_len,
            len(layers),
            elapsed_ms,
        )

        return ActivationData(
            model_id=self._mm.model_id,
            tokens=[str(t) for t in str_tokens],
            layers=layers,
            metadata={
                "seq_len": seq_len,
                "n_layers": cfg.n_layers,
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    # ------------------------------------------------------------------ #
    # DTI: Circuit Scan
    # ------------------------------------------------------------------ #

    def scan_circuits(self, req: CircuitScanRequest) -> CircuitData:
        """DTI scan: trace important circuits via attention + zero-ablation."""
        start = time.time()
        model = self._mm.get_model()
        cfg = model.cfg

        tokens = model.to_tokens(req.prompt)  # [1, seq_len]
        str_tokens = model.to_str_tokens(req.prompt)
        seq_len = tokens.shape[1]
        target_idx = req.target_token_idx if req.target_token_idx >= 0 else seq_len - 1

        # --- (1) Baseline logits ---
        with torch.no_grad():
            baseline_logits, cache = model.run_with_cache(tokens)
        baseline_logit_at_target = baseline_logits[0, target_idx].clone()

        # --- (2) Extract attention patterns ---
        attention_heads: list[AttentionHead] = []
        for i in range(cfg.n_layers):
            pattern = cache[f"blocks.{i}.attn.hook_pattern"]  # [1, heads, seq, seq]
            for h in range(cfg.n_heads):
                attention_heads.append(
                    AttentionHead(
                        layer_idx=i,
                        head_idx=h,
                        pattern=pattern[0, h].tolist(),
                    )
                )

        # --- (3) Zero-ablation importance for each component ---
        # Components: embed + (attn + mlp) * n_layers + unembed = 2 + 2*n_layers
        component_ids: list[str] = ["embed"]
        hook_points: list[str] = ["hook_embed"]
        for i in range(cfg.n_layers):
            component_ids.append(f"blocks.{i}.attn")
            hook_points.append(f"blocks.{i}.hook_attn_out")
            component_ids.append(f"blocks.{i}.mlp")
            hook_points.append(f"blocks.{i}.hook_mlp_out")

        raw_importances: list[float] = []
        for comp_id, hook_name in zip(component_ids, hook_points):
            original_activation = cache[hook_name].clone()

            def zero_ablation_hook(value, hook, _orig=original_activation):
                return torch.zeros_like(value)

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, zero_ablation_hook)],
                )
            ablated_logit_at_target = ablated_logits[0, target_idx]

            # Importance = change in target logit distribution (L2 distance)
            diff = torch.norm(baseline_logit_at_target - ablated_logit_at_target).item()
            raw_importances.append(diff)

        # Normalize importance 0-1
        imp_max = max(raw_importances) if raw_importances else 1.0
        imp_min = min(raw_importances) if raw_importances else 0.0
        imp_rng = imp_max - imp_min if imp_max > imp_min else 1.0

        threshold = 0.3
        components: list[ComponentImportance] = []
        for comp_id, raw_imp in zip(component_ids, raw_importances):
            norm_imp = round((raw_imp - imp_min) / imp_rng, 4)
            components.append(
                ComponentImportance(
                    layer_id=comp_id,
                    importance=norm_imp,
                    is_pathway=norm_imp >= threshold,
                )
            )

        # Build connections between sequential components with importance-based strength
        comp_map = {c.layer_id: c for c in components}
        connections: list[PathwayConnection] = []
        for i in range(len(component_ids) - 1):
            from_id = component_ids[i]
            to_id = component_ids[i + 1]
            strength = (comp_map[from_id].importance + comp_map[to_id].importance) / 2
            connections.append(
                PathwayConnection(
                    from_id=from_id,
                    to_id=to_id,
                    strength=round(strength, 4),
                    is_pathway=strength >= threshold,
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "DTI scan: %d tokens, %d components, %d pathways, %.1fms",
            seq_len,
            len(components),
            sum(1 for c in connections if c.is_pathway),
            elapsed_ms,
        )

        return CircuitData(
            model_id=self._mm.model_id,
            tokens=[str(t) for t in str_tokens],
            target_token_idx=target_idx,
            connections=connections,
            components=components,
            attention_heads=attention_heads,
            metadata={
                "seq_len": seq_len,
                "n_components_ablated": len(component_ids),
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    # ------------------------------------------------------------------ #
    # FLAIR: Anomaly Scan (Logit Lens + Entropy)
    # ------------------------------------------------------------------ #

    def scan_anomaly(self, req: AnomalyScanRequest) -> AnomalyData:
        """FLAIR scan: detect anomalous regions via Logit Lens + Entropy."""
        start = time.time()
        model = self._mm.get_model()
        cfg = model.cfg

        tokens = model.to_tokens(req.prompt)  # [1, seq_len]
        str_tokens = model.to_str_tokens(req.prompt)
        seq_len = tokens.shape[1]

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        # Final prediction distribution (reference) — upcast to float32 for numerical stability
        final_logits = logits[0].float()  # [seq_len, d_vocab]
        final_probs = torch.softmax(final_logits, dim=-1)

        # Unembed matrix + final LayerNorm (critical for Logit Lens)
        W_U = model.W_U.float()  # [d_model, d_vocab]
        b_U = model.b_U.float() if model.b_U is not None else 0
        ln_final = model.ln_final

        all_kl: list[torch.Tensor] = []
        all_ent: list[torch.Tensor] = []

        for i in range(cfg.n_layers):
            resid = cache[f"blocks.{i}.hook_resid_post"]  # [1, seq, d_model]

            # Logit Lens: apply final LayerNorm then unembed (float32 for log stability)
            ln_resid = ln_final(resid[0]).float()  # [seq, d_model]
            intermediate_logits = ln_resid @ W_U + b_U  # [seq, d_vocab]
            intermediate_probs = torch.softmax(intermediate_logits, dim=-1)

            # KL divergence: KL(final || intermediate) per token
            kl = torch.sum(
                final_probs
                * (torch.log(final_probs + 1e-10) - torch.log(intermediate_probs + 1e-10)),
                dim=-1,
            )  # [seq_len]
            all_kl.append(kl)

            # Entropy of intermediate distribution per token
            entropy = -torch.sum(
                intermediate_probs * torch.log(intermediate_probs + 1e-10),
                dim=-1,
            )
            all_ent.append(entropy)

        # Stack for global normalization
        kl_stack = torch.stack(all_kl)  # [n_layers, seq_len]
        ent_stack = torch.stack(all_ent)

        # 0-1 normalize globally
        kl_min, kl_max = kl_stack.min(), kl_stack.max()
        kl_rng = (kl_max - kl_min) if kl_max > kl_min else torch.tensor(1.0)
        kl_norm = (kl_stack - kl_min) / kl_rng

        ent_min, ent_max = ent_stack.min(), ent_stack.max()
        ent_rng = (ent_max - ent_min) if ent_max > ent_min else torch.tensor(1.0)
        ent_norm = (ent_stack - ent_min) / ent_rng

        # Anomaly score: weighted combination
        anomaly = 0.6 * kl_norm + 0.4 * ent_norm  # [n_layers, seq_len]

        layers: list[LayerAnomaly] = []
        for i in range(cfg.n_layers):
            layers.append(
                LayerAnomaly(
                    layer_id=f"blocks.{i}",
                    anomaly_scores=[round(v, 4) for v in anomaly[i].tolist()],
                    kl_scores=[round(v, 4) for v in kl_norm[i].tolist()],
                    entropy_scores=[round(v, 4) for v in ent_norm[i].tolist()],
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "FLAIR scan: %d tokens, %d layers, %.1fms",
            seq_len,
            len(layers),
            elapsed_ms,
        )

        return AnomalyData(
            model_id=self._mm.model_id,
            tokens=[str(t) for t in str_tokens],
            layers=layers,
            metadata={
                "seq_len": seq_len,
                "n_layers": cfg.n_layers,
                "alpha_kl": 0.6,
                "beta_entropy": 0.4,
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )

    # ------------------------------------------------------------------ #
    # SAE: Sparse Autoencoder Feature Scan
    # ------------------------------------------------------------------ #

    def scan_sae(self, req: SAEScanRequest, sae_mgr: SAEManager) -> SAEData:
        """Decode residual stream into sparse SAE features for a given layer."""
        start = time.time()
        model = self._mm.get_model()
        model_id = self._mm.model_id
        device = str(model.cfg.device)

        sae_info = get_sae_info(model_id)
        if sae_info is None:
            raise ValueError(f"No SAE available for model: {model_id}")

        # Load SAE for the target layer
        sae = sae_mgr.get_sae(model_id, req.layer_idx, device)
        # SAE-Lens >=4: hook_name is in metadata, not cfg directly
        hook_name = sae.cfg.metadata.get("hook_name") if sae.cfg.metadata else None
        if not hook_name:
            # Fallback: derive from sae_id_template
            hook_name = sae_info["sae_id_template"].format(layer=req.layer_idx)

        # Tokenize
        tokens = model.to_tokens(req.prompt)  # [1, seq_len]
        str_tokens = model.to_str_tokens(req.prompt)

        # Forward pass with cache
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract activations at the SAE hook point
        activations = cache[hook_name]  # [1, seq_len, d_model]

        # float16 models need upcast for SAE (which expects float32)
        activations = activations.float()

        # Encode → sparse features [1, seq_len, d_sae]
        features = sae.encode(activations)

        # Decode → reconstruction [1, seq_len, d_model]
        reconstruction = sae.decode(features)

        # Reconstruction loss (MSE per token, averaged)
        recon_loss = torch.mean((activations - reconstruction) ** 2).item()

        # Sparsity: fraction of features with activation > 0
        features_2d = features[0]  # [seq_len, d_sae]
        active_mask = features_2d > 0
        sparsity = active_mask.float().mean().item()

        # Per-token top-K features
        seq_len = features_2d.shape[0]
        top_k = min(req.top_k, features_2d.shape[1])

        neuronpedia_template = sae_info.get("neuronpedia_url_template")

        token_features_list: list[SAETokenFeatures] = []
        all_active_indices: set[int] = set()

        # Find global max for normalization
        global_max = features_2d.max().item() if features_2d.numel() > 0 else 1.0
        global_max = max(global_max, 1e-8)

        for t_idx in range(seq_len):
            tok_feats = features_2d[t_idx]  # [d_sae]
            topk_vals, topk_idxs = torch.topk(tok_feats, top_k)

            feat_infos: list[SAEFeatureInfo] = []
            for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
                np_url = None
                if neuronpedia_template:
                    np_url = neuronpedia_template.format(
                        layer=req.layer_idx, feature_idx=idx
                    )
                feat_infos.append(
                    SAEFeatureInfo(
                        feature_idx=idx,
                        activation=round(val, 4),
                        activation_normalized=round(val / global_max, 4),
                        neuronpedia_url=np_url,
                    )
                )
                if val > 0:
                    all_active_indices.add(idx)

            token_features_list.append(
                SAETokenFeatures(
                    token_idx=t_idx,
                    token_str=str(str_tokens[t_idx]),
                    top_features=feat_infos,
                )
            )

        # Heatmap: rows = tokens, cols = union of active feature indices
        heatmap_feature_indices = sorted(all_active_indices)
        idx_to_col = {idx: col for col, idx in enumerate(heatmap_feature_indices)}
        n_feats = len(heatmap_feature_indices)

        heatmap_values: list[list[float]] = []
        for t_idx in range(seq_len):
            row = [0.0] * n_feats
            tok_feats = features_2d[t_idx]
            for feat_idx in heatmap_feature_indices:
                col = idx_to_col[feat_idx]
                row[col] = round(tok_feats[feat_idx].item() / global_max, 4)
            heatmap_values.append(row)

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "SAE scan: layer %d, %d tokens, %d active features, %.1fms",
            req.layer_idx,
            seq_len,
            n_feats,
            elapsed_ms,
        )

        return SAEData(
            model_id=model_id,
            prompt=req.prompt,
            layer_idx=req.layer_idx,
            hook_name=hook_name,
            d_sae=sae.cfg.d_sae,
            tokens=[str(t) for t in str_tokens],
            token_features=token_features_list,
            reconstruction_loss=round(recon_loss, 6),
            sparsity=round(sparsity, 4),
            heatmap_feature_indices=heatmap_feature_indices,
            heatmap_values=heatmap_values,
            metadata={
                "seq_len": seq_len,
                "top_k": top_k,
                "total_active_features": n_feats,
                "compute_time_ms": round(elapsed_ms, 1),
            },
        )
