from __future__ import annotations

import logging
import time

import torch

from neural_mri.core.model_manager import ModelManager
from neural_mri.schemas.causal_trace import (
    CausalTraceCell,
    CausalTraceRequest,
    CausalTraceResult,
)
from neural_mri.schemas.perturb import (
    AblateRequest,
    AmplifyRequest,
    PatchRequest,
    PatchResult,
    PerturbResult,
    TokenPrediction,
    ZeroOutRequest,
)

logger = logging.getLogger(__name__)


class PerturbationEngine:
    """Applies perturbations to model components and compares results.

    All perturbations are stateless — each request sets up fresh hooks
    via run_with_hooks() and never modifies model weights.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    @staticmethod
    def _resolve_hook(component: str) -> str:
        """Map component ID to TransformerLens hook name."""
        if component == "embed":
            return "hook_embed"
        if component.endswith(".attn"):
            return component.replace(".attn", ".hook_attn_out")
        if component.endswith(".mlp"):
            return component.replace(".mlp", ".hook_mlp_out")
        raise ValueError(f"Unknown component: {component}")

    def _get_predictions(
        self,
        logits_at_pos: torch.Tensor,
        k: int = 5,
    ) -> tuple[TokenPrediction, list[TokenPrediction]]:
        """Extract top-k token predictions from logits at a single position."""
        probs = torch.softmax(logits_at_pos, dim=-1)
        topk = torch.topk(probs, k)
        tokenizer = self._mm.get_model().tokenizer

        results = []
        for i in range(k):
            idx = topk.indices[i].item()
            results.append(
                TokenPrediction(
                    token=tokenizer.decode([idx]),
                    logit=round(logits_at_pos[idx].item(), 4),
                    prob=round(topk.values[i].item(), 4),
                )
            )

        return results[0], results

    def _compare(
        self,
        orig_logits: torch.Tensor,
        pert_logits: torch.Tensor,
        target_idx: int,
        component: str,
        pert_type: str,
        elapsed_ms: float,
    ) -> PerturbResult:
        """Compare original vs perturbed logits and build result."""
        orig_at = orig_logits[0, target_idx]
        pert_at = pert_logits[0, target_idx]

        orig_pred, orig_topk = self._get_predictions(orig_at)
        pert_pred, pert_topk = self._get_predictions(pert_at)

        # Logit diff for top-1 original token
        top_token_idx = torch.argmax(torch.softmax(orig_at, dim=-1)).item()
        logit_diff = pert_at[top_token_idx].item() - orig_at[top_token_idx].item()

        # KL divergence: KL(original || perturbed)
        orig_probs = torch.softmax(orig_at, dim=-1)
        pert_probs = torch.softmax(pert_at, dim=-1)
        kl = torch.sum(
            orig_probs * (torch.log(orig_probs + 1e-10) - torch.log(pert_probs + 1e-10))
        ).item()

        return PerturbResult(
            model_id=self._mm.model_id,
            component=component,
            perturbation_type=pert_type,
            original=orig_pred,
            perturbed=pert_pred,
            top_k_original=orig_topk,
            top_k_perturbed=pert_topk,
            logit_diff=round(logit_diff, 4),
            kl_divergence=round(kl, 4),
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
        )

    def zero_out(self, req: ZeroOutRequest) -> PerturbResult:
        """Zero-out a component's output and compare predictions."""
        start = time.time()
        model = self._mm.get_model()
        tokens = model.to_tokens(req.prompt)
        target_idx = tokens.shape[1] - 1
        hook_name = self._resolve_hook(req.component)

        with torch.no_grad():
            original_logits = model(tokens)

        def zero_hook(value, hook):
            return torch.zeros_like(value)

        with torch.no_grad():
            perturbed_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, zero_hook)],
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info("Perturbation zero_out on %s: %.1fms", req.component, elapsed_ms)
        return self._compare(
            original_logits,
            perturbed_logits,
            target_idx,
            req.component,
            "zero_out",
            elapsed_ms,
        )

    def amplify(self, req: AmplifyRequest) -> PerturbResult:
        """Amplify a component's output by a factor and compare predictions."""
        start = time.time()
        model = self._mm.get_model()
        tokens = model.to_tokens(req.prompt)
        target_idx = tokens.shape[1] - 1
        hook_name = self._resolve_hook(req.component)
        factor = req.factor

        with torch.no_grad():
            original_logits = model(tokens)

        def amplify_hook(value, hook):
            return value * factor

        with torch.no_grad():
            perturbed_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, amplify_hook)],
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info("Perturbation amplify(%.1fx) on %s: %.1fms", factor, req.component, elapsed_ms)
        return self._compare(
            original_logits,
            perturbed_logits,
            target_idx,
            req.component,
            "amplify",
            elapsed_ms,
        )

    def ablate(self, req: AblateRequest) -> PerturbResult:
        """Mean ablation: replace a component's output with its mean activation."""
        start = time.time()
        model = self._mm.get_model()
        tokens = model.to_tokens(req.prompt)
        target_idx = tokens.shape[1] - 1
        hook_name = self._resolve_hook(req.component)

        with torch.no_grad():
            original_logits, cache = model.run_with_cache(tokens)

        # Compute mean activation across token positions
        mean_activation = cache[hook_name].mean(dim=1, keepdim=True)  # [1, 1, ...]

        def ablate_hook(value, hook):
            return mean_activation.expand_as(value)

        with torch.no_grad():
            perturbed_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, ablate_hook)],
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info("Perturbation ablate on %s: %.1fms", req.component, elapsed_ms)
        return self._compare(
            original_logits,
            perturbed_logits,
            target_idx,
            req.component,
            "ablate",
            elapsed_ms,
        )

    def activation_patch(self, req: PatchRequest) -> PatchResult:
        """Activation patching (causal tracing): patch clean activation into corrupt run."""
        start = time.time()
        model = self._mm.get_model()
        hook_name = self._resolve_hook(req.component)

        clean_tokens = model.to_tokens(req.clean_prompt)
        corrupt_tokens = model.to_tokens(req.corrupt_prompt)
        target_idx = (
            req.target_token_idx if req.target_token_idx >= 0 else clean_tokens.shape[1] - 1
        )

        with torch.no_grad():
            # Clean run
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            clean_activation = clean_cache[hook_name].clone()

            # Corrupt baseline
            corrupt_logits = model(corrupt_tokens)

        # Patch: inject clean activation into corrupt run
        def patch_hook(value, hook):
            # Only patch if shapes match (same sequence length)
            if value.shape == clean_activation.shape:
                return clean_activation
            # If sequence lengths differ, patch up to the shorter length
            min_seq = min(value.shape[1], clean_activation.shape[1])
            patched = value.clone()
            patched[:, :min_seq] = clean_activation[:, :min_seq]
            return patched

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, patch_hook)],
            )

        # Get predictions
        clean_pred, _ = self._get_predictions(clean_logits[0, target_idx])
        corrupt_pred, _ = self._get_predictions(corrupt_logits[0, target_idx])
        patched_pred, _ = self._get_predictions(patched_logits[0, target_idx])

        # Recovery score: how much patching restored clean behavior
        # recovery = (patched_logit - corrupt_logit) / (clean_logit - corrupt_logit)
        clean_top_idx = torch.argmax(torch.softmax(clean_logits[0, target_idx], dim=-1)).item()
        clean_l = clean_logits[0, target_idx, clean_top_idx].item()
        corrupt_l = corrupt_logits[0, target_idx, clean_top_idx].item()
        patched_l = patched_logits[0, target_idx, clean_top_idx].item()

        denom = clean_l - corrupt_l
        recovery = (patched_l - corrupt_l) / denom if abs(denom) > 1e-6 else 0.0
        recovery = max(0.0, min(1.0, recovery))

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Activation patch on %s: recovery=%.3f, %.1fms",
            req.component,
            recovery,
            elapsed_ms,
        )

        return PatchResult(
            model_id=self._mm.model_id,
            component=req.component,
            clean_prompt=req.clean_prompt,
            corrupt_prompt=req.corrupt_prompt,
            clean_prediction=clean_pred,
            corrupt_prediction=corrupt_pred,
            patched_prediction=patched_pred,
            recovery_score=round(recovery, 4),
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
        )

    def causal_trace(self, req: CausalTraceRequest) -> CausalTraceResult:
        """Full causal tracing sweep: patch each component and measure recovery.

        Runs clean and corrupt forward passes once, then iterates over all
        components (embed + blocks.N.attn + blocks.N.mlp) patching clean
        activations into the corrupt run to compute recovery scores.
        """
        start = time.time()
        model = self._mm.get_model()
        n_layers = model.cfg.n_layers

        clean_tokens = model.to_tokens(req.clean_prompt)
        corrupt_tokens = model.to_tokens(req.corrupt_prompt)
        target_idx = (
            req.target_token_idx if req.target_token_idx >= 0 else clean_tokens.shape[1] - 1
        )

        # Build component list
        components = ["embed"]
        for layer_i in range(n_layers):
            components.append(f"blocks.{layer_i}.attn")
            components.append(f"blocks.{layer_i}.mlp")

        with torch.no_grad():
            # Single clean forward pass with full cache
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            # Single corrupt baseline
            corrupt_logits = model(corrupt_tokens)

        # Reference values for recovery computation
        clean_top_idx = torch.argmax(torch.softmax(clean_logits[0, target_idx], dim=-1)).item()
        clean_l = clean_logits[0, target_idx, clean_top_idx].item()
        corrupt_l = corrupt_logits[0, target_idx, clean_top_idx].item()
        denom = clean_l - corrupt_l

        # Get predictions for summary
        tokenizer = model.tokenizer
        clean_prediction = tokenizer.decode([clean_top_idx])
        corrupt_top_idx = torch.argmax(torch.softmax(corrupt_logits[0, target_idx], dim=-1)).item()
        corrupt_prediction = tokenizer.decode([corrupt_top_idx])

        cells: list[CausalTraceCell] = []

        for comp in components:
            hook_name = self._resolve_hook(comp)
            clean_activation = clean_cache[hook_name].clone()

            def make_patch_hook(clean_act: torch.Tensor):
                def hook_fn(value, hook):
                    if value.shape == clean_act.shape:
                        return clean_act
                    min_seq = min(value.shape[1], clean_act.shape[1])
                    patched = value.clone()
                    patched[:, :min_seq] = clean_act[:, :min_seq]
                    return patched

                return hook_fn

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupt_tokens,
                    fwd_hooks=[(hook_name, make_patch_hook(clean_activation))],
                )

            patched_l = patched_logits[0, target_idx, clean_top_idx].item()
            recovery = (patched_l - corrupt_l) / denom if abs(denom) > 1e-6 else 0.0
            recovery = max(0.0, min(1.0, recovery))

            # Parse component type and layer index
            if comp == "embed":
                comp_type = "embed"
                layer_idx = -1
            elif comp.endswith(".attn"):
                comp_type = "attn"
                layer_idx = int(comp.split(".")[1])
            else:
                comp_type = "mlp"
                layer_idx = int(comp.split(".")[1])

            cells.append(
                CausalTraceCell(
                    component=comp,
                    layer_idx=layer_idx,
                    component_type=comp_type,
                    recovery_score=round(recovery, 4),
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            "Causal trace: %d components, %.1fms",
            len(cells),
            elapsed_ms,
        )

        return CausalTraceResult(
            model_id=self._mm.model_id,
            clean_prompt=req.clean_prompt,
            corrupt_prompt=req.corrupt_prompt,
            target_token_idx=target_idx,
            clean_prediction=clean_prediction,
            corrupt_prediction=corrupt_prediction,
            cells=cells,
            n_layers=n_layers,
            metadata={"compute_time_ms": round(elapsed_ms, 1)},
        )
