"""SAE provider abstraction — adapter pattern for SAELens / EleutherAI backends.

Each provider wraps a different SAE library behind a unified interface:
    encode(activations) → (top_acts, top_indices, full_acts)
    decode(top_acts, top_indices) → reconstruction
    d_sae, hook_name — unified metadata
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class EncodeResult:
    """Unified encode output across providers."""

    top_acts: torch.Tensor  # [batch, seq_len, top_k] — top-k activations
    top_indices: torch.Tensor  # [batch, seq_len, top_k] — feature indices
    full_acts: torch.Tensor | None  # [batch, seq_len, d_sae] — dense (SAELens only)


class SAEProvider(ABC):
    """Abstract interface that all SAE backends must implement."""

    @abstractmethod
    def encode(self, activations: torch.Tensor) -> EncodeResult:
        """Encode residual stream activations into sparse features.

        Args:
            activations: [batch, seq_len, d_model]

        Returns:
            EncodeResult with top_acts, top_indices, and optionally full dense acts.
        """

    @abstractmethod
    def decode_from_top(self, top_acts: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct from top-k sparse representation.

        Args:
            top_acts: [batch, seq_len, top_k]
            top_indices: [batch, seq_len, top_k]

        Returns:
            Reconstruction tensor [batch, seq_len, d_model].
        """

    @property
    @abstractmethod
    def d_sae(self) -> int:
        """Number of SAE latent features."""

    @property
    @abstractmethod
    def hook_name(self) -> str:
        """TransformerLens hook point name for activation extraction."""

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


# --------------------------------------------------------------------------- #
# SAELens Provider
# --------------------------------------------------------------------------- #


class SAELensProvider(SAEProvider):
    """Wraps sae_lens.SAE with the unified provider interface."""

    def __init__(self, sae, hook_name_override: str | None = None) -> None:
        self._sae = sae
        self._hook = hook_name_override or self._resolve_hook()

    def _resolve_hook(self) -> str:
        if self._sae.cfg.metadata:
            h = self._sae.cfg.metadata.get("hook_name")
            if h:
                return h
        return ""

    def encode(self, activations: torch.Tensor) -> EncodeResult:
        acts_float = activations.float()
        # SAELens encode returns dense [batch, seq, d_sae]
        full = self._sae.encode(acts_float)
        # Extract top-k per token
        k = min(64, full.shape[-1])
        top_vals, top_idxs = torch.topk(full, k, dim=-1)
        return EncodeResult(top_acts=top_vals, top_indices=top_idxs, full_acts=full)

    def decode_from_top(self, top_acts: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        # Scatter back to dense, then decode
        d = self._sae.cfg.d_sae
        shape = top_acts.shape[:-1] + (d,)
        dense = torch.zeros(shape, device=top_acts.device, dtype=top_acts.dtype)
        dense.scatter_(-1, top_indices, top_acts)
        return self._sae.decode(dense)

    @property
    def d_sae(self) -> int:
        return self._sae.cfg.d_sae

    @property
    def hook_name(self) -> str:
        return self._hook


# --------------------------------------------------------------------------- #
# EleutherAI (sparsify) Provider
# --------------------------------------------------------------------------- #


class EleutherAIProvider(SAEProvider):
    """Wraps sparsify.Sae with the unified provider interface."""

    def __init__(self, sae, hook_name: str) -> None:
        self._sae = sae
        self._hook = hook_name

    def encode(self, activations: torch.Tensor) -> EncodeResult:
        acts_float = activations.float()
        orig_shape = acts_float.shape  # [batch, seq_len, d_model]
        flat = acts_float.flatten(0, -2)  # [batch*seq_len, d_model]
        out = self._sae.encode(flat)  # EncoderOutput(top_acts, top_indices, pre_acts)
        # Reshape back to [batch, seq_len, top_k]
        top_k = out.top_acts.shape[-1]
        top_acts = out.top_acts.view(*orig_shape[:-1], top_k)
        top_indices = out.top_indices.view(*orig_shape[:-1], top_k)
        return EncodeResult(top_acts=top_acts, top_indices=top_indices, full_acts=None)

    def decode_from_top(self, top_acts: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        orig_shape = top_acts.shape[:-1]  # [batch, seq_len]
        flat_acts = top_acts.flatten(0, -2)
        flat_idxs = top_indices.flatten(0, -2)
        recon_flat = self._sae.decode(flat_acts, flat_idxs)  # [N, d_model]
        return recon_flat.view(*orig_shape, -1)

    @property
    def d_sae(self) -> int:
        return self._sae.num_latents

    @property
    def hook_name(self) -> str:
        return self._hook


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def load_sae_provider(
    provider_type: str,
    model_id: str,
    layer_idx: int,
    device: str,
    registry_entry: dict,
) -> SAEProvider:
    """Factory: load an SAE and wrap it in the appropriate provider.

    Args:
        provider_type: "saelens" or "eleutherai"
        model_id: HuggingFace model ID
        layer_idx: target layer index
        device: torch device string
        registry_entry: SAE_REGISTRY entry for this model
    """
    if provider_type == "saelens":
        from sae_lens import SAE

        release = registry_entry["release"]
        sae_id = registry_entry["sae_id_template"].format(layer=layer_idx)
        hook_fallback = sae_id

        logger.info("Loading SAELens: release=%s, sae_id=%s", release, sae_id)
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        return SAELensProvider(sae, hook_name_override=hook_fallback)

    elif provider_type == "eleutherai":
        from sparsify import Sae

        hub_id = registry_entry["hub_id"]
        hookpoint = registry_entry["hookpoint_template"].format(layer=layer_idx)

        logger.info("Loading EleutherAI SAE: hub=%s, hookpoint=%s", hub_id, hookpoint)
        sae = Sae.load_from_hub(hub_id, hookpoint=hookpoint, device=device)

        # Map EleutherAI hookpoint → TransformerLens hook name
        tl_hook = registry_entry.get("tl_hook_template", "blocks.{layer}.hook_resid_pre")
        tl_hook_name = tl_hook.format(layer=layer_idx)

        return EleutherAIProvider(sae, hook_name=tl_hook_name)

    else:
        raise ValueError(f"Unknown SAE provider: {provider_type}")
