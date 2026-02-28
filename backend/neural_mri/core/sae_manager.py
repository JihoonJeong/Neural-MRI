from __future__ import annotations

import gc
import logging

import torch
from sae_lens import SAE

from neural_mri.core.sae_registry import get_sae_info

logger = logging.getLogger(__name__)


class SAEManager:
    """Manages SAE loading/caching — one SAE at a time (model+layer)."""

    def __init__(self) -> None:
        self._sae: SAE | None = None
        self._model_id: str | None = None
        self._layer_idx: int | None = None

    def get_sae(self, model_id: str, layer_idx: int, device: str) -> SAE:
        """Return cached SAE or load a new one for the given model+layer."""
        # Cache hit
        if (
            self._sae is not None
            and self._model_id == model_id
            and self._layer_idx == layer_idx
        ):
            return self._sae

        # Different model or layer — unload first
        self.unload()

        info = get_sae_info(model_id)
        if info is None:
            raise ValueError(f"No SAE available for model: {model_id}")

        if layer_idx not in info["layers"]:
            raise ValueError(
                f"Layer {layer_idx} not available for SAE. Valid: {info['layers']}"
            )

        release = info["release"]
        sae_id = info["sae_id_template"].format(layer=layer_idx)

        logger.info("Loading SAE: release=%s, sae_id=%s, device=%s", release, sae_id, device)
        sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )

        self._sae = sae
        self._model_id = model_id
        self._layer_idx = layer_idx
        logger.info("SAE loaded: %s layer %d (d_sae=%d)", model_id, layer_idx, sae.cfg.d_sae)
        return sae

    def unload(self) -> None:
        """Free SAE from memory."""
        if self._sae is not None:
            mid, lid = self._model_id, self._layer_idx
            del self._sae
            self._sae = None
            self._model_id = None
            self._layer_idx = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            logger.info("SAE unloaded: %s layer %s", mid, lid)

    def unload_if_model(self, model_id: str) -> None:
        """Unload SAE if it belongs to the given model (for model switching)."""
        if self._model_id == model_id:
            self.unload()
