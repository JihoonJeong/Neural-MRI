from __future__ import annotations

import gc
import logging

import torch

from neural_mri.core.cuda_guard import OOMError, free_vram, get_vram_info
from neural_mri.core.sae_providers import SAEProvider, load_sae_provider
from neural_mri.core.sae_registry import get_sae_info

logger = logging.getLogger(__name__)


class SAEManager:
    """Manages SAE loading/caching — one SAE provider at a time (model+layer)."""

    def __init__(self) -> None:
        self._provider: SAEProvider | None = None
        self._model_id: str | None = None
        self._layer_idx: int | None = None

    @property
    def provider(self) -> SAEProvider | None:
        return self._provider

    def get_provider(self, model_id: str, layer_idx: int, device: str) -> SAEProvider:
        """Return cached provider or load a new one for the given model+layer."""
        if (
            self._provider is not None
            and self._model_id == model_id
            and self._layer_idx == layer_idx
        ):
            return self._provider

        self.unload()

        info = get_sae_info(model_id)
        if info is None:
            raise ValueError(f"No SAE available for model: {model_id}")

        if layer_idx not in info["layers"]:
            raise ValueError(f"Layer {layer_idx} not available for SAE. Valid: {info['layers']}")

        # VRAM guard: check if enough free memory before loading SAE
        sae_vram_mb = info.get("vram_mb", 0)
        if sae_vram_mb > 0 and device != "cpu":
            vram = get_vram_info()
            if vram.get("available") and vram["free_mb"] < sae_vram_mb:
                raise OOMError(
                    f"Not enough VRAM to load SAE. "
                    f"Need ~{sae_vram_mb}MB, only {vram['free_mb']:.0f}MB free "
                    f"({vram['allocated_mb']:.0f}MB used / {vram['total_mb']:.0f}MB total). "
                    f"Try unloading the model first or using a smaller SAE."
                )

        provider_type = info.get("provider", "saelens")
        try:
            provider = load_sae_provider(provider_type, model_id, layer_idx, device, info)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            free_vram()
            raise OOMError(
                f"Failed to load SAE for {model_id} layer {layer_idx}: {exc}. "
                f"The SAE may be too large for available VRAM."
            ) from exc

        self._provider = provider
        self._model_id = model_id
        self._layer_idx = layer_idx
        logger.info(
            "SAE loaded: %s layer %d via %s (d_sae=%d)",
            model_id,
            layer_idx,
            provider.provider_name,
            provider.d_sae,
        )
        return provider

    # Backward-compat alias for existing code paths
    def get_sae(self, model_id: str, layer_idx: int, device: str) -> SAEProvider:
        """Alias for get_provider — backward compatibility."""
        return self.get_provider(model_id, layer_idx, device)

    def unload(self) -> None:
        """Free SAE from memory."""
        if self._provider is not None:
            mid, lid = self._model_id, self._layer_idx
            del self._provider
            self._provider = None
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
