from __future__ import annotations

import gc
import logging

import torch
from transformer_lens import HookedTransformer

from neural_mri.schemas.model import LayerConfig, ModelInfo

logger = logging.getLogger(__name__)

# Models above this threshold (in params string) get float16 by default
_LARGE_MODEL_THRESHOLD = 1_000_000_000  # 1B


def _parse_param_str(s: str) -> int:
    """Parse '124M' or '1.4B' to an integer."""
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


class ModelManager:
    """Manages model loading, swapping, and memory lifecycle."""

    def __init__(self) -> None:
        self._model: HookedTransformer | None = None
        self._model_id: str | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _free_device_cache() -> None:
        """Free GPU/MPS memory caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def load_model(self, model_id: str, device: str = "auto") -> ModelInfo:
        """Load a model via TransformerLens HookedTransformer."""
        if self._model is not None:
            self.unload_model()

        resolved_device = self._resolve_device(device)
        logger.info("Loading model %s on %s...", model_id, resolved_device)

        # Determine dtype: use float16 for large models to save memory
        from neural_mri.core.model_registry import get_model_info as get_registry_info

        registry_meta = get_registry_info(model_id)
        use_fp16 = False
        if registry_meta:
            param_count = _parse_param_str(registry_meta["params"])
            if param_count >= _LARGE_MODEL_THRESHOLD:
                use_fp16 = True
                logger.info("Large model (%s) — using float16", registry_meta["params"])

        load_kwargs: dict = {"device": resolved_device}
        if use_fp16:
            load_kwargs["dtype"] = torch.float16

        if not registry_meta:
            logger.info(
                "Model %s not in registry, attempting dynamic load...",
                model_id,
            )

        try:
            self._model = HookedTransformer.from_pretrained(model_id, **load_kwargs)
        except ValueError as exc:
            # TransformerLens raises ValueError for unsupported architectures
            raise RuntimeError(
                f"TransformerLens does not support this model architecture: {exc}"
            ) from exc
        except OSError as exc:
            err_msg = str(exc).lower()
            if "401" in err_msg or "unauthorized" in err_msg or "token" in err_msg:
                raise RuntimeError(
                    f"Authentication required for {model_id}. Set an HF token in Settings."
                ) from exc
            if "404" in err_msg or "not found" in err_msg:
                raise RuntimeError(f"Model '{model_id}' not found on HuggingFace Hub.") from exc
            raise
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # GPU out of memory — fallback to CPU
            if resolved_device != "cpu":
                logger.warning(
                    "Failed to load on %s (%s), falling back to CPU...",
                    resolved_device,
                    exc,
                )
                self._free_device_cache()
                load_kwargs["device"] = "cpu"
                self._model = HookedTransformer.from_pretrained(model_id, **load_kwargs)
            else:
                raise

        self._model_id = model_id
        logger.info("Model %s loaded successfully.", model_id)
        return self.get_model_info()

    def unload_model(self) -> None:
        """Unload the current model and free memory."""
        if self._model is not None:
            model_id = self._model_id
            del self._model
            self._model = None
            self._model_id = None
            self._free_device_cache()
            logger.info("Model %s unloaded.", model_id)

    def get_model_info(self) -> ModelInfo:
        """Extract model architecture info from the loaded model config."""
        if self._model is None:
            raise RuntimeError("No model loaded")

        cfg = self._model.cfg
        layers = []
        for i in range(cfg.n_layers):
            layers.append(
                LayerConfig(
                    layer_index=i,
                    layer_type="transformer_block",
                    components=["attn", "mlp"],
                    num_heads=cfg.n_heads,
                    d_model=cfg.d_model,
                    d_head=cfg.d_head,
                    d_mlp=cfg.d_mlp,
                )
            )

        return ModelInfo(
            model_id=self._model_id,
            model_name=cfg.model_name,
            n_params=sum(p.numel() for p in self._model.parameters()),
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            d_vocab=cfg.d_vocab,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            d_mlp=cfg.d_mlp,
            max_seq_len=cfg.n_ctx,
            device=str(cfg.device),
            layers=layers,
            dtype=str(cfg.dtype),
        )

    def get_model(self) -> HookedTransformer:
        """Return the currently loaded model instance."""
        if self._model is None:
            raise RuntimeError("No model loaded")
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id
