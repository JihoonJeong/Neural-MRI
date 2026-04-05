"""CUDA error recovery and VRAM management utilities."""

from __future__ import annotations

import functools
import gc
import logging
from collections.abc import Callable
from typing import TypeVar

import torch

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def free_vram() -> None:
    """Aggressively free GPU/MPS memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def get_vram_info() -> dict:
    """Return current VRAM usage in MB (CUDA only)."""
    if not torch.cuda.is_available():
        return {"available": False}
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    total = torch.cuda.get_device_properties(0).total_mem / 1e6
    return {
        "available": True,
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "total_mb": round(total, 1),
        "free_mb": round(total - allocated, 1),
    }


class CUDAError(RuntimeError):
    """CUDA-specific error with recovery info."""

    def __init__(self, message: str, vram_info: dict | None = None) -> None:
        self.vram_info = vram_info or get_vram_info()
        super().__init__(message)


class OOMError(CUDAError):
    """Out of memory error with VRAM details."""


def classify_cuda_error(exc: Exception) -> RuntimeError:
    """Convert raw CUDA/torch errors to specific error types with details."""
    msg = str(exc).lower()
    vram = get_vram_info()

    if "out of memory" in msg or "oom" in msg or "allocation" in msg:
        vram_str = ""
        if vram.get("available"):
            vram_str = (
                f" (VRAM: {vram['allocated_mb']:.0f}MB used / "
                f"{vram['total_mb']:.0f}MB total, "
                f"{vram['free_mb']:.0f}MB free)"
            )
        return OOMError(
            f"GPU out of memory{vram_str}. "
            f"Try: reduce max_new_tokens, unload SAE, or use a smaller model. "
            f"Original: {exc}"
        )

    if "invalid resource handle" in msg or "cuda error" in msg:
        return CUDAError(
            f"CUDA state corrupted. The GPU context may need a reset. "
            f"Try reloading the model. Original: {exc}"
        )

    return RuntimeError(f"GPU error: {exc}")


def cuda_safe(func: F) -> F:
    """Decorator: catch CUDA errors, free VRAM, and re-raise with details."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            msg = str(exc).lower()
            if any(
                k in msg
                for k in ["cuda", "out of memory", "oom", "mps", "invalid resource", "allocation"]
            ):
                logger.error("CUDA/GPU error in %s: %s — freeing VRAM", func.__name__, exc)
                free_vram()
                raise classify_cuda_error(exc) from exc
            raise

    return wrapper  # type: ignore[return-value]
