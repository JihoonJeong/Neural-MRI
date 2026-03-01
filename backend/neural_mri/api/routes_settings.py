from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from neural_mri.core.scan_cache import ScanCache

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory token store — NEVER persisted to disk
_runtime_token: str | None = None


class TokenUpdateRequest(BaseModel):
    token: str


class TokenStatusResponse(BaseModel):
    is_set: bool
    is_valid: bool | None  # None = not yet validated
    source: str  # "runtime" | "env" | "none"


class CacheStatusResponse(BaseModel):
    entry_count: int
    max_entries: int


def get_scan_cache() -> ScanCache:
    from neural_mri.main import scan_cache

    return scan_cache


def get_settings():
    from neural_mri.main import settings

    return settings


def _validate_token(token: str) -> bool:
    """Validate HF token by calling whoami endpoint."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.whoami()
        return True
    except Exception:
        return False


def _get_active_token() -> tuple[str | None, str]:
    """Return (token, source) based on runtime or env."""
    global _runtime_token  # noqa: PLW0602
    if _runtime_token:
        return _runtime_token, "runtime"
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token, "env"
    return None, "none"


@router.post("/token")
async def update_token(req: TokenUpdateRequest) -> TokenStatusResponse:
    """Update the HF token at runtime. Stored in-memory only."""
    global _runtime_token
    _runtime_token = req.token
    os.environ["HF_TOKEN"] = req.token
    logger.info("HuggingFace token updated (runtime).")
    is_valid = _validate_token(req.token)
    return TokenStatusResponse(is_set=True, is_valid=is_valid, source="runtime")


@router.delete("/token")
async def clear_token() -> TokenStatusResponse:
    """Clear the runtime token."""
    global _runtime_token
    _runtime_token = None
    os.environ.pop("HF_TOKEN", None)
    logger.info("HuggingFace token cleared.")
    return TokenStatusResponse(is_set=False, is_valid=None, source="none")


@router.get("/token/status")
async def get_token_status() -> TokenStatusResponse:
    """Check current token status without revealing the token value."""
    token, source = _get_active_token()
    if not token:
        return TokenStatusResponse(is_set=False, is_valid=None, source="none")
    is_valid = _validate_token(token)
    return TokenStatusResponse(is_set=True, is_valid=is_valid, source=source)


@router.get("/cache")
async def get_cache_status(
    cache: ScanCache = Depends(get_scan_cache),
    settings=Depends(get_settings),
) -> CacheStatusResponse:
    return CacheStatusResponse(
        entry_count=len(cache._store),
        max_entries=settings.max_cache_entries,
    )


@router.delete("/cache")
async def clear_cache(
    cache: ScanCache = Depends(get_scan_cache),
) -> dict:
    cache.clear()
    logger.info("Scan cache cleared via API.")
    return {"status": "cleared"}
