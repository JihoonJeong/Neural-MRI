from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from neural_mri.core.model_manager import ModelManager
from neural_mri.core.model_registry import list_models
from neural_mri.core.sae_manager import SAEManager
from neural_mri.core.scan_cache import ScanCache
from neural_mri.schemas.model import ModelInfo, ModelLoadRequest

router = APIRouter()


def get_model_manager() -> ModelManager:
    from neural_mri.main import model_manager

    return model_manager


def get_sae_manager() -> SAEManager:
    from neural_mri.main import sae_manager

    return sae_manager


def get_scan_cache() -> ScanCache:
    from neural_mri.main import scan_cache

    return scan_cache


@router.get("/list")
async def get_model_list(
    mm: ModelManager = Depends(get_model_manager),
) -> list[dict]:
    return list_models(mm.model_id)


@router.post("/load", response_model=ModelInfo)
async def load_model(
    req: ModelLoadRequest,
    mm: ModelManager = Depends(get_model_manager),
    cache: ScanCache = Depends(get_scan_cache),
    sae_mgr: SAEManager = Depends(get_sae_manager),
) -> ModelInfo:
    try:
        # Invalidate cache and SAE for the old model if switching
        if mm.model_id and mm.model_id != req.model_id:
            cache.invalidate_model(mm.model_id)
            sae_mgr.unload()
        result = mm.load_model(req.model_id, req.device)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info", response_model=ModelInfo)
async def get_model_info(
    mm: ModelManager = Depends(get_model_manager),
) -> ModelInfo:
    if not mm.is_loaded:
        raise HTTPException(status_code=404, detail="No model loaded")
    return mm.get_model_info()


@router.delete("/unload")
async def unload_model(
    mm: ModelManager = Depends(get_model_manager),
    cache: ScanCache = Depends(get_scan_cache),
) -> dict:
    if mm.model_id:
        cache.invalidate_model(mm.model_id)
    mm.unload_model()
    return {"status": "unloaded"}
