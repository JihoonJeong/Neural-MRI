from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from neural_mri.core.analysis_engine import AnalysisEngine
from neural_mri.core.model_manager import ModelManager
from neural_mri.core.sae_manager import SAEManager
from neural_mri.core.sae_registry import get_sae_info, list_sae_support
from neural_mri.core.scan_cache import ScanCache
from neural_mri.schemas.scan import SAEData, SAEScanRequest

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


def get_analysis_engine(
    mm: ModelManager = Depends(get_model_manager),
) -> AnalysisEngine:
    return AnalysisEngine(mm)


def _require_model(mm: ModelManager) -> None:
    if not mm.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")


@router.get("/info")
async def sae_info(
    mm: ModelManager = Depends(get_model_manager),
) -> dict:
    """Return SAE availability info for the currently loaded model."""
    if not mm.is_loaded:
        return {"available": False, "model_id": None, "layers": [], "d_sae": 0}

    info = get_sae_info(mm.model_id)
    if info is None:
        return {"available": False, "model_id": mm.model_id, "layers": [], "d_sae": 0}

    return {
        "available": True,
        "model_id": mm.model_id,
        "release": info["release"],
        "layers": info["layers"],
        "d_sae": info["d_sae"],
        "has_neuronpedia": info.get("neuronpedia_url_template") is not None,
    }


@router.get("/support")
async def sae_support() -> dict[str, bool]:
    """Return SAE support status for all registered models."""
    return list_sae_support()


@router.post("/scan", response_model=SAEData)
async def sae_scan(
    req: SAEScanRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: AnalysisEngine = Depends(get_analysis_engine),
    sae_mgr: SAEManager = Depends(get_sae_manager),
    cache: ScanCache = Depends(get_scan_cache),
) -> SAEData:
    """Run SAE feature scan on a specific layer."""
    _require_model(mm)

    info = get_sae_info(mm.model_id)
    if info is None:
        raise HTTPException(
            status_code=400,
            detail=f"No SAE available for model: {mm.model_id}",
        )

    cache_key = f"{req.prompt}::layer{req.layer_idx}::k{req.top_k}"
    cached = cache.get(mm.model_id, "sae", cache_key)
    if cached is not None:
        return SAEData(**cached)

    result = await asyncio.to_thread(engine.scan_sae, req, sae_mgr)
    cache.put(mm.model_id, "sae", cache_key, result.model_dump())
    return result
