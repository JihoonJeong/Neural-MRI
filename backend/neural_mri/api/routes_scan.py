from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from neural_mri.core.analysis_engine import AnalysisEngine
from neural_mri.core.model_manager import ModelManager
from neural_mri.core.scan_cache import ScanCache
from neural_mri.schemas.scan import (
    ActivationData,
    ActivationScanRequest,
    AnomalyData,
    AnomalyScanRequest,
    CircuitData,
    CircuitScanRequest,
    StructuralData,
    StructuralScanRequest,
    WeightData,
    WeightScanRequest,
)

router = APIRouter()


def get_model_manager() -> ModelManager:
    from neural_mri.main import model_manager

    return model_manager


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


@router.post("/structural", response_model=StructuralData)
async def scan_structural(
    _req: StructuralScanRequest = StructuralScanRequest(),
    mm: ModelManager = Depends(get_model_manager),
    engine: AnalysisEngine = Depends(get_analysis_engine),
    cache: ScanCache = Depends(get_scan_cache),
) -> StructuralData:
    _require_model(mm)
    cached = cache.get(mm.model_id, "structural", "")
    if cached is not None:
        return StructuralData(**cached)
    result = engine.scan_structural()
    cache.put(mm.model_id, "structural", "", result.model_dump())
    return result


@router.post("/weights", response_model=WeightData)
async def scan_weights(
    req: WeightScanRequest = WeightScanRequest(),
    mm: ModelManager = Depends(get_model_manager),
    engine: AnalysisEngine = Depends(get_analysis_engine),
    cache: ScanCache = Depends(get_scan_cache),
) -> WeightData:
    _require_model(mm)
    cached = cache.get(mm.model_id, "weights", "")
    if cached is not None:
        return WeightData(**cached)
    result = engine.scan_weights(req.layers)
    cache.put(mm.model_id, "weights", "", result.model_dump())
    return result


@router.post("/activation", response_model=ActivationData)
async def scan_activation(
    req: ActivationScanRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: AnalysisEngine = Depends(get_analysis_engine),
    cache: ScanCache = Depends(get_scan_cache),
) -> ActivationData:
    _require_model(mm)
    cached = cache.get(mm.model_id, "activation", req.prompt)
    if cached is not None:
        return ActivationData(**cached)
    result = await asyncio.to_thread(engine.scan_activation, req)
    cache.put(mm.model_id, "activation", req.prompt, result.model_dump())
    return result


@router.post("/circuits", response_model=CircuitData)
async def scan_circuits(
    req: CircuitScanRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: AnalysisEngine = Depends(get_analysis_engine),
    cache: ScanCache = Depends(get_scan_cache),
) -> CircuitData:
    _require_model(mm)
    cached = cache.get(mm.model_id, "circuits", req.prompt)
    if cached is not None:
        return CircuitData(**cached)
    result = await asyncio.to_thread(engine.scan_circuits, req)
    cache.put(mm.model_id, "circuits", req.prompt, result.model_dump())
    return result


@router.post("/anomaly", response_model=AnomalyData)
async def scan_anomaly(
    req: AnomalyScanRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: AnalysisEngine = Depends(get_analysis_engine),
    cache: ScanCache = Depends(get_scan_cache),
) -> AnomalyData:
    _require_model(mm)
    cached = cache.get(mm.model_id, "anomaly", req.prompt)
    if cached is not None:
        return AnomalyData(**cached)
    result = await asyncio.to_thread(engine.scan_anomaly, req)
    cache.put(mm.model_id, "anomaly", req.prompt, result.model_dump())
    return result
