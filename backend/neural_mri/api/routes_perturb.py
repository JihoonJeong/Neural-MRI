from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from neural_mri.core.model_manager import ModelManager
from neural_mri.core.perturbation_engine import PerturbationEngine
from neural_mri.schemas.causal_trace import (
    CausalTraceRequest,
    CausalTraceResult,
)
from neural_mri.schemas.perturb import (
    AblateRequest,
    AmplifyRequest,
    PatchRequest,
    PatchResult,
    PerturbResult,
    ZeroOutRequest,
)

router = APIRouter()


def get_model_manager() -> ModelManager:
    from neural_mri.main import model_manager

    return model_manager


def get_perturbation_engine(
    mm: ModelManager = Depends(get_model_manager),
) -> PerturbationEngine:
    return PerturbationEngine(mm)


def _require_model(mm: ModelManager) -> None:
    if not mm.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")


@router.post("/zero", response_model=PerturbResult)
async def perturb_zero(
    req: ZeroOutRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: PerturbationEngine = Depends(get_perturbation_engine),
) -> PerturbResult:
    _require_model(mm)
    return await asyncio.to_thread(engine.zero_out, req)


@router.post("/amplify", response_model=PerturbResult)
async def perturb_amplify(
    req: AmplifyRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: PerturbationEngine = Depends(get_perturbation_engine),
) -> PerturbResult:
    _require_model(mm)
    return await asyncio.to_thread(engine.amplify, req)


@router.post("/ablate", response_model=PerturbResult)
async def perturb_ablate(
    req: AblateRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: PerturbationEngine = Depends(get_perturbation_engine),
) -> PerturbResult:
    _require_model(mm)
    return await asyncio.to_thread(engine.ablate, req)


@router.post("/patch", response_model=PatchResult)
async def perturb_patch(
    req: PatchRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: PerturbationEngine = Depends(get_perturbation_engine),
) -> PatchResult:
    _require_model(mm)
    return await asyncio.to_thread(engine.activation_patch, req)


@router.post("/causal-trace", response_model=CausalTraceResult)
async def causal_trace(
    req: CausalTraceRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: PerturbationEngine = Depends(get_perturbation_engine),
) -> CausalTraceResult:
    _require_model(mm)
    return await asyncio.to_thread(engine.causal_trace, req)


@router.post("/reset")
async def perturb_reset():
    """No-op since all perturbations are stateless (hook-based per request)."""
    return {"status": "reset"}
