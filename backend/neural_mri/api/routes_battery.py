from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from neural_mri.core.battery_engine import BatteryEngine
from neural_mri.core.model_manager import ModelManager
from neural_mri.core.sae_manager import SAEManager
from neural_mri.core.test_registry import get_all_tests
from neural_mri.schemas.battery import BatteryResult, BatteryRunRequest, TestCase

router = APIRouter()


def get_model_manager() -> ModelManager:
    from neural_mri.main import model_manager

    return model_manager


def get_sae_manager() -> SAEManager:
    from neural_mri.main import sae_manager

    return sae_manager


def get_battery_engine(
    mm: ModelManager = Depends(get_model_manager),
    sae_mgr: SAEManager = Depends(get_sae_manager),
) -> BatteryEngine:
    return BatteryEngine(mm, sae_manager=sae_mgr)


def _require_model(mm: ModelManager) -> None:
    if not mm.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")


@router.post("/run", response_model=BatteryResult)
async def run_battery(
    req: BatteryRunRequest = BatteryRunRequest(),
    mm: ModelManager = Depends(get_model_manager),
    engine: BatteryEngine = Depends(get_battery_engine),
) -> BatteryResult:
    _require_model(mm)
    result = await asyncio.to_thread(
        engine.run_battery, req.categories, req.locale, req.include_sae, req.sae_layer,
    )
    return result


@router.get("/tests", response_model=list[TestCase])
async def list_tests() -> list[TestCase]:
    return get_all_tests()
