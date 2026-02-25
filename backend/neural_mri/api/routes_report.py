from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from neural_mri.core.analysis_engine import AnalysisEngine
from neural_mri.core.model_manager import ModelManager
from neural_mri.core.report_engine import ReportEngine
from neural_mri.schemas.report import DiagnosticReport, ReportRequest

router = APIRouter()


def get_model_manager() -> ModelManager:
    from neural_mri.main import model_manager

    return model_manager


def get_report_engine(
    mm: ModelManager = Depends(get_model_manager),
) -> ReportEngine:
    return ReportEngine(mm, AnalysisEngine(mm))


def _require_model(mm: ModelManager) -> None:
    if not mm.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")


@router.post("/generate", response_model=DiagnosticReport)
async def generate_report(
    req: ReportRequest = ReportRequest(),
    mm: ModelManager = Depends(get_model_manager),
    engine: ReportEngine = Depends(get_report_engine),
) -> DiagnosticReport:
    _require_model(mm)
    result = await asyncio.to_thread(engine.generate, req)
    return result
