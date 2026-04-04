from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from neural_mri.core.emotion_engine import EmotionEngine
from neural_mri.core.model_manager import ModelManager
from neural_mri.schemas.emotion import (
    ExtractProbesRequest,
    ExtractProbesResponse,
    SteerRequest,
    SteerResponse,
)

router = APIRouter()


def get_model_manager() -> ModelManager:
    from neural_mri.main import model_manager

    return model_manager


def get_emotion_engine(
    mm: ModelManager = Depends(get_model_manager),
) -> EmotionEngine:
    from neural_mri.main import emotion_engine

    return emotion_engine


def _require_model(mm: ModelManager) -> None:
    if not mm.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")


@router.post("/extract-probes", response_model=ExtractProbesResponse)
async def extract_probes(
    req: ExtractProbesRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: EmotionEngine = Depends(get_emotion_engine),
) -> ExtractProbesResponse:
    """Extract emotion vectors from the loaded model via comprehension texts."""
    _require_model(mm)
    return await asyncio.to_thread(engine.extract_probes, req)


@router.post("/steer", response_model=SteerResponse)
async def steer(
    req: SteerRequest,
    mm: ModelManager = Depends(get_model_manager),
    engine: EmotionEngine = Depends(get_emotion_engine),
) -> SteerResponse:
    """Run prompt with emotion vector steering and compare outputs."""
    _require_model(mm)
    return await asyncio.to_thread(engine.steer, req)


@router.get("/emotions")
async def list_emotions(
    mm: ModelManager = Depends(get_model_manager),
    engine: EmotionEngine = Depends(get_emotion_engine),
) -> dict:
    """List available emotions and probe status."""
    texts = engine._get_texts()
    model_id = mm.model_id if mm.is_loaded else None

    has_probes = False
    probe_layers: list[int] = []
    if model_id:
        model_probes = engine._probes.get(model_id, {})
        has_probes = len(model_probes) > 0
        probe_layers = sorted(model_probes.keys())

    return {
        "emotions": sorted(texts.keys()),
        "n_emotions": len(texts),
        "passages_per_emotion": {e: len(p) for e, p in texts.items()},
        "model_id": model_id,
        "has_probes": has_probes,
        "probe_layers": probe_layers,
    }
