from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from neural_mri.api.routes_battery import router as battery_router
from neural_mri.api.routes_model import router as model_router
from neural_mri.api.routes_perturb import router as perturb_router
from neural_mri.api.routes_report import router as report_router
from neural_mri.api.routes_scan import router as scan_router
from neural_mri.api.ws_stream import router as ws_router
from neural_mri.config import Settings
from neural_mri.core.model_manager import ModelManager
from neural_mri.core.scan_cache import ScanCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

settings = Settings()

# Propagate HF token to huggingface_hub so gated models can be downloaded
if settings.hf_token:
    import os

    os.environ.setdefault("HF_TOKEN", settings.hf_token)
    logger.info("HuggingFace token configured for gated model access.")

model_manager = ModelManager()
scan_cache = ScanCache(max_entries=settings.max_cache_entries)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load default model
    if settings.default_model:
        logger.info("Loading default model: %s", settings.default_model)
        model_manager.load_model(settings.default_model, device=settings.device)
    yield
    # Shutdown: free GPU memory
    model_manager.unload_model()
    logger.info("Neural MRI Scanner shut down.")


app = FastAPI(
    title="Neural MRI Scanner",
    description="Model Resonance Imaging for AI Interpretability",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router, prefix="/api/model", tags=["model"])
app.include_router(scan_router, prefix="/api/scan", tags=["scan"])
app.include_router(perturb_router, prefix="/api/perturb", tags=["perturb"])
app.include_router(report_router, prefix="/api/report", tags=["report"])
app.include_router(battery_router, prefix="/api/battery", tags=["battery"])
app.include_router(ws_router, tags=["websocket"])


@app.get("/")
async def root():
    return {
        "name": "Neural MRI Scanner",
        "version": "0.1.0",
        "description": "Model Resonance Imaging for AI Interpretability",
    }
