from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from neural_mri.core.model_search import search_models

router = APIRouter()


@router.get("/search")
async def search_hub_models(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=50),
    tl_only: bool = Query(
        False,
        description="Filter to TransformerLens-compatible models only",
    ),
) -> list[dict]:
    """Search HuggingFace Hub for text-generation models."""
    try:
        results = await asyncio.to_thread(
            search_models,
            query=q,
            limit=limit,
            filter_tl_compat=tl_only,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hub search failed: {e}")
    return [
        {
            "model_id": r.model_id,
            "author": r.author,
            "downloads": r.downloads,
            "likes": r.likes,
            "pipeline_tag": r.pipeline_tag,
            "gated": r.gated,
            "tl_compat": r.tl_compat,
            "architectures": r.architectures,
        }
        for r in results
    ]
