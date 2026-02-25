from __future__ import annotations

import asyncio
import json
import logging
import time

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_engine():
    from neural_mri.core.analysis_engine import AnalysisEngine
    from neural_mri.main import model_manager

    return AnalysisEngine(model_manager)


def _get_model_manager():
    from neural_mri.main import model_manager

    return model_manager


@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time scan streaming.

    Protocol:
    - Client sends: {"type": "scan_stream", "mode": "fMRI"|"DTI", "prompt": "..."}
    - Server sends: {"type": "scan_start", "tokens": [...], "n_layers": N}
    - Server sends N: {"type": "activation_frame", "token_idx": i, "layers": [...]}
    - Server sends: {"type": "scan_complete", "compute_time_ms": ...}
    """
    await ws.accept()
    await ws.send_json({"type": "info", "message": "Neural MRI WebSocket connected."})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")
            if msg_type == "scan_stream":
                await _handle_scan_stream(ws, msg)
            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})
            else:
                await ws.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


async def _handle_scan_stream(ws: WebSocket, msg: dict) -> None:
    """Handle a scan_stream request: run forward pass and stream token-by-token frames."""
    mode = msg.get("mode", "fMRI")
    prompt = msg.get("prompt", "")

    if not prompt:
        await ws.send_json({"type": "error", "message": "Empty prompt"})
        return

    mm = _get_model_manager()
    if not mm.is_loaded:
        await ws.send_json({"type": "error", "message": "No model loaded"})
        return

    model = mm.get_model()
    cfg = model.cfg

    start = time.time()

    # Tokenize
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)
    seq_len = tokens.shape[1]

    # Forward pass with cache (in thread to avoid blocking)
    def _run_cache():
        with torch.no_grad():
            return model.run_with_cache(tokens)

    logits, cache = await asyncio.to_thread(_run_cache)

    # Send scan_start
    await ws.send_json(
        {
            "type": "scan_start",
            "mode": mode,
            "tokens": [str(t) for t in str_tokens],
            "n_layers": cfg.n_layers,
            "seq_len": seq_len,
        }
    )

    if mode == "fMRI":
        await _stream_fmri_frames(ws, model, cfg, cache, seq_len)
    elif mode == "DTI":
        await _stream_dti_frames(ws, model, cfg, cache, tokens, logits, seq_len)

    elapsed_ms = (time.time() - start) * 1000
    await ws.send_json(
        {
            "type": "scan_complete",
            "compute_time_ms": round(elapsed_ms, 1),
        }
    )


async def _stream_fmri_frames(ws, model, cfg, cache, seq_len: int) -> None:
    """Stream one activation_frame per token position."""
    # Collect all raw norms first for global normalization
    all_norms = []

    embed_norms = torch.norm(cache["hook_embed"][0], dim=-1).tolist()
    all_norms.extend(embed_norms)

    block_attn_norms = []
    block_mlp_norms = []
    for i in range(cfg.n_layers):
        attn_z = cache[f"blocks.{i}.attn.hook_z"]  # [1, seq, heads, d_head]
        a_norms = torch.norm(attn_z[0], dim=(-1, -2)).tolist()
        all_norms.extend(a_norms)
        block_attn_norms.append(a_norms)

        mlp_out = cache[f"blocks.{i}.hook_mlp_out"]
        m_norms = torch.norm(mlp_out[0], dim=-1).tolist()
        all_norms.extend(m_norms)
        block_mlp_norms.append(m_norms)

    resid = cache[f"blocks.{cfg.n_layers - 1}.hook_resid_post"]
    unembed_norms = torch.norm(resid[0], dim=-1).tolist()
    all_norms.extend(unembed_norms)

    norm_max = max(all_norms) if all_norms else 1.0
    norm_min = min(all_norms) if all_norms else 0.0
    rng = norm_max - norm_min if norm_max > norm_min else 1.0

    def norm(v):
        return round((v - norm_min) / rng, 4)

    # Stream frame per token
    for t in range(seq_len):
        layers = []
        layers.append({"layer_id": "embed", "activation": norm(embed_norms[t])})
        for i in range(cfg.n_layers):
            attn_val = norm(block_attn_norms[i][t])
            mlp_val = norm(block_mlp_norms[i][t])
            layers.append({"layer_id": f"blocks.{i}.attn", "activation": attn_val})
            layers.append({"layer_id": f"blocks.{i}.mlp", "activation": mlp_val})
        layers.append({"layer_id": "unembed", "activation": norm(unembed_norms[t])})

        await ws.send_json(
            {
                "type": "activation_frame",
                "token_idx": t,
                "layers": layers,
            }
        )
        # Small yield to allow client to process
        await asyncio.sleep(0.01)


async def _stream_dti_frames(ws, model, cfg, cache, tokens, logits, seq_len: int) -> None:
    """Stream DTI data: attention patterns + component importance."""
    target_idx = seq_len - 1

    # Send attention patterns
    for i in range(cfg.n_layers):
        pattern = cache[f"blocks.{i}.attn.hook_pattern"]  # [1, heads, seq, seq]
        for h in range(cfg.n_heads):
            await ws.send_json(
                {
                    "type": "attention_pattern",
                    "layer_idx": i,
                    "head_idx": h,
                    "pattern": pattern[0, h].tolist(),
                }
            )
        await asyncio.sleep(0.01)

    # Component importance via zero-ablation (streamed per component)
    baseline_logit = logits[0, target_idx].clone()
    component_ids = ["embed"]
    hook_points = ["hook_embed"]
    for i in range(cfg.n_layers):
        component_ids.append(f"blocks.{i}.attn")
        hook_points.append(f"blocks.{i}.hook_attn_out")
        component_ids.append(f"blocks.{i}.mlp")
        hook_points.append(f"blocks.{i}.hook_mlp_out")

    raw_importances = []
    for comp_id, hook_name in zip(component_ids, hook_points):

        def zero_hook(value, hook):
            return torch.zeros_like(value)

        def _ablate():
            with torch.no_grad():
                return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_hook)])

        ablated = await asyncio.to_thread(_ablate)
        diff = torch.norm(baseline_logit - ablated[0, target_idx]).item()
        raw_importances.append(diff)

    # Normalize and send
    imp_max = max(raw_importances) if raw_importances else 1.0
    imp_min = min(raw_importances) if raw_importances else 0.0
    imp_rng = imp_max - imp_min if imp_max > imp_min else 1.0
    threshold = 0.3

    for comp_id, raw_imp in zip(component_ids, raw_importances):
        norm_imp = round((raw_imp - imp_min) / imp_rng, 4)
        await ws.send_json(
            {
                "type": "component_importance",
                "layer_id": comp_id,
                "importance": norm_imp,
                "is_pathway": norm_imp >= threshold,
            }
        )
