#!/usr/bin/env python3
"""Paper #5 — Unified emotion vector extraction for MTI × Emotion × DFC triangulation.

Extracts comprehension-based + generation-based emotion vectors for cross-architecture
comparison. Outputs a single .pt file per model with multi-layer vectors, anisotropy,
and steering regime metadata.

Usage:
    # Priority pair (Qwen vs Llama — Paper #5 H2 test case)
    python scripts/paper5_extract.py --priority

    # All 6 models
    python scripts/paper5_extract.py

    # Specific model
    python scripts/paper5_extract.py --models "Qwen/Qwen2.5-1.5B-Instruct"

    # Skip generation mode (comprehension only, faster)
    python scripts/paper5_extract.py --no-generation

Requirements:
    pip install transformer-lens torch transformers accelerate

Output: scripts/paper5_output/<model_name>/
    - vectors_comprehension.pt   (multi-layer emotion vectors)
    - vectors_generation.pt      (if --no-generation not set)
    - metadata.json              (anisotropy, best layer, steering regime, etc.)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch

# TransformerLens: optional, will fallback to HF if not available or model unsupported
try:
    from transformer_lens import HookedTransformer

    HAS_TL = True
except ImportError:
    HAS_TL = False

# ── Config ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "paper5_output"

# Emotion comprehension texts (bundled in Neural-MRI)
TEXTS_FILE = PROJECT_ROOT / "backend" / "neural_mri" / "data" / "emotion_comprehension_texts.csv"

# Paper #5 target models (n=6)
MODELS_ALL = [
    # Instruct versions (main analysis)
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "google/gemma-3-1b-it",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    # Base versions (RLHF Shell-Core supplementary)
    "mistralai/Mistral-7B-v0.1",
    "HuggingFaceTB/SmolLM2-1.7B",
    "google/gemma-3-1b-pt",
    "google/gemma-2-2b",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-1.5B",
]

# Priority pair for H2 validation
MODELS_PRIORITY = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-3B",
]

# Steering regime thresholds (from Paper #6 findings)
STEERING_SCENARIOS = [
    {
        "name": "aggressive_to_calm",
        "prompt": "I am going to destroy everything you have built.",
        "emotion": "calm",
        "strengths": [0.005, 0.01, 0.02, 0.03, 0.05],
    },
    {
        "name": "neutral_to_hostile",
        "prompt": "The weather today is partly cloudy with a chance of rain.",
        "emotion": "hostile",
        "strengths": [0.005, 0.01, 0.02, 0.03, 0.05],
    },
]

# Generation prompts for emotion-eliciting text (Method 1)
GENERATION_PROMPTS = {
    "happy": "Write a short paragraph about a character who feels deeply happy:",
    "sad": "Write a short paragraph about a character who feels deeply sad:",
    "calm": "Write a short paragraph about a character who feels deeply calm:",
    "angry": "Write a short paragraph about a character who feels deeply angry:",
    "afraid": "Write a short paragraph about a character who feels deeply afraid:",
    "desperate": "Write a short paragraph about a character who feels deeply desperate:",
    "proud": "Write a short paragraph about a character who feels deeply proud:",
    "guilty": "Write a short paragraph about a character who feels deeply guilty:",
    "nervous": "Write a short paragraph about a character who feels deeply nervous:",
    "hopeful": "Write a short paragraph about a character who feels deeply hopeful:",
    "hostile": "Write a short paragraph about a character who feels deeply hostile:",
    "loving": "Write a short paragraph about a character who feels deeply loving:",
    "enthusiastic": "Write a short paragraph about a character who feels deeply enthusiastic:",
    "reflective": "Write a short paragraph about a character who feels deeply reflective:",
    "brooding": "Write a short paragraph about a character who feels deeply brooding:",
    "gloomy": "Write a short paragraph about a character who feels deeply gloomy:",
    "blissful": "Write a short paragraph about a character who feels deeply blissful:",
    "anxious": "Write a short paragraph about a character who feels deeply anxious:",
    "grateful": "Write a short paragraph about a character who feels deeply grateful:",
    "exasperated": "Write a short paragraph about a character who feels deeply exasperated:",
}


# ── Data Loading ────────────────────────────────────────────────────────────


def load_comprehension_texts() -> dict[str, list[str]]:
    """Load emotion → [passage, ...] from CSV."""
    texts: dict[str, list[str]] = {}
    with open(TEXTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            emotion = row[0].strip()
            passage = row[2].strip()
            texts.setdefault(emotion, []).append(passage)
    return texts


# ── Core Extraction ─────────────────────────────────────────────────────────


def extract_vectors_at_layer(
    model: HookedTransformer,
    texts: dict[str, list[str]],
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Extract emotion vectors via comprehension mode at a specific layer.

    Returns {emotion: vector} where vector is mean-subtracted.
    """
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    emotion_means: dict[str, torch.Tensor] = {}
    all_vecs: list[torch.Tensor] = []

    with torch.no_grad():
        for emotion, passages in texts.items():
            acts = []
            for passage in passages:
                tokens = model.to_tokens(passage)
                _, cache = model.run_with_cache(tokens)
                last_act = cache[hook_name][0, -1].float().cpu()
                acts.append(last_act)
            mean_vec = torch.stack(acts).mean(dim=0)
            emotion_means[emotion] = mean_vec
            all_vecs.append(mean_vec)

    global_mean = torch.stack(all_vecs).mean(dim=0)
    return {e: v - global_mean for e, v in emotion_means.items()}


# ── HF Transformers Fallback ─────────────────────────────────────────────


def load_hf_model(model_id: str, device: str, quantize: str | None = None):
    """Load model via HuggingFace transformers (fallback for TL-unsupported)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading via HF transformers (fallback)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"output_hidden_states": True}

    if quantize == "int8":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = device
    elif quantize == "int4":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["device_map"] = device
    else:
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    return model, tokenizer, n_layers, d_model


def extract_vectors_hf(
    model,
    tokenizer,
    texts: dict[str, list[str]],
    layer_idx: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Extract emotion vectors using HF model + hidden_states (no TL)."""
    emotion_means: dict[str, torch.Tensor] = {}
    all_vecs: list[torch.Tensor] = []

    with torch.no_grad():
        for emotion, passages in texts.items():
            acts = []
            for passage in passages:
                inputs = tokenizer(passage, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # hidden_states[0]=embed, [1..n]=layers
                act = outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().float()
                acts.append(act)
            mean_vec = torch.stack(acts).mean(dim=0)
            emotion_means[emotion] = mean_vec
            all_vecs.append(mean_vec)

    global_mean = torch.stack(all_vecs).mean(dim=0)
    return {e: v - global_mean for e, v in emotion_means.items()}


def compute_anisotropy_hf(model, tokenizer, layer_idx: int, device: str = "cuda") -> float:
    """Compute anisotropy using HF model."""
    neutral = [
        "Today is Tuesday.",
        "3 plus 4 equals 7.",
        "There is a cup on the table.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits the Sun.",
        "A triangle has three sides.",
        "Paris is the capital of France.",
        "The periodic table has 118 elements.",
        "A kilometer is 1000 meters.",
        "The speed of light is approximately 300,000 km/s.",
    ]
    vecs: list[torch.Tensor] = []
    with torch.no_grad():
        for s in neutral:
            inputs = tokenizer(s, return_tensors="pt").to(device)
            outputs = model(**inputs)
            vecs.append(outputs.hidden_states[layer_idx + 1][0, -1, :].cpu().float())
    mat = torch.stack(vecs)
    mat_norm = mat / mat.norm(dim=-1, keepdim=True)
    cos = mat_norm @ mat_norm.T
    n = cos.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return cos[mask].mean().item()


def find_best_layer_hf(model, tokenizer, texts, device, n_layers) -> tuple[int, list[dict]]:
    """Sweep all layers (HF backend)."""

    sweep: list[dict] = []
    best_layer = 0
    best_cos = 1.0
    print(f"    HF layer sweep (0-{n_layers - 1}):", end=" ", flush=True)
    for li in range(n_layers):
        vecs = extract_vectors_hf(model, tokenizer, texts, li, device)
        mat = torch.stack(list(vecs.values()))
        mat_n = mat / mat.norm(dim=-1, keepdim=True)
        cos = mat_n @ mat_n.T
        n = cos.shape[0]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        mc = cos[mask].mean().item()
        sweep.append({"layer": li, "mean_cosine": round(mc, 4)})
        if mc < best_cos:
            best_cos = mc
            best_layer = li
        print("." if li % 4 else str(li), end="", flush=True)
    print(f" → best={best_layer} (cos={best_cos:.4f})")
    return best_layer, sweep


def extract_vectors_generation(
    model,
    layer_idx: int,
    max_new_tokens: int = 100,
) -> dict[str, torch.Tensor]:
    """Extract emotion vectors via generation mode at a specific layer.

    Generates emotion-eliciting text, then extracts activation at last generated token.
    """
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    emotion_means: dict[str, torch.Tensor] = {}
    all_vecs: list[torch.Tensor] = []

    with torch.no_grad():
        for emotion, prompt in GENERATION_PROMPTS.items():
            tokens = model.to_tokens(prompt)
            # Generate
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            # Forward pass on full generated text
            _, cache = model.run_with_cache(output)
            last_act = cache[hook_name][0, -1].float().cpu()
            emotion_means[emotion] = last_act
            all_vecs.append(last_act)

    global_mean = torch.stack(all_vecs).mean(dim=0)
    return {e: v - global_mean for e, v in emotion_means.items()}


def compute_anisotropy(
    model: HookedTransformer,
    layer_idx: int,
    n_sentences: int = 20,
) -> float:
    """Compute mean pairwise cosine similarity of neutral sentences (baseline anisotropy)."""
    neutral_sentences = [
        "Today is Tuesday.",
        "3 plus 4 equals 7.",
        "There is a cup on the table.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits the Sun.",
        "A triangle has three sides.",
        "Paris is the capital of France.",
        "The periodic table has 118 elements.",
        "A kilometer is 1000 meters.",
        "The speed of light is approximately 300,000 km/s.",
        "Photosynthesis converts sunlight into energy.",
        "The Atlantic Ocean separates Europe from America.",
        "A standard piano has 88 keys.",
        "DNA stands for deoxyribonucleic acid.",
        "The square root of 144 is 12.",
        "Nitrogen makes up about 78% of the atmosphere.",
        "Mercury is the closest planet to the Sun.",
        "An octagon has eight sides.",
        "The human body has 206 bones.",
        "Sound travels at about 343 meters per second.",
    ]
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    vecs: list[torch.Tensor] = []

    with torch.no_grad():
        for s in neutral_sentences[:n_sentences]:
            tokens = model.to_tokens(s)
            _, cache = model.run_with_cache(tokens)
            vecs.append(cache[hook_name][0, -1].float().cpu())

    mat = torch.stack(vecs)
    mat_norm = mat / mat.norm(dim=-1, keepdim=True)
    cos_sim = mat_norm @ mat_norm.T
    # Mean of upper triangle (excluding diagonal)
    n = cos_sim.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return cos_sim[mask].mean().item()


def find_best_layer(
    model: HookedTransformer,
    texts: dict[str, list[str]],
) -> tuple[int, list[dict]]:
    """Sweep all layers, find the one with lowest mean pairwise cosine (best separation).

    Returns (best_layer_idx, sweep_data).
    """
    n_layers = model.cfg.n_layers
    sweep_data: list[dict] = []
    best_layer = 0
    best_cosine = 1.0

    print(f"    Layer sweep (0-{n_layers - 1}):", end=" ", flush=True)

    for layer_idx in range(n_layers):
        vectors = extract_vectors_at_layer(model, texts, layer_idx)
        # Compute mean pairwise cosine
        vecs = torch.stack(list(vectors.values()))
        vecs_norm = vecs / vecs.norm(dim=-1, keepdim=True)
        cos_sim = vecs_norm @ vecs_norm.T
        n = cos_sim.shape[0]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        mean_cosine = cos_sim[mask].mean().item()

        sweep_data.append({"layer": layer_idx, "mean_cosine": round(mean_cosine, 4)})

        if mean_cosine < best_cosine:
            best_cosine = mean_cosine
            best_layer = layer_idx

        # Progress dot
        if layer_idx % 4 == 0:
            print(f"{layer_idx}", end="", flush=True)
        else:
            print(".", end="", flush=True)

    print(f" → best={best_layer} (cosine={best_cosine:.4f})")
    return best_layer, sweep_data


def classify_steering_regime(
    model: HookedTransformer,
    vectors: dict[str, torch.Tensor],
    layer_idx: int,
) -> dict:
    """Classify model's steering response pattern.

    Regimes (from Paper #6):
    - surgical: clean behavioral change, new coherent content
    - repetitive_collapse: output degenerates into repetition
    - explosive: erratic/incoherent output
    """
    n_layers = model.cfg.n_layers
    results: list[dict] = []

    for scenario in STEERING_SCENARIOS:
        emotion = scenario["emotion"]
        if emotion not in vectors:
            continue

        emotion_vec = vectors[emotion]
        steer_dir = emotion_vec / (emotion_vec.norm() + 1e-8)
        tokens = model.to_tokens(scenario["prompt"])
        hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

        for strength in scenario["strengths"]:

            def make_hook(s):
                def hook_fn(value, hook):
                    resid_norm = value.norm(dim=-1, keepdim=True).mean()
                    return value + steer_dir.to(value.device, value.dtype) * s * resid_norm

                return hook_fn

            with torch.no_grad():
                hook_fn = make_hook(strength)
                steered_ids = tokens.clone()
                for _ in range(20):
                    logits = model.run_with_hooks(
                        steered_ids,
                        fwd_hooks=[(n, hook_fn) for n in hook_names],
                    )
                    next_tok = logits[0, -1].argmax(dim=-1, keepdim=True)
                    steered_ids = torch.cat([steered_ids, next_tok.unsqueeze(0)], dim=-1)
                    if next_tok.item() == model.tokenizer.eos_token_id:
                        break

                text = model.tokenizer.decode(
                    steered_ids[0, tokens.shape[1] :],
                    skip_special_tokens=True,
                )

            results.append(
                {
                    "scenario": scenario["name"],
                    "emotion": emotion,
                    "strength": strength,
                    "text": text[:200],
                }
            )

    # Classify regime from results
    regime = "unknown"
    if results:
        texts = [r["text"] for r in results]
        # Check for repetition (same 3-gram appearing 3+ times)
        rep_count = sum(1 for t in texts if _has_repetition(t))
        # Check for incoherence (very short or garbled)
        incoherent_count = sum(1 for t in texts if len(t.split()) < 3)

        if rep_count > len(texts) * 0.5:
            regime = "repetitive_collapse"
        elif incoherent_count > len(texts) * 0.3:
            regime = "explosive"
        else:
            regime = "surgical"

    return {"regime": regime, "steering_samples": results}


def _has_repetition(text: str, ngram: int = 3, threshold: int = 3) -> bool:
    """Check if any n-gram repeats threshold+ times."""
    words = text.lower().split()
    if len(words) < ngram:
        return False
    ngrams: dict[str, int] = {}
    for i in range(len(words) - ngram + 1):
        key = " ".join(words[i : i + ngram])
        ngrams[key] = ngrams.get(key, 0) + 1
        if ngrams[key] >= threshold:
            return True
    return False


# ── Main Processing ─────────────────────────────────────────────────────────


def process_model(
    model_id: str,
    texts: dict[str, list[str]],
    include_generation: bool = True,
    device: str = "auto",
) -> None:
    """Process a single model: extract vectors, compute metadata, save.

    Two-phase approach:
    1. Try TransformerLens (primary) — supports hooks, steering regime
    2. Fall back to HF transformers if TL fails — hidden_states extraction
    Partial success is preserved: if comprehension works but generation
    fails, comprehension data is still saved.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    safe_name = model_id.replace("/", "_")
    out_dir = OUTPUT_DIR / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing: {model_id}")
    print(f"Output: {out_dir}")
    print(f"{'=' * 60}")

    start = time.time()
    backend = "unknown"
    tl_model = None
    hf_model = None
    hf_tokenizer = None
    n_layers = 0
    d_model = 0

    # ── Phase 1: Try TransformerLens ──
    if HAS_TL:
        try:
            print(f"  Trying TransformerLens on {device}...")
            tl_model = HookedTransformer.from_pretrained(
                model_id, device=device, dtype=torch.float16
            )
            n_layers = tl_model.cfg.n_layers
            d_model = tl_model.cfg.d_model
            backend = "transformer_lens"
            print(f"  TL loaded: {n_layers}L, d={d_model}")
        except Exception as e:
            print(f"  TL failed: {e}")
            tl_model = None

    # ── Phase 2: Fallback to HF transformers ──
    if tl_model is None:
        try:
            print("  Falling back to HF transformers...")
            hf_model, hf_tokenizer, n_layers, d_model = load_hf_model(model_id, device)
            backend = "hf_raw_hooks"
            print(f"  HF loaded: {n_layers}L, d={d_model}")
        except Exception as e:
            print(f"  HF also failed: {e}")
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "model_id": model_id,
                        "error": str(e),
                        "extraction_backend": "failed",
                    },
                    f,
                    indent=2,
                )
            return

    # Target layers
    target_layers = {
        "25pct": max(0, n_layers // 4),
        "50pct": n_layers // 2,
        "75pct": n_layers * 3 // 4,
    }
    print(f"  Backend: {backend}, Target layers: {target_layers}")

    # ── Step 1: Layer sweep ──
    print("  Step 1: Layer sweep...")
    if tl_model:
        best_layer, sweep_data = find_best_layer(tl_model, texts)
    else:
        best_layer, sweep_data = find_best_layer_hf(hf_model, hf_tokenizer, texts, device, n_layers)
    target_layers["best"] = best_layer

    # ── Step 2: Comprehension vectors ──
    comp_status = "success"
    comp_vectors: dict[str, dict[str, torch.Tensor]] = {}
    try:
        for label, layer_idx in target_layers.items():
            print(f"  Comprehension @ layer {layer_idx} ({label})...")
            if tl_model:
                comp_vectors[str(layer_idx)] = extract_vectors_at_layer(tl_model, texts, layer_idx)
            else:
                comp_vectors[str(layer_idx)] = extract_vectors_hf(
                    hf_model, hf_tokenizer, texts, layer_idx, device
                )
        torch.save(
            {
                "model_id": model_id,
                "n_layers": n_layers,
                "d_model": d_model,
                "target_layers": target_layers,
                "vectors": {k: {e: v for e, v in vecs.items()} for k, vecs in comp_vectors.items()},
            },
            out_dir / "vectors_comprehension.pt",
        )
        print("  Saved vectors_comprehension.pt")
    except Exception as e:
        comp_status = f"failed: {e}"
        print(f"  Comprehension FAILED: {e}")

    # ── Step 3: Generation vectors (independent, partial success OK) ──
    gen_status = "skipped"
    if include_generation and comp_status == "success":
        gen_status = "attempted"
        gen_layer = target_layers["best"]
        print(f"  Generation @ layer {gen_layer}...")
        try:
            if tl_model:
                gen_vecs = extract_vectors_generation(tl_model, gen_layer)
            else:
                gen_status = "skipped_hf"  # HF can't easily do steered gen
                gen_vecs = None
            if gen_vecs:
                torch.save(
                    {"model_id": model_id, "layer": gen_layer, "vectors": gen_vecs},
                    out_dir / "vectors_generation.pt",
                )
                gen_status = "success"
                print("  Saved vectors_generation.pt")
        except Exception as e:
            gen_status = f"failed: {e}"
            print(f"  Generation FAILED (comprehension preserved): {e}")

    # ── Step 4: Anisotropy ──
    anisotropy: dict[str, float] = {}
    if comp_status == "success":
        for label, layer_idx in target_layers.items():
            if tl_model:
                ani = compute_anisotropy(tl_model, layer_idx)
            else:
                ani = compute_anisotropy_hf(hf_model, hf_tokenizer, layer_idx, device)
            anisotropy[f"layer_{layer_idx}_{label}"] = round(ani, 4)
        print(f"  Anisotropy: {anisotropy}")

    # ── Step 5: Steering regime (TL only) ──
    steering_result = {"regime": "unknown", "steering_samples": []}
    if tl_model and comp_status == "success":
        print("  Classifying steering regime...")
        try:
            best_vecs = comp_vectors[str(target_layers["best"])]
            steering_result = classify_steering_regime(tl_model, best_vecs, target_layers["best"])
            print(f"  Regime: {steering_result['regime']}")
        except Exception as e:
            steering_result["regime"] = f"failed: {e}"
            print(f"  Steering regime failed: {e}")
    elif not tl_model:
        steering_result["regime"] = "not_available_hf_backend"

    # ── Step 6: Pairwise cosines ──
    pairwise_cosines: dict[str, float] = {}
    if comp_status == "success":
        for label, layer_idx in target_layers.items():
            vecs = comp_vectors[str(layer_idx)]
            mat = torch.stack(list(vecs.values()))
            mat_n = mat / mat.norm(dim=-1, keepdim=True)
            cos = mat_n @ mat_n.T
            n = cos.shape[0]
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            pairwise_cosines[f"layer_{layer_idx}_{label}"] = round(cos[mask].mean().item(), 4)

    # ── Save metadata (always, even on partial failure) ──
    metadata = {
        "model_id": model_id,
        "extraction_backend": backend,
        "comprehension_status": comp_status,
        "generation_status": gen_status,
        "n_layers": n_layers,
        "d_model": d_model,
        "target_layers": target_layers,
        "best_layer": best_layer,
        "best_layer_cosine": round(sweep_data[best_layer]["mean_cosine"], 4)
        if sweep_data
        else None,
        "anisotropy": anisotropy,
        "pairwise_cosines": pairwise_cosines,
        "steering_regime": steering_result["regime"],
        "steering_samples": steering_result["steering_samples"],
        "layer_sweep": sweep_data,
        "emotions": sorted(comp_vectors[str(target_layers["best"])].keys())
        if comp_status == "success"
        else [],
        "n_emotions": len(comp_vectors[str(target_layers["best"])])
        if comp_status == "success"
        else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved metadata.json")

    # ── Cleanup ──
    del tl_model, hf_model, hf_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elapsed = time.time() - start
    print(f"  Done! backend={backend}, comp={comp_status}, gen={gen_status} ({elapsed:.1f}s)")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Paper #5 — Unified emotion vector extraction",
    )
    parser.add_argument(
        "--priority",
        action="store_true",
        help="Run Qwen + Llama priority pair only",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model IDs",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip generation-mode extraction",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output:
        OUTPUT_DIR = Path(args.output)

    # Select models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.priority:
        models = MODELS_PRIORITY
    else:
        models = MODELS_ALL

    print("Paper #5 Emotion Vector Extraction")
    print(f"Models: {len(models)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Generation: {'yes' if not args.no_generation else 'no'}")
    print(f"Device: {args.device}")
    print()

    # Load texts
    if not TEXTS_FILE.exists():
        print(f"ERROR: Comprehension texts not found at {TEXTS_FILE}")
        print("Run from Neural-MRI project root.")
        sys.exit(1)

    texts = load_comprehension_texts()
    print(f"Loaded {sum(len(v) for v in texts.values())} passages for {len(texts)} emotions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each model
    failed: list[str] = []
    for model_id in models:
        try:
            process_model(
                model_id,
                texts,
                include_generation=not args.no_generation,
                device=args.device,
            )
        except Exception as e:
            print(f"\n  ERROR processing {model_id}: {e}")
            failed.append(model_id)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(models) - len(failed)}/{len(models)} models processed")
    if failed:
        print(f"FAILED: {failed}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
