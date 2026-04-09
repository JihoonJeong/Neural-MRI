"""Paper #5 Finding #8 — Protocol-Matched fp16 Generation Extraction

PURPOSE: Replicates Paper #6's emotion_vector_pilot_v2.py protocol EXACTLY,
changing only the precision (INT8 → fp16) so that a clean precision comparison
becomes possible.

The 8 protocol parameters held identical to Paper #6:
  1. 5 story templates × 10 generations per emotion (50 samples)
  2. Greedy decoding (do_sample=False)
  3. max_new_tokens = 256
  4. Mid-generation extraction position (full_ids[prompt_len + gen_len // 2])
  5. Layer = middle (n_layers // 2 = 16 for 32-layer models)
  6. apply_chat_template = True
  7. Vector formula: (emotion_mean - neutral_mean), unit-normalized
  8. 10 neutral baseline stories (same templates as Paper #6)

The only delta from Paper #6:
  - dtype = torch.float16 (Paper #6 used INT8 quantization)

Output: scripts/paper5_output_protocol_matched/<model>/emotion_vectors_v2.pt
        (same .pt format as Paper #6 for direct comparison)

Usage:
    python3 scripts/paper5_extract_protocol_matched.py \
        --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch

# ── CONFIG (matches Paper #6's emotion_vector_pilot_v2.py CONFIG exactly,
#            except dtype/quantize) ────────────────────────────────────────
CONFIG = {
    "device": "cuda",
    "dtype": torch.float16,
    "quantize": None,  # ← The ONLY change vs Paper #6 (was "int8")
    "max_new_tokens": 256,
    "n_stories_per_emotion": 10,
    "n_neutral_stories": 10,
    "extraction_layer": "middle",
    "extraction_position": "mid_generation",
    "output_root": Path(__file__).parent.parent / "scripts" / "paper5_output_protocol_matched",
}

# Same 20 emotions as Paper #6 (subset of Paper #5's 21)
EMOTIONS = [
    "happy", "sad", "calm", "desperate", "afraid",
    "angry", "proud", "guilty", "nervous", "hopeful",
    "brooding", "gloomy", "reflective", "enthusiastic", "hostile",
    "loving", "exasperated", "blissful", "anxious", "grateful",
]


# ── Prompt templates (verbatim from Paper #6) ────────────────────────────


def make_emotion_prompt(emotion: str, variant: int = 0) -> str:
    templates = [
        f"Write a short story (3-5 paragraphs) about a character who is feeling {emotion}. Show the emotion through the character's thoughts, actions, and interactions. Do not name the emotion directly.",
        f"Tell a brief story about someone experiencing deep {emotion}. Focus on their internal monologue and physical sensations.",
        f"Write a scene where a person is overwhelmed by {emotion}. Describe what they see, think, and do.",
        f"Describe a moment in someone's life when they feel intensely {emotion}. Use vivid details.",
        f"Write a narrative about a character going through {emotion}. Show, don't tell.",
    ]
    return templates[variant % len(templates)]


def make_neutral_prompt(idx: int) -> str:
    topics = [
        "a person organizing their bookshelf",
        "a person walking through a park on a mild day",
        "a person cooking a simple meal",
        "a person waiting at a bus stop",
        "a person reading a newspaper",
        "a person folding laundry",
        "a person watering plants",
        "a person checking the weather forecast",
        "a person writing a grocery list",
        "a person sorting mail",
    ]
    return (
        f"Write a short story (3-5 paragraphs) about {topics[idx % len(topics)]}. "
        f"Keep the tone neutral and matter-of-fact."
    )


# ── Model loading (fp16 only — Paper #6 had int8 branch) ──────────────────


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    print(f"  Precision: fp16 (no quantization)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=CONFIG["dtype"],
        device_map=CONFIG["device"],
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    vram_gb = torch.cuda.memory_allocated() / 1e9

    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}, VRAM: {vram_gb:.1f}GB")
    return model, tokenizer, n_layers, hidden_dim


# ── Generation + extraction (verbatim from Paper #6) ─────────────────────


@torch.no_grad()
def generate_and_extract(model, tokenizer, prompt: str, n_layers: int):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(CONFIG["device"])
    prompt_len = inputs["input_ids"].shape[1]

    # Greedy generation (Paper #6 setting)
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=CONFIG["max_new_tokens"],
        do_sample=False,
    )

    full_ids = gen_outputs[0]
    generated_len = full_ids.shape[0] - prompt_len

    # Mid-generation extraction position
    if CONFIG["extraction_position"] == "mid_generation" and generated_len > 10:
        extract_pos = prompt_len + generated_len // 2
    else:
        extract_pos = prompt_len - 1

    # Forward pass on full sequence to get hidden states
    outputs = model(full_ids.unsqueeze(0), output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)

    # Per-layer activations at extract_pos
    layer_activations = []
    for layer_idx in range(1, len(hidden_states)):  # skip embedding
        activation = hidden_states[layer_idx][0, extract_pos, :].cpu().float()
        layer_activations.append(activation)

    generated_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)

    del outputs, hidden_states
    torch.cuda.empty_cache()

    return torch.stack(layer_activations), generated_text


# ── Vector computation (Paper #6 formula: neutral baseline + unit norm) ──


def compute_emotion_vectors(emotion_activations, neutral_mean, layer_idx):
    vectors = {}
    for emotion, acts_list in emotion_activations.items():
        stacked = torch.stack([a[layer_idx] for a in acts_list])
        emotion_mean = stacked.mean(dim=0)
        vector = emotion_mean - neutral_mean
        norm = vector.norm()
        if norm > 0:
            vector = vector / norm
        vectors[emotion] = vector
    return vectors


# ── Main pipeline ────────────────────────────────────────────────────────


def process_model(model_name: str):
    print("=" * 70)
    print(f"Protocol-Matched fp16 Extraction — {model_name}")
    print(f"Replicates Paper #6 protocol exactly, only dtype=fp16")
    print("=" * 70)

    safe_name = model_name.replace("/", "_")
    output_dir = CONFIG["output_root"] / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, n_layers, hidden_dim = load_model(model_name)

    # Layer selection — same as Paper #6 (middle = n_layers // 2)
    if CONFIG["extraction_layer"] == "middle":
        layer_idx = n_layers // 2
    else:
        layer_idx = CONFIG["extraction_layer"]
    print(f"Extraction: layer {layer_idx}/{n_layers}, position: {CONFIG['extraction_position']}")

    # ── Phase 1: Neutral baseline ──
    print(f"\n--- Phase 1: Neutral baseline ({CONFIG['n_neutral_stories']} stories) ---")
    neutral_activations = []
    for i in range(CONFIG["n_neutral_stories"]):
        prompt = make_neutral_prompt(i)
        print(f"  Neutral {i+1}/{CONFIG['n_neutral_stories']}...", end=" ", flush=True)
        acts, text = generate_and_extract(model, tokenizer, prompt, n_layers)
        neutral_activations.append(acts)
        print(f"({len(text)} chars)")

    neutral_mean = torch.stack([a[layer_idx] for a in neutral_activations]).mean(dim=0)

    # ── Phase 2: Emotion stories ──
    print(f"\n--- Phase 2: Emotion stories ({len(EMOTIONS)} × {CONFIG['n_stories_per_emotion']}) ---")
    emotion_activations: dict[str, list] = {}
    emotion_stories: dict[str, list[str]] = {}

    for emotion in EMOTIONS:
        print(f"\n  [{emotion}]")
        emotion_activations[emotion] = []
        emotion_stories[emotion] = []

        for s in range(CONFIG["n_stories_per_emotion"]):
            prompt = make_emotion_prompt(emotion, variant=s)
            print(f"    {s+1}/{CONFIG['n_stories_per_emotion']}...", end=" ", flush=True)
            acts, text = generate_and_extract(model, tokenizer, prompt, n_layers)
            emotion_activations[emotion].append(acts)
            emotion_stories[emotion].append(text[:200])
            print(f"({len(text)}c)")

    # ── Phase 3: Compute vectors (Paper #6 formula) ──
    print(f"\n--- Phase 3: Computing emotion vectors ---")
    vectors = compute_emotion_vectors(emotion_activations, neutral_mean, layer_idx)

    all_vecs = torch.stack(list(vectors.values()))
    cos_matrix = torch.mm(all_vecs, all_vecs.t())
    mask = ~torch.eye(len(vectors), dtype=bool)
    mean_cos = cos_matrix[mask].mean().item()

    # Key pair sanity checks
    pair_checks = {}
    check_pairs = [("happy", "sad"), ("calm", "desperate"), ("loving", "hostile"), ("proud", "guilty")]
    for a, b in check_pairs:
        if a in vectors and b in vectors:
            cos = torch.dot(vectors[a], vectors[b]).item()
            pair_checks[f"{a}<->{b}"] = round(cos, 4)
            print(f"  {a} <-> {b}: {cos:.3f}")

    print(f"  Mean pairwise cosine: {mean_cos:.3f}")

    # ── Save (same .pt format as Paper #6 for direct compatibility) ──
    torch.save(
        {"vectors": vectors, "layer": layer_idx},
        output_dir / "emotion_vectors_v2.pt",
    )

    report = {
        "version": "v2_protocol_matched_fp16",
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "precision": "fp16",
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "extraction_layer": layer_idx,
        "extraction_position": CONFIG["extraction_position"],
        "n_emotions": len(EMOTIONS),
        "n_stories_per_emotion": CONFIG["n_stories_per_emotion"],
        "n_neutral_stories": CONFIG["n_neutral_stories"],
        "max_new_tokens": CONFIG["max_new_tokens"],
        "do_sample": False,
        "vector_formula": "(emotion_mean - neutral_mean) / norm",
        "applies_chat_template": True,
        "mean_pairwise_cosine": mean_cos,
        "pair_checks": pair_checks,
        "fork_of": "ludex/research/emotion_benchmark/emotion_vector_pilot_v2.py",
        "delta_from_source": "dtype=fp16 (was int8); model_name parameterized via CLI",
    }

    with open(output_dir / "pilot_report_v2.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save emotion story samples for inspection
    with open(output_dir / "emotion_stories_sample.json", "w") as f:
        json.dump(emotion_stories, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved → {output_dir}/")
    print(f"    emotion_vectors_v2.pt")
    print(f"    pilot_report_v2.json")
    print(f"    emotion_stories_sample.json")

    # Free model
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        required=True,
        help="HF model id (can be repeated for multiple models)",
    )
    args = parser.parse_args()

    for m in args.model:
        process_model(m)
        print()

    print("=" * 70)
    print(f"COMPLETE: {len(args.model)} model(s) processed")
    print(f"Output: {CONFIG['output_root']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
