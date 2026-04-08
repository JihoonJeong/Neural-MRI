"""Paper #5 — n=12 Comprehensive Analysis (Tasks 1-4)

Tasks from Luca:
  1. RDM-of-RDMs 12×12 heatmap
  2. Precision confound (fp16 vs INT8)
  3. Size effect analysis
  4. Tier 1 cluster re-verification

Usage:
    cd backend && uv run python ../paper5/analysis/n12_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

OUTPUT_DIR = Path(__file__).parent / "n12_output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "paper5_output"
P6_DIR = Path("/Users/jihoon/Projects/ludex/research/emotion_benchmark")

# All 12 models (6 instruct + 6 base)
ALL_MODELS = [
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-1.5B",
    "HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "HuggingFaceTB_SmolLM2-1.7B",
    "meta-llama_Llama-3.2-3B-Instruct",
    "meta-llama_Llama-3.2-3B",
    "google_gemma-3-1b-it",
    "google_gemma-3-1b-pt",
    "mistralai_Mistral-7B-v0.1",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "meta-llama_Llama-3.1-8B",
    "meta-llama_Llama-3.1-8B-Instruct",
]

INSTRUCT_MODELS = [
    "Qwen_Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "meta-llama_Llama-3.2-3B-Instruct",
    "google_gemma-3-1b-it",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "meta-llama_Llama-3.1-8B-Instruct",
]

# Short display names
SHORT_NAMES = {
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen1.5B-I",
    "Qwen_Qwen2.5-1.5B": "Qwen1.5B-B",
    "HuggingFaceTB_SmolLM2-1.7B-Instruct": "SmolLM1.7B-I",
    "HuggingFaceTB_SmolLM2-1.7B": "SmolLM1.7B-B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama3.2-3B-I",
    "meta-llama_Llama-3.2-3B": "Llama3.2-3B-B",
    "google_gemma-3-1b-it": "Gemma1B-IT",
    "google_gemma-3-1b-pt": "Gemma1B-PT",
    "mistralai_Mistral-7B-v0.1": "Mistral7B-B",
    "mistralai_Mistral-7B-Instruct-v0.3": "Mistral7B-I",
    "meta-llama_Llama-3.1-8B": "Llama3.1-8B-B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama3.1-8B-I",
}

MODEL_SIZES = {
    "Qwen_Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen_Qwen2.5-1.5B": 1.5,
    "HuggingFaceTB_SmolLM2-1.7B-Instruct": 1.7,
    "HuggingFaceTB_SmolLM2-1.7B": 1.7,
    "meta-llama_Llama-3.2-3B-Instruct": 3.0,
    "meta-llama_Llama-3.2-3B": 3.0,
    "google_gemma-3-1b-it": 1.0,
    "google_gemma-3-1b-pt": 1.0,
    "mistralai_Mistral-7B-v0.1": 7.0,
    "mistralai_Mistral-7B-Instruct-v0.3": 7.0,
    "meta-llama_Llama-3.1-8B": 8.0,
    "meta-llama_Llama-3.1-8B-Instruct": 8.0,
}


# ── Data loading ─────────────────────────────────────────────────────────


def load_model_data(model_dir: str) -> dict | None:
    d = DATA_DIR / model_dir
    if not (d / "metadata.json").exists():
        print(f"  SKIP: {model_dir} (no metadata)")
        return None
    meta = json.loads((d / "metadata.json").read_text())
    pt = torch.load(d / "vectors_comprehension.pt", map_location="cpu", weights_only=False)
    return {"meta": meta, "pt": pt, "dir": model_dir}


def get_best_layer_vectors(data: dict) -> dict[str, np.ndarray]:
    best = str(data["meta"]["best_layer"])
    vectors = data["pt"]["vectors"][best]
    return {e: v.numpy() for e, v in vectors.items()}


def compute_rdm(vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    emotions = sorted(vectors.keys())
    mat = np.stack([vectors[e] for e in emotions])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_norm = mat / norms
    cos_sim = mat_norm @ mat_norm.T
    return 1 - cos_sim, emotions


def upper_triangle(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return mat[mask]


# ── Task 1: RDM-of-RDMs 12×12 heatmap ───────────────────────────────────


def task1_rdm_of_rdms(all_data: dict):
    print("\n" + "=" * 60)
    print("TASK 1: RDM-of-RDMs 12×12 Heatmap")
    print("=" * 60)

    models = list(all_data.keys())
    n = len(models)

    # Compute per-model RDMs and upper triangles
    rdms = {}
    for m in models:
        vecs = get_best_layer_vectors(all_data[m])
        rdm, emos = compute_rdm(vecs)
        rdms[m] = upper_triangle(rdm)

    # Compute Spearman ρ between all model pairs
    rho_matrix = np.ones((n, n))
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, p = stats.spearmanr(rdms[models[i]], rdms[models[j]])
            rho_matrix[i, j] = rho
            rho_matrix[j, i] = rho
            p_matrix[i, j] = p
            p_matrix[j, i] = p

    labels = [SHORT_NAMES[m] for m in models]

    # Heatmap with hierarchical clustering
    fig, (ax_dendro, ax_heat) = plt.subplots(
        1, 2, figsize=(14, 10),
        gridspec_kw={"width_ratios": [1, 4]}
    )

    # Dendrogram
    dist = 1 - rho_matrix
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="average")
    dendro = dendrogram(Z, orientation="left", labels=labels, ax=ax_dendro,
                        leaf_font_size=9, color_threshold=0)
    ax_dendro.set_xlabel("1 - ρ")
    ax_dendro.invert_yaxis()

    # Reorder by dendrogram
    order = dendro["leaves"]
    rho_ordered = rho_matrix[np.ix_(order, order)]
    labels_ordered = [labels[i] for i in order]

    im = ax_heat.imshow(rho_ordered, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=9)
    ax_heat.set_yticklabels(labels_ordered, fontsize=9)

    # Annotate values
    for i in range(n):
        for j in range(n):
            color = "white" if rho_ordered[i, j] < 0.4 or rho_ordered[i, j] > 0.85 else "black"
            ax_heat.text(j, i, f"{rho_ordered[i, j]:.2f}",
                         ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax_heat, shrink=0.7, label="Spearman ρ")
    plt.suptitle("RDM-of-RDMs: 12-Model Emotion Geometry Similarity\n(raw cosine, best layer)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task1_rdm_of_rdms_12x12.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved task1_rdm_of_rdms_12x12.png")

    # Instruct-only 6×6 for cleaner comparison
    inst_models = [m for m in models if m in INSTRUCT_MODELS]
    n_inst = len(inst_models)
    rho_inst = np.ones((n_inst, n_inst))
    for i in range(n_inst):
        for j in range(i + 1, n_inst):
            rho, _ = stats.spearmanr(rdms[inst_models[i]], rdms[inst_models[j]])
            rho_inst[i, j] = rho
            rho_inst[j, i] = rho

    inst_labels = [SHORT_NAMES[m] for m in inst_models]

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im2 = ax2.imshow(rho_inst, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax2.set_xticks(range(n_inst))
    ax2.set_yticks(range(n_inst))
    ax2.set_xticklabels(inst_labels, rotation=45, ha="right", fontsize=10)
    ax2.set_yticklabels(inst_labels, fontsize=10)
    for i in range(n_inst):
        for j in range(n_inst):
            color = "white" if rho_inst[i, j] < 0.4 or rho_inst[i, j] > 0.85 else "black"
            ax2.text(j, i, f"{rho_inst[i, j]:.2f}",
                     ha="center", va="center", fontsize=9, color=color)
    fig2.colorbar(im2, ax=ax2, shrink=0.7, label="Spearman ρ")
    ax2.set_title("Instruct-Only 6×6 RDM-of-RDMs", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task1_rdm_instruct_6x6.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved task1_rdm_instruct_6x6.png")

    # Print summary
    print("\n  12×12 Spearman ρ matrix (instruct models):")
    for i, m in enumerate(inst_models):
        row = "  " + SHORT_NAMES[m].ljust(16)
        for j, m2 in enumerate(inst_models):
            idx_i = models.index(m)
            idx_j = models.index(m2)
            row += f" {rho_matrix[idx_i, idx_j]:.3f}"
        print(row)

    # Check Luca's scenarios
    print("\n  Luca Scenario Check:")
    mistral_idx = models.index("mistralai_Mistral-7B-v0.1") if "mistralai_Mistral-7B-v0.1" in models else None
    if mistral_idx is not None:
        tier1_candidates = ["Qwen_Qwen2.5-1.5B-Instruct", "meta-llama_Llama-3.2-3B-Instruct",
                            "HuggingFaceTB_SmolLM2-1.7B-Instruct"]
        mistral_rhos = []
        for t in tier1_candidates:
            if t in models:
                idx = models.index(t)
                r = rho_matrix[mistral_idx, idx]
                mistral_rhos.append(r)
                print(f"    Mistral7B vs {SHORT_NAMES[t]}: ρ={r:.4f}")
        if all(r > 0.85 for r in mistral_rhos):
            print("    → Mistral at Tier 1 CENTER (all ρ > 0.85)")
        elif all(r > 0.7 for r in mistral_rhos):
            print("    → Mistral IN Tier 1 (ρ > 0.7)")
        else:
            print("    → Mistral OUTSIDE Tier 1")

    return {"rho_matrix": rho_matrix, "models": models, "rdms": rdms}


# ── Task 2: Precision confound (fp16 vs INT8) ───────────────────────────


def task2_precision_confound(all_data: dict):
    print("\n" + "=" * 60)
    print("TASK 2: Precision Confound — fp16 vs INT8")
    print("=" * 60)

    # Paper #6 INT8 generation vectors
    p6_models = {
        "Mistral 7B Instruct": {
            "fp16_key": "mistralai_Mistral-7B-Instruct-v0.3",
            "int8_gen": P6_DIR / "emotion_pilot_v2_output" / "emotion_vectors_v2.pt",
        },
        "Llama 3.1 8B Instruct": {
            "fp16_key": "meta-llama_Llama-3.1-8B-Instruct",
            "int8_gen": P6_DIR / "llama31_inst_gen_output" / "emotion_vectors_v2.pt",
        },
        "Llama 3.1 8B Base": {
            "fp16_key": "meta-llama_Llama-3.1-8B",
            "int8_gen": P6_DIR / "llama31_base_gen_output" / "emotion_vectors_v2.pt",
        },
        "Llama 3.2 3B Instruct": {
            "fp16_key": "meta-llama_Llama-3.2-3B-Instruct",
            "int8_gen": P6_DIR / "llama32_inst_gen_output" / "emotion_vectors_v2.pt",
        },
        "Llama 3.2 3B Base": {
            "fp16_key": "meta-llama_Llama-3.2-3B",
            "int8_gen": P6_DIR / "llama32_base_gen_output" / "emotion_vectors_v2.pt",
        },
    }

    results = {}
    for name, paths in p6_models.items():
        print(f"\n  --- {name} ---")

        # Load fp16 (Paper #5)
        if paths["fp16_key"] not in all_data:
            print(f"    fp16 data not found")
            continue
        fp16_vecs = get_best_layer_vectors(all_data[paths["fp16_key"]])

        # Load INT8 (Paper #6 generation vectors)
        if not paths["int8_gen"].exists():
            print(f"    INT8 data not found at {paths['int8_gen']}")
            continue
        int8_pt = torch.load(paths["int8_gen"], map_location="cpu", weights_only=False)
        int8_vecs = {e: v.numpy() for e, v in int8_pt["vectors"].items()}

        # Common emotions (INT8 has 20, fp16 has 21)
        common = sorted(set(fp16_vecs.keys()) & set(int8_vecs.keys()))
        print(f"    Common emotions: {len(common)}")

        if len(common) < 10:
            print(f"    Too few common emotions, skipping")
            continue

        # 1) Per-emotion cosine similarity
        per_emotion_cos = {}
        for e in common:
            v1 = fp16_vecs[e] / (np.linalg.norm(fp16_vecs[e]) + 1e-8)
            v2 = int8_vecs[e] / (np.linalg.norm(int8_vecs[e]) + 1e-8)
            per_emotion_cos[e] = float(np.dot(v1, v2))
        mean_cos = np.mean(list(per_emotion_cos.values()))
        print(f"    Per-emotion cosine (fp16 vs INT8): mean={mean_cos:.4f}")
        print(f"    Range: [{min(per_emotion_cos.values()):.4f}, {max(per_emotion_cos.values()):.4f}]")

        # 2) RDM comparison
        fp16_sub = {e: fp16_vecs[e] for e in common}
        int8_sub = {e: int8_vecs[e] for e in common}
        rdm_fp16, _ = compute_rdm(fp16_sub)
        rdm_int8, _ = compute_rdm(int8_sub)
        rho, p = stats.spearmanr(upper_triangle(rdm_fp16), upper_triangle(rdm_int8))
        print(f"    RDM Spearman ρ: {rho:.4f} (p={p:.2e})")

        # 3) Anisotropy comparison
        fp16_meta = all_data[paths["fp16_key"]]["meta"]
        best_key = f"layer_{fp16_meta['best_layer']}_best"
        fp16_aniso = fp16_meta["anisotropy"].get(best_key, "N/A")
        # INT8 anisotropy: compute from vectors
        int8_mat = np.stack([int8_vecs[e] for e in common])
        int8_norms = np.linalg.norm(int8_mat, axis=1, keepdims=True)
        int8_norms[int8_norms == 0] = 1
        int8_normed = int8_mat / int8_norms
        int8_cos_all = int8_normed @ int8_normed.T
        int8_aniso = float(np.mean(int8_cos_all[np.triu_indices(len(common), k=1)]))
        print(f"    Anisotropy: fp16={fp16_aniso}, INT8(gen)={int8_aniso:.4f}")

        # 4) Best layer comparison (only available for fp16)
        print(f"    Best layer (fp16): {fp16_meta['best_layer']}/{fp16_meta['n_layers']}")
        print(f"    (INT8 extracted at single fixed layer — not comparable)")

        # 5) Steering regime
        regime = fp16_meta.get("steering_regime", "N/A")
        print(f"    Steering regime (fp16): {regime}")

        results[name] = {
            "per_emotion_cosine_mean": round(mean_cos, 4),
            "per_emotion_cosine_range": [
                round(min(per_emotion_cos.values()), 4),
                round(max(per_emotion_cos.values()), 4),
            ],
            "per_emotion_cosines": {k: round(v, 4) for k, v in per_emotion_cos.items()},
            "rdm_spearman_rho": round(rho, 4),
            "rdm_spearman_p": float(f"{p:.2e}"),
            "fp16_anisotropy": fp16_aniso if isinstance(fp16_aniso, (int, float)) else fp16_aniso,
            "int8_anisotropy_gen": round(int8_aniso, 4),
            "fp16_steering_regime": regime,
        }

    # Interpretation
    print("\n  Interpretation:")
    rdm_rhos = [r["rdm_spearman_rho"] for r in results.values()]
    if rdm_rhos:
        mean_rho = np.mean(rdm_rhos)
        if mean_rho > 0.95:
            print("    → Quantization preserves emotion geometry (ρ > 0.95)")
        elif mean_rho > 0.7:
            print(f"    → Partial preservation (mean ρ = {mean_rho:.3f})")
        else:
            print(f"    → Significant divergence (mean ρ = {mean_rho:.3f}) — precision artifact risk")
        print(f"    Note: Comparing fp16 COMPREHENSION vs INT8 GENERATION — method difference confound!")

    return results


# ── Task 3: Size effect analysis ─────────────────────────────────────────


def task3_size_effects(all_data: dict):
    print("\n" + "=" * 60)
    print("TASK 3: Size Effect Analysis")
    print("=" * 60)

    models = list(all_data.keys())
    sizes = [MODEL_SIZES[m] for m in models]
    d_models_list = [all_data[m]["meta"]["d_model"] for m in models]

    # Metrics per model
    anisotropies = []
    best_layers_pct = []
    mean_pairwise_stds = []
    for m in models:
        meta = all_data[m]["meta"]
        best_key = f"layer_{meta['best_layer']}_best"
        aniso = meta["anisotropy"].get(best_key, None)
        if aniso is None or (isinstance(aniso, float) and np.isnan(aniso)):
            # Try other keys
            for k, v in meta["anisotropy"].items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    aniso = v
                    break
        anisotropies.append(aniso if aniso is not None else np.nan)
        best_layers_pct.append(meta["best_layer"] / meta["n_layers"] * 100)
        # Mean pairwise std from RDM
        vecs = get_best_layer_vectors(all_data[m])
        rdm, _ = compute_rdm(vecs)
        mean_pairwise_stds.append(np.std(upper_triangle(rdm)))

    sizes = np.array(sizes)
    d_models_arr = np.array(d_models_list)
    anisotropies = np.array(anisotropies, dtype=float)
    best_layers_pct = np.array(best_layers_pct)
    mean_pairwise_stds = np.array(mean_pairwise_stds)
    labels = [SHORT_NAMES[m] for m in models]

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1) Anisotropy vs model size
    ax = axes[0, 0]
    valid = ~np.isnan(anisotropies)
    ax.scatter(sizes[valid], anisotropies[valid], c="steelblue", s=80, zorder=3)
    for i, m in enumerate(models):
        if valid[i]:
            ax.annotate(labels[i], (sizes[i], anisotropies[i]),
                        fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                        textcoords="offset points")
    if valid.sum() > 2:
        r, p = stats.spearmanr(sizes[valid], anisotropies[valid])
        ax.set_title(f"Anisotropy vs Size (ρ={r:.3f}, p={p:.3f})", fontsize=11)
    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel("Anisotropy (best layer)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 2) Best layer % vs model size
    ax = axes[0, 1]
    ax.scatter(sizes, best_layers_pct, c="coral", s=80, zorder=3)
    for i, m in enumerate(models):
        ax.annotate(labels[i], (sizes[i], best_layers_pct[i]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")
    r, p = stats.spearmanr(sizes, best_layers_pct)
    ax.set_title(f"Best Layer Depth vs Size (ρ={r:.3f}, p={p:.3f})", fontsize=11)
    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel("Best Layer (% depth)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 3) Pairwise std vs model size
    ax = axes[1, 0]
    ax.scatter(sizes, mean_pairwise_stds, c="forestgreen", s=80, zorder=3)
    for i, m in enumerate(models):
        ax.annotate(labels[i], (sizes[i], mean_pairwise_stds[i]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")
    r, p = stats.spearmanr(sizes, mean_pairwise_stds)
    ax.set_title(f"RDM Std vs Size (ρ={r:.3f}, p={p:.3f})", fontsize=11)
    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel("Mean Pairwise Std")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 4) Anisotropy vs d_model
    ax = axes[1, 1]
    ax.scatter(d_models_arr[valid], anisotropies[valid], c="purple", s=80, zorder=3)
    for i, m in enumerate(models):
        if valid[i]:
            ax.annotate(labels[i], (d_models_arr[i], anisotropies[i]),
                        fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                        textcoords="offset points")
    if valid.sum() > 2:
        r, p = stats.spearmanr(d_models_arr[valid], anisotropies[valid])
        ax.set_title(f"Anisotropy vs d_model (ρ={r:.3f}, p={p:.3f})", fontsize=11)
    ax.set_xlabel("d_model (hidden dim)")
    ax.set_ylabel("Anisotropy (best layer)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Task 3: Size Effect Analysis (n=12)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task3_size_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved task3_size_effects.png")

    # Print correlations
    print("\n  Size effect correlations:")
    for label, vals in [
        ("Aniso vs size(B)", (sizes[valid], anisotropies[valid])),
        ("Aniso vs d_model", (d_models_arr[valid], anisotropies[valid])),
        ("BestLayer% vs size", (sizes, best_layers_pct)),
        ("RDM_std vs size", (sizes, mean_pairwise_stds)),
    ]:
        r, p = stats.spearmanr(*vals)
        print(f"    {label:25s}: ρ={r:+.4f} (p={p:.4f})")

    return {
        "correlations": {
            "aniso_vs_size": {"rho": round(float(stats.spearmanr(sizes[valid], anisotropies[valid])[0]), 4),
                              "p": round(float(stats.spearmanr(sizes[valid], anisotropies[valid])[1]), 4)},
            "aniso_vs_dmodel": {"rho": round(float(stats.spearmanr(d_models_arr[valid], anisotropies[valid])[0]), 4),
                                "p": round(float(stats.spearmanr(d_models_arr[valid], anisotropies[valid])[1]), 4)},
            "best_layer_vs_size": {"rho": round(float(stats.spearmanr(sizes, best_layers_pct)[0]), 4),
                                   "p": round(float(stats.spearmanr(sizes, best_layers_pct)[1]), 4)},
            "rdm_std_vs_size": {"rho": round(float(stats.spearmanr(sizes, mean_pairwise_stds)[0]), 4),
                                "p": round(float(stats.spearmanr(sizes, mean_pairwise_stds)[1]), 4)},
        },
        "per_model": {
            SHORT_NAMES[m]: {
                "size_B": MODEL_SIZES[m],
                "d_model": all_data[m]["meta"]["d_model"],
                "anisotropy": round(float(anisotropies[i]), 4) if not np.isnan(anisotropies[i]) else None,
                "best_layer_pct": round(float(best_layers_pct[i]), 1),
                "rdm_std": round(float(mean_pairwise_stds[i]), 4),
            }
            for i, m in enumerate(models)
        },
    }


# ── Task 4: Tier 1 cluster re-verification ───────────────────────────────


def task4_tier1_recheck(all_data: dict, rdm_data: dict):
    print("\n" + "=" * 60)
    print("TASK 4: Tier 1 Cluster Re-Verification (n=12)")
    print("=" * 60)

    models = rdm_data["models"]
    rho_matrix = rdm_data["rho_matrix"]
    rdms = rdm_data["rdms"]

    # All instruct pairs
    inst = [m for m in models if m in INSTRUCT_MODELS]
    print(f"\n  Instruct models ({len(inst)}):")
    for m in inst:
        print(f"    {SHORT_NAMES[m]}")

    # All pairwise ρ for instruct models
    print(f"\n  All instruct pairwise ρ:")
    pair_data = []
    for i, m1 in enumerate(inst):
        for j, m2 in enumerate(inst):
            if j > i:
                idx1 = models.index(m1)
                idx2 = models.index(m2)
                rho = rho_matrix[idx1, idx2]
                pair_data.append((SHORT_NAMES[m1], SHORT_NAMES[m2], rho))
                print(f"    {SHORT_NAMES[m1]:16s} × {SHORT_NAMES[m2]:16s} : ρ={rho:.4f}")

    # Special pairs
    print("\n  Key pairs:")
    key_pairs = [
        ("meta-llama_Llama-3.1-8B-Instruct", "meta-llama_Llama-3.2-3B-Instruct",
         "Llama family (3B vs 8B)"),
        ("mistralai_Mistral-7B-Instruct-v0.3", "meta-llama_Llama-3.1-8B-Instruct",
         "Similar size, diff arch"),
        ("mistralai_Mistral-7B-Instruct-v0.3", "Qwen_Qwen2.5-1.5B-Instruct",
         "Mistral vs Qwen"),
    ]
    for m1, m2, desc in key_pairs:
        if m1 in models and m2 in models:
            idx1, idx2 = models.index(m1), models.index(m2)
            rho = rho_matrix[idx1, idx2]
            print(f"    {desc}: {SHORT_NAMES[m1]} × {SHORT_NAMES[m2]} = ρ={rho:.4f}")

    # Tier classification
    print("\n  Tier classification (threshold ρ > 0.7 for Tier 1):")
    rhos = np.array([p[2] for p in pair_data])
    tier1_pairs = [(a, b, r) for a, b, r in pair_data if r > 0.7]
    print(f"    Tier 1 pairs (ρ > 0.7): {len(tier1_pairs)}/{len(pair_data)}")
    for a, b, r in sorted(tier1_pairs, key=lambda x: -x[2]):
        print(f"      {a} × {b}: {r:.4f}")

    # Box plot of all instruct pairs
    fig, ax = plt.subplots(figsize=(10, 5))
    # Group by: within-Tier1-original, new-pairs-with-7B+, gemma-pairs
    original_tier1 = {"Qwen1.5B-I", "SmolLM1.7B-I", "Llama3.2-3B-I"}
    new_7b = {"Mistral7B-I", "Llama3.1-8B-I"}
    gemma = {"Gemma1B-IT"}

    groups = {"Original Tier 1": [], "New 7B+ pairs": [], "Gemma pairs": [], "Cross-group": []}
    for a, b, r in pair_data:
        if a in original_tier1 and b in original_tier1:
            groups["Original Tier 1"].append(r)
        elif a in gemma or b in gemma:
            groups["Gemma pairs"].append(r)
        elif a in new_7b or b in new_7b:
            groups["New 7B+ pairs"].append(r)
        else:
            groups["Cross-group"].append(r)

    group_names = [k for k in groups if groups[k]]
    group_vals = [groups[k] for k in group_names]
    bp = ax.boxplot(group_vals, labels=group_names, patch_artist=True)
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]
    for patch, color in zip(bp["boxes"], colors[:len(group_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Overlay individual points
    for i, vals in enumerate(group_vals):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.7, s=30, zorder=3)

    ax.set_ylabel("Spearman ρ")
    ax.set_title("Tier 1 Cluster Re-Verification: Instruct Pair ρ Distribution", fontweight="bold")
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="Tier 1 threshold (ρ=0.7)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task4_tier1_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved task4_tier1_boxplot.png")

    # Lead finding update
    print("\n  Lead finding update:")
    all_rhos = [p[2] for p in pair_data]
    # Exclude gemma outlier for Tier 1 check
    non_gemma_rhos = [r for a, b, r in pair_data if "Gemma" not in a and "Gemma" not in b]
    if non_gemma_rhos:
        mean_ng = np.mean(non_gemma_rhos)
        min_ng = np.min(non_gemma_rhos)
        print(f"    Non-Gemma instruct pairs: mean ρ = {mean_ng:.4f}, min ρ = {min_ng:.4f}")
        if min_ng > 0.7:
            n_arch = 5  # Qwen, SmolLM, Llama3.2, Llama3.1, Mistral
            print(f"    → '{n_arch} architecturally distinct models spanning 1.5B-8B'")
            print(f"       share emotion geometry (ρ={min_ng:.2f}-{max(non_gemma_rhos):.2f})")
        elif min_ng > 0.5:
            print(f"    → Partial universality (min ρ = {min_ng:.4f})")
        else:
            print(f"    → Weak universality (min ρ = {min_ng:.4f})")

    return {
        "all_instruct_pairs": [
            {"model_a": a, "model_b": b, "spearman_rho": round(r, 4)}
            for a, b, r in pair_data
        ],
        "group_stats": {
            k: {"mean": round(np.mean(v), 4), "min": round(np.min(v), 4),
                "max": round(np.max(v), 4), "n": len(v)}
            for k, v in groups.items() if v
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Paper #5 — n=12 Comprehensive Analysis")
    print("=" * 60)

    # Load all models
    print("\nLoading all 12 models...")
    all_data = {}
    for m in ALL_MODELS:
        data = load_model_data(m)
        if data:
            all_data[m] = data
            print(f"  {SHORT_NAMES[m]:20s} layers={data['meta']['n_layers']}, "
                  f"d={data['meta']['d_model']}, best={data['meta']['best_layer']}")
    print(f"\nLoaded: {len(all_data)}/{len(ALL_MODELS)} models")

    results = {}

    # Task 1
    rdm_data = task1_rdm_of_rdms(all_data)
    results["task1_rdm_of_rdms"] = {
        "rho_matrix": rdm_data["rho_matrix"].tolist(),
        "models": [SHORT_NAMES[m] for m in rdm_data["models"]],
    }

    # Task 2
    results["task2_precision_confound"] = task2_precision_confound(all_data)

    # Task 3
    results["task3_size_effects"] = task3_size_effects(all_data)

    # Task 4
    results["task4_tier1_recheck"] = task4_tier1_recheck(all_data, rdm_data)

    # Save all results
    results_file = OUTPUT_DIR / "n12_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nAll results saved to {results_file}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
