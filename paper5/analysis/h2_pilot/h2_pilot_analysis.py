"""Paper #5 — H2 Pilot Analysis: Qwen 2.5 1.5B vs Llama 3.2 3B

Discriminant validity test: Do emotion vector structures differ between
models with opposite Compliance facet patterns?

- Qwen: yields to opinion pressure (B=0.80), refuses format instructions (D=0.61)
- Llama: refuses opinion pressure (B=0.20), performs format instructions (D=0.85)

Outputs:
  - h2_pilot_results.json    (all metrics)
  - rdm_comparison.png       (side-by-side RDM heatmaps)
  - rdm_diff.png             (RDM difference plot)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "paper5_output"


def load_vectors(model_dir: str, mode: str = "comprehension") -> dict:
    """Load vectors and metadata for a model."""
    d = DATA_DIR / model_dir
    meta = json.loads((d / "metadata.json").read_text())
    pt = torch.load(d / f"vectors_{mode}.pt", map_location="cpu", weights_only=False)
    return {"meta": meta, "pt": pt}


def get_best_layer_vectors(data: dict) -> dict[str, np.ndarray]:
    """Extract emotion vectors at the best layer as numpy arrays."""
    best = str(data["meta"]["best_layer"])
    vectors = data["pt"]["vectors"][best]
    return {e: v.numpy() for e, v in vectors.items()}


def compute_rdm(vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute representational dissimilarity matrix (cosine-based)."""
    emotions = sorted(vectors.keys())
    n = len(emotions)
    mat = np.stack([vectors[e] for e in emotions])
    # Normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_norm = mat / norms
    # Cosine similarity → dissimilarity
    cos_sim = mat_norm @ mat_norm.T
    rdm = 1 - cos_sim
    return rdm, emotions


def anisotropy_normalize(rdm: np.ndarray, anisotropy: float) -> np.ndarray:
    """Z-score normalize RDM by anisotropy baseline."""
    # Anisotropy = mean pairwise cosine of neutral sentences
    # Expected dissimilarity for random directions = 1 - anisotropy
    baseline = 1 - anisotropy
    if baseline > 0:
        return (rdm - baseline) / max(np.std(rdm), 1e-8)
    return rdm


def upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Extract upper triangle values (excluding diagonal)."""
    n = mat.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return mat[mask]


def top_diff_pairs(rdm1: np.ndarray, rdm2: np.ndarray, emotions: list[str], top_k: int = 5):
    """Find emotion pairs with largest RDM difference."""
    diff = rdm2 - rdm1
    n = len(emotions)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((emotions[i], emotions[j], diff[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs[:top_k]


def plot_rdm_comparison(rdm_q, rdm_l, emotions, filename):
    """Plot side-by-side RDM heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    vmin = min(rdm_q.min(), rdm_l.min())
    vmax = max(rdm_q.max(), rdm_l.max())

    im1 = ax1.imshow(rdm_q, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax1.set_xticks(range(len(emotions)))
    ax1.set_yticks(range(len(emotions)))
    ax1.set_xticklabels(emotions, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(emotions, fontsize=8)
    ax1.set_title("Qwen 2.5 1.5B Instruct\n(Compliance-yielding)", fontsize=11)

    im2 = ax2.imshow(rdm_l, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax2.set_xticks(range(len(emotions)))
    ax2.set_yticks(range(len(emotions)))
    ax2.set_xticklabels(emotions, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(emotions, fontsize=8)
    ax2.set_title("Llama 3.2 3B Instruct\n(Compliance-refusing)", fontsize=11)

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, label="Dissimilarity (anisotropy-normalized)")
    plt.suptitle("H2 Pilot: Emotion RDM Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_rdm_diff(diff, emotions, top_pairs, filename):
    """Plot RDM difference matrix with annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    vabs = max(abs(diff.min()), abs(diff.max()))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(len(emotions)))
    ax.set_yticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(emotions, fontsize=8)

    # Annotate top pairs
    for e1, e2, val in top_pairs:
        i, j = emotions.index(e1), emotions.index(e2)
        ax.plot(j, i, "ko", markersize=8, fillstyle="none", markeredgewidth=2)
        ax.plot(i, j, "ko", markersize=8, fillstyle="none", markeredgewidth=2)

    fig.colorbar(im, ax=ax, shrink=0.7, label="Llama − Qwen (dissimilarity diff)")
    ax.set_title("RDM Difference (Llama − Qwen)\nCircled = top 5 differences", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def main():
    print("=" * 60)
    print("Paper #5 — H2 Pilot: Qwen vs Llama Emotion Vector Comparison")
    print("=" * 60)

    # ── Load data ──
    print("\n1. Loading data...")
    qwen_inst = load_vectors("Qwen_Qwen2.5-1.5B-Instruct")
    qwen_base = load_vectors("Qwen_Qwen2.5-1.5B")
    llama_inst = load_vectors("meta-llama_Llama-3.2-3B-Instruct")
    llama_base = load_vectors("meta-llama_Llama-3.2-3B")

    print(f"  Qwen Instruct: best_layer={qwen_inst['meta']['best_layer']}, "
          f"anisotropy={qwen_inst['meta']['anisotropy']}")
    print(f"  Llama Instruct: best_layer={llama_inst['meta']['best_layer']}, "
          f"anisotropy={llama_inst['meta']['anisotropy']}")

    # ── Compute RDMs (comprehension, instruct) ──
    print("\n2. Computing RDMs (comprehension, instruct)...")
    qwen_vecs = get_best_layer_vectors(qwen_inst)
    llama_vecs = get_best_layer_vectors(llama_inst)

    rdm_q_raw, emotions = compute_rdm(qwen_vecs)
    rdm_l_raw, _ = compute_rdm(llama_vecs)

    # Anisotropy normalization
    best_key_q = f"layer_{qwen_inst['meta']['best_layer']}_best"
    best_key_l = f"layer_{llama_inst['meta']['best_layer']}_best"
    ani_q = qwen_inst["meta"]["anisotropy"].get(best_key_q, 0.84)
    ani_l = llama_inst["meta"]["anisotropy"].get(best_key_l, 0.63)

    rdm_q = anisotropy_normalize(rdm_q_raw, ani_q)
    rdm_l = anisotropy_normalize(rdm_l_raw, ani_l)

    print(f"  Qwen RDM: mean={rdm_q_raw.mean():.4f}, anisotropy={ani_q:.4f}")
    print(f"  Llama RDM: mean={rdm_l_raw.mean():.4f}, anisotropy={ani_l:.4f}")

    # ── RDM comparison: Spearman correlation ──
    print("\n3. RDM comparison...")
    ut_q = upper_triangle(rdm_q)
    ut_l = upper_triangle(rdm_l)
    spearman_r, spearman_p = stats.spearmanr(ut_q, ut_l)
    pearson_r, pearson_p = stats.pearsonr(ut_q, ut_l)
    print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"  Pearson r  = {pearson_r:.4f} (p = {pearson_p:.2e})")

    # Top 5 difference pairs
    diff = rdm_l - rdm_q  # Llama minus Qwen
    top_pairs = top_diff_pairs(rdm_q, rdm_l, emotions, top_k=5)
    print("\n  Top 5 RDM differences (Llama − Qwen):")
    for e1, e2, val in top_pairs:
        print(f"    {e1:15s} × {e2:15s}  Δ = {val:+.4f}")

    # ── Base vs Instruct (RLHF effect) ──
    print("\n4. Base vs Instruct (RLHF effect)...")
    qwen_base_vecs = get_best_layer_vectors(qwen_base)
    llama_base_vecs = get_best_layer_vectors(llama_base)

    rdm_q_base_raw, _ = compute_rdm(qwen_base_vecs)
    rdm_l_base_raw, _ = compute_rdm(llama_base_vecs)

    # RLHF effect = instruct RDM - base RDM
    rlhf_q = rdm_q_raw - rdm_q_base_raw
    rlhf_l = rdm_l_raw - rdm_l_base_raw

    rlhf_corr, rlhf_p = stats.spearmanr(
        upper_triangle(rlhf_q), upper_triangle(rlhf_l)
    )
    print(f"  RLHF effect correlation (Qwen vs Llama): ρ = {rlhf_corr:.4f} (p = {rlhf_p:.2e})")
    print(f"  Qwen RLHF magnitude: mean|Δ| = {np.abs(rlhf_q).mean():.4f}")
    print(f"  Llama RLHF magnitude: mean|Δ| = {np.abs(rlhf_l).mean():.4f}")

    # ── Generation vs Comprehension ──
    print("\n5. Generation vs Comprehension (Shell-Core test)...")
    qwen_gen = load_vectors("Qwen_Qwen2.5-1.5B-Instruct", mode="generation")
    llama_gen = load_vectors("meta-llama_Llama-3.2-3B-Instruct", mode="generation")

    qwen_gen_vecs = {e: v.numpy() for e, v in qwen_gen["pt"]["vectors"].items()}
    llama_gen_vecs = {e: v.numpy() for e, v in llama_gen["pt"]["vectors"].items()}

    # Only compare emotions present in both gen and comp (gen has 20, no neutral)
    common_emotions = sorted(set(qwen_gen_vecs.keys()) & set(qwen_vecs.keys()))

    rdm_q_gen, _ = compute_rdm({e: qwen_gen_vecs[e] for e in common_emotions})
    rdm_l_gen, _ = compute_rdm({e: llama_gen_vecs[e] for e in common_emotions})
    rdm_q_comp_sub, _ = compute_rdm({e: qwen_vecs[e] for e in common_emotions})
    rdm_l_comp_sub, _ = compute_rdm({e: llama_vecs[e] for e in common_emotions})

    # Cross-model gen correlation
    gen_corr, gen_p = stats.spearmanr(
        upper_triangle(rdm_q_gen), upper_triangle(rdm_l_gen)
    )
    # Within-model gen-comp correlation
    qwen_gc, qwen_gc_p = stats.spearmanr(
        upper_triangle(rdm_q_gen), upper_triangle(rdm_q_comp_sub)
    )
    llama_gc, llama_gc_p = stats.spearmanr(
        upper_triangle(rdm_l_gen), upper_triangle(rdm_l_comp_sub)
    )

    print(f"  Cross-model (gen): ρ = {gen_corr:.4f} (p = {gen_p:.2e})")
    print(f"  Qwen gen-comp:     ρ = {qwen_gc:.4f} (p = {qwen_gc_p:.2e})")
    print(f"  Llama gen-comp:    ρ = {llama_gc:.4f} (p = {llama_gc_p:.2e})")

    shell_core_verdict = (
        "Persona DOES penetrate Core" if min(qwen_gc, llama_gc) > 0.7
        else "Persona appears Shell-only (gen ≠ comp)"
        if max(qwen_gc, llama_gc) < 0.5
        else "Mixed — persona partially penetrates"
    )
    print(f"  Shell-Core verdict: {shell_core_verdict}")

    # ── Plot figures ──
    print("\n6. Generating figures...")
    plot_rdm_comparison(rdm_q, rdm_l, emotions, OUTPUT_DIR / "rdm_comparison.png")
    plot_rdm_diff(diff, emotions, top_pairs, OUTPUT_DIR / "rdm_diff.png")

    # ── Save results ──
    results = {
        "analysis": "H2 Pilot — Qwen vs Llama Emotion Vector Comparison",
        "models": {
            "qwen_instruct": {
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "best_layer": qwen_inst["meta"]["best_layer"],
                "anisotropy": ani_q,
                "mti_compliance": {"B": 0.80, "D": 0.61, "pattern": "yielding"},
            },
            "llama_instruct": {
                "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                "best_layer": llama_inst["meta"]["best_layer"],
                "anisotropy": ani_l,
                "mti_compliance": {"B": 0.20, "D": 0.85, "pattern": "refusing"},
            },
        },
        "rdm_comparison": {
            "spearman_rho": round(spearman_r, 4),
            "spearman_p": float(f"{spearman_p:.2e}"),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": float(f"{pearson_p:.2e}"),
            "interpretation": (
                "HIGH correlation → similar emotion geometry despite opposite behavior"
                if spearman_r > 0.7
                else "LOW correlation → different emotion geometry matches different behavior"
                if spearman_r < 0.3
                else "MODERATE correlation → partial structural difference"
            ),
        },
        "top_5_rdm_differences": [
            {"emotion1": e1, "emotion2": e2, "diff": round(d, 4)}
            for e1, e2, d in top_pairs
        ],
        "rlhf_effect": {
            "qwen_llama_correlation": round(rlhf_corr, 4),
            "qwen_magnitude": round(float(np.abs(rlhf_q).mean()), 4),
            "llama_magnitude": round(float(np.abs(rlhf_l).mean()), 4),
        },
        "generation_vs_comprehension": {
            "cross_model_gen_rho": round(gen_corr, 4),
            "qwen_gen_comp_rho": round(qwen_gc, 4),
            "llama_gen_comp_rho": round(llama_gc, 4),
            "shell_core_verdict": shell_core_verdict,
        },
    }

    results_file = OUTPUT_DIR / "h2_pilot_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"  Saved {results_file}")

    # ── Best layer position comparison (Luca addition) ──
    print("\n7. Best layer position comparison...")
    qwen_bl = qwen_inst["meta"]["best_layer"]
    llama_bl = llama_inst["meta"]["best_layer"]
    qwen_nl = qwen_inst["meta"]["n_layers"]
    llama_nl = llama_inst["meta"]["n_layers"]
    qwen_rel = qwen_bl / qwen_nl
    llama_rel = llama_bl / llama_nl
    print(f"  Qwen:  layer {qwen_bl}/{qwen_nl} = {qwen_rel:.1%} depth")
    print(f"  Llama: layer {llama_bl}/{llama_nl} = {llama_rel:.1%} depth")
    print(f"  Δ = {abs(qwen_rel - llama_rel):.1%} — {'DIFFERENT' if abs(qwen_rel - llama_rel) > 0.1 else 'SIMILAR'} optimal depth")

    results["best_layer_comparison"] = {
        "qwen": {"layer": qwen_bl, "total": qwen_nl, "relative": round(qwen_rel, 3)},
        "llama": {"layer": llama_bl, "total": llama_nl, "relative": round(llama_rel, 3)},
        "delta_relative": round(abs(qwen_rel - llama_rel), 3),
    }

    # ── Summary ──
    print("\n" + "=" * 60)
    print("H2 PILOT SUMMARY")
    print("=" * 60)
    print(f"  RDM Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"  → {results['rdm_comparison']['interpretation']}")
    print(f"  RLHF effect correlation: ρ = {rlhf_corr:.4f}")
    print(f"  Shell-Core: {shell_core_verdict}")
    print(f"  Top RDM diff: {top_pairs[0][0]} × {top_pairs[0][1]} (Δ={top_pairs[0][2]:+.4f})")


if __name__ == "__main__":
    main()
