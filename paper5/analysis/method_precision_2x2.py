"""Paper #5 Finding #8 — Method × Precision 2×2 Design

Decomposes the conflated comparison into two clean analyses:
  Analysis A: fp16 comp vs fp16 gen   (method effect, precision controlled)
  Analysis B: fp16 gen  vs INT8 gen   (precision effect, method controlled)

Combined with the original (mis-)comparison for context.

Models: Mistral 7B Instruct, Llama 3.1 8B Instruct
Data:
  - fp16 comprehension + generation: scripts/paper5_output/
  - INT8 generation:                 ludex/research/emotion_benchmark/

Usage:
    cd backend && uv run python ../paper5/analysis/method_precision_2x2.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

OUTPUT_DIR = Path(__file__).parent / "method_precision_output"
OUTPUT_DIR.mkdir(exist_ok=True)

P5_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "paper5_output"
P6_DIR = Path("/Users/jihoon/Projects/ludex/research/emotion_benchmark")

MODELS = {
    "Mistral 7B Instruct": {
        "p5_key": "mistralai_Mistral-7B-Instruct-v0.3",
        "p6_gen": P6_DIR / "emotion_pilot_v2_output" / "emotion_vectors_v2.pt",
    },
    "Llama 3.1 8B Instruct": {
        "p5_key": "meta-llama_Llama-3.1-8B-Instruct",
        "p6_gen": P6_DIR / "llama31_inst_gen_output" / "emotion_vectors_v2.pt",
    },
}


# ── Loading helpers ──────────────────────────────────────────────────────


def load_p5_comp(model_dir: str) -> tuple[dict[str, np.ndarray], dict]:
    """fp16 comprehension vectors at best layer."""
    d = P5_DIR / model_dir
    meta = json.loads((d / "metadata.json").read_text())
    pt = torch.load(d / "vectors_comprehension.pt", map_location="cpu", weights_only=False)
    best = str(meta["best_layer"])
    vecs = {e: v.numpy() for e, v in pt["vectors"][best].items()}
    return vecs, meta


def load_p5_gen(model_dir: str) -> dict[str, np.ndarray]:
    """fp16 generation vectors (extracted at best layer = same as comp)."""
    d = P5_DIR / model_dir
    pt = torch.load(d / "vectors_generation.pt", map_location="cpu", weights_only=False)
    # vectors_generation.pt structure: {model_id, layer, vectors: {emotion: tensor}}
    return {e: v.numpy() for e, v in pt["vectors"].items()}


def load_p6_gen(path: Path) -> dict[str, np.ndarray]:
    """INT8 generation vectors from Paper #6."""
    pt = torch.load(path, map_location="cpu", weights_only=False)
    return {e: v.numpy() for e, v in pt["vectors"].items()}


# ── Analysis primitives ──────────────────────────────────────────────────


def compute_rdm(vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    emos = sorted(vectors.keys())
    mat = np.stack([vectors[e] for e in emos])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_n = mat / norms
    cos_sim = mat_n @ mat_n.T
    return 1 - cos_sim, emos


def upper_tri(m):
    return m[np.triu(np.ones(m.shape, dtype=bool), k=1)]


def post_center(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Post-hoc mean centering: subtract the mean of the vector set.

    Equalizes the 'baseline' confound between Paper #5 (subtracts emotion-global-mean)
    and Paper #6 (subtracts neutral-mean). After this, both sets are centered
    at the same origin within each vector set.
    """
    keys = sorted(vectors.keys())
    mat = np.stack([vectors[k] for k in keys])
    mean = mat.mean(axis=0)
    return {k: vectors[k] - mean for k in keys}


def compare_vector_sets(
    set_a: dict[str, np.ndarray],
    set_b: dict[str, np.ndarray],
    label_a: str,
    label_b: str,
    apply_centering: bool = True,
) -> dict:
    """Compute per-emotion cosine + RDM Spearman + anisotropy.

    apply_centering: if True, post-hoc center both sets to common baseline.
    """
    if apply_centering:
        set_a = post_center(set_a)
        set_b = post_center(set_b)
    common = sorted(set(set_a.keys()) & set(set_b.keys()))

    # Per-emotion cosine
    per_emotion_cos = {}
    for e in common:
        v1 = set_a[e] / (np.linalg.norm(set_a[e]) + 1e-8)
        v2 = set_b[e] / (np.linalg.norm(set_b[e]) + 1e-8)
        per_emotion_cos[e] = float(np.dot(v1, v2))
    mean_cos = float(np.mean(list(per_emotion_cos.values())))
    median_cos = float(np.median(list(per_emotion_cos.values())))

    # RDM Spearman
    a_sub = {e: set_a[e] for e in common}
    b_sub = {e: set_b[e] for e in common}
    rdm_a, _ = compute_rdm(a_sub)
    rdm_b, _ = compute_rdm(b_sub)
    rho, p = stats.spearmanr(upper_tri(rdm_a), upper_tri(rdm_b))

    # Internal pairwise cosine (anisotropy proxy from these vectors)
    def internal_aniso(vecs):
        mat = np.stack([vecs[e] for e in common])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        cos = (mat / norms) @ (mat / norms).T
        return float(np.mean(cos[np.triu_indices(len(common), k=1)]))

    aniso_a = internal_aniso(set_a)
    aniso_b = internal_aniso(set_b)

    return {
        "n_common": len(common),
        "common_emotions": common,
        "per_emotion_cosine": {
            "mean": round(mean_cos, 4),
            "median": round(median_cos, 4),
            "min": round(min(per_emotion_cos.values()), 4),
            "max": round(max(per_emotion_cos.values()), 4),
            "values": {k: round(v, 4) for k, v in per_emotion_cos.items()},
        },
        "rdm_spearman": {
            "rho": round(float(rho), 4),
            "p": float(f"{p:.2e}"),
        },
        f"internal_anisotropy_{label_a}": round(aniso_a, 4),
        f"internal_anisotropy_{label_b}": round(aniso_b, 4),
        "rdm_a": rdm_a,
        "rdm_b": rdm_b,
    }


# ── Main analysis ────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Paper #5 Finding #8 — Method × Precision 2×2 Design")
    print("=" * 70)

    all_results = {}
    summary_rows = []

    for name, paths in MODELS.items():
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")

        # Load 3 vector sets (the 3 filled cells of the 2x2)
        p5_comp, meta = load_p5_comp(paths["p5_key"])
        p5_gen = load_p5_gen(paths["p5_key"])
        p6_gen = load_p6_gen(paths["p6_gen"])

        print(f"  fp16 comp: {len(p5_comp)} emotions (best layer {meta['best_layer']})")
        print(f"  fp16 gen:  {len(p5_gen)} emotions")
        print(f"  INT8 gen:  {len(p6_gen)} emotions")

        # ── Analysis A: fp16 comp vs fp16 gen (METHOD effect) ──
        print("\n  [A] fp16 comp vs fp16 gen — METHOD effect (precision controlled)")
        a = compare_vector_sets(p5_comp, p5_gen, "fp16_comp", "fp16_gen")
        print(f"      n_common = {a['n_common']}")
        print(f"      Per-emotion cos: mean={a['per_emotion_cosine']['mean']:.4f} "
              f"(range [{a['per_emotion_cosine']['min']:.4f}, {a['per_emotion_cosine']['max']:.4f}])")
        print(f"      RDM Spearman ρ = {a['rdm_spearman']['rho']:.4f} (p={a['rdm_spearman']['p']:.2e})")
        print(f"      Internal aniso: comp={a['internal_anisotropy_fp16_comp']:.4f}, "
              f"gen={a['internal_anisotropy_fp16_gen']:.4f}")

        # ── Analysis B: fp16 gen vs INT8 gen (PRECISION effect) ──
        print("\n  [B] fp16 gen vs INT8 gen — PRECISION effect (method controlled)")
        b = compare_vector_sets(p5_gen, p6_gen, "fp16_gen", "int8_gen")
        print(f"      n_common = {b['n_common']}")
        print(f"      Per-emotion cos: mean={b['per_emotion_cosine']['mean']:.4f} "
              f"(range [{b['per_emotion_cosine']['min']:.4f}, {b['per_emotion_cosine']['max']:.4f}])")
        print(f"      RDM Spearman ρ = {b['rdm_spearman']['rho']:.4f} (p={b['rdm_spearman']['p']:.2e})")
        print(f"      Internal aniso: fp16gen={b['internal_anisotropy_fp16_gen']:.4f}, "
              f"int8gen={b['internal_anisotropy_int8_gen']:.4f}")

        # ── Original (conflated) comparison: fp16 comp vs INT8 gen ──
        print("\n  [C] fp16 comp vs INT8 gen — CONFLATED (for reference)")
        c = compare_vector_sets(p5_comp, p6_gen, "fp16_comp", "int8_gen")
        print(f"      RDM Spearman ρ = {c['rdm_spearman']['rho']:.4f} (p={c['rdm_spearman']['p']:.2e})")
        print(f"      → This is what we wrongly compared in Task 2")

        # ── 2×2 interaction check ──
        # If method effect and precision effect are independent,
        # ρ(comp, INT8gen) ≈ ρ(comp, fp16gen) × ρ(fp16gen, INT8gen) (multiplicative)
        # Or: residual variance is additive
        method_rho = a["rdm_spearman"]["rho"]
        precision_rho = b["rdm_spearman"]["rho"]
        conflated_rho = c["rdm_spearman"]["rho"]
        predicted_rho = method_rho * precision_rho
        interaction_residual = conflated_rho - predicted_rho

        print(f"\n  [Interaction] ρ_method × ρ_precision = "
              f"{method_rho:.3f} × {precision_rho:.3f} = {predicted_rho:.3f}")
        print(f"                ρ_conflated_observed = {conflated_rho:.3f}")
        print(f"                Residual = {interaction_residual:+.3f}")
        if abs(interaction_residual) < 0.1:
            print(f"                → Effects appear INDEPENDENT (additive)")
        else:
            print(f"                → Possible interaction (non-additive)")

        all_results[name] = {
            "analysis_A_method_effect": {
                "label": "fp16 comp vs fp16 gen (precision controlled)",
                "n_common": a["n_common"],
                "per_emotion_cosine": a["per_emotion_cosine"],
                "rdm_spearman": a["rdm_spearman"],
                "internal_anisotropy_fp16_comp": a["internal_anisotropy_fp16_comp"],
                "internal_anisotropy_fp16_gen": a["internal_anisotropy_fp16_gen"],
            },
            "analysis_B_precision_effect": {
                "label": "fp16 gen vs INT8 gen (method controlled)",
                "n_common": b["n_common"],
                "per_emotion_cosine": b["per_emotion_cosine"],
                "rdm_spearman": b["rdm_spearman"],
                "internal_anisotropy_fp16_gen": b["internal_anisotropy_fp16_gen"],
                "internal_anisotropy_int8_gen": b["internal_anisotropy_int8_gen"],
            },
            "analysis_C_conflated_reference": {
                "label": "fp16 comp vs INT8 gen (Task 2 — conflated)",
                "rdm_spearman": c["rdm_spearman"],
            },
            "interaction": {
                "method_rho": round(method_rho, 4),
                "precision_rho": round(precision_rho, 4),
                "conflated_observed_rho": round(conflated_rho, 4),
                "predicted_multiplicative": round(predicted_rho, 4),
                "residual": round(interaction_residual, 4),
                "verdict": "independent" if abs(interaction_residual) < 0.1 else "interaction",
            },
        }

        summary_rows.append({
            "model": name,
            "method_rho": method_rho,
            "precision_rho": precision_rho,
            "conflated_rho": conflated_rho,
        })

        # Store rdms for plotting
        all_results[name]["_rdms"] = {
            "fp16_comp_vs_gen": (a["rdm_a"], a["rdm_b"], a["common_emotions"]),
            "fp16_gen_vs_int8_gen": (b["rdm_a"], b["rdm_b"], b["common_emotions"]),
        }

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY — 2×2 Decomposition")
    print("=" * 70)
    print(f"\n  {'Model':<25s} {'Method ρ':>12s} {'Precision ρ':>14s} {'Conflated ρ':>14s}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 14} {'-' * 14}")
    for r in summary_rows:
        print(f"  {r['model']:<25s} {r['method_rho']:>12.4f} {r['precision_rho']:>14.4f} {r['conflated_rho']:>14.4f}")

    # ── Interpretation ──
    method_rhos = [r["method_rho"] for r in summary_rows]
    precision_rhos = [r["precision_rho"] for r in summary_rows]
    print(f"\n  Mean method ρ:    {np.mean(method_rhos):.4f}")
    print(f"  Mean precision ρ: {np.mean(precision_rhos):.4f}")

    print("\n  Interpretation:")
    if np.mean(method_rhos) < 0.3:
        print("    METHOD effect: STRONG dissociation")
        print("      → Comprehension and generation produce nearly orthogonal emotion representations")
        print("      → Replicates Paper #6 finding in Mistral/Llama 3.1 (n=2 confirmation)")
    elif np.mean(method_rhos) < 0.7:
        print("    METHOD effect: MODERATE dissociation")
    else:
        print("    METHOD effect: MILD")

    if np.mean(precision_rhos) > 0.9:
        print("    PRECISION effect: NEGLIGIBLE (ρ > 0.9)")
        print("      → INT8 quantization preserves emotion geometry within method")
        print("      → Paper #6 4-bit results compatible with fp16 baseline (when method matches)")
    elif np.mean(precision_rhos) > 0.7:
        print("    PRECISION effect: MILD (0.7 < ρ < 0.9)")
        print("      → Quantization preserves coarse structure but fine details differ")
    else:
        print(f"    PRECISION effect: SIGNIFICANT (ρ = {np.mean(precision_rhos):.3f})")
        print("      → Quantization measurably alters emotion vector geometry")

    # ── Plot 2×2 visualization ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (name, _) in enumerate(MODELS.items()):
        rdms = all_results[name]["_rdms"]

        # Column 0: fp16 comp RDM
        rdm_comp, rdm_gen, emos = rdms["fp16_comp_vs_gen"]
        n = len(emos)
        ax = axes[row, 0]
        im = ax.imshow(rdm_comp, cmap="RdBu_r", vmin=0, vmax=2)
        ax.set_title(f"{name}\nfp16 COMPREHENSION RDM", fontsize=10)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(emos, rotation=90, fontsize=6)
        ax.set_yticklabels(emos, fontsize=6)

        # Column 1: fp16 gen RDM
        ax = axes[row, 1]
        ax.imshow(rdm_gen, cmap="RdBu_r", vmin=0, vmax=2)
        method_rho = all_results[name]["analysis_A_method_effect"]["rdm_spearman"]["rho"]
        ax.set_title(f"fp16 GENERATION RDM\n(vs comp: ρ={method_rho:.3f})", fontsize=10)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(emos, rotation=90, fontsize=6)
        ax.set_yticklabels(emos, fontsize=6)

        # Column 2: INT8 gen RDM
        rdm_fpgen, rdm_intgen, emos2 = rdms["fp16_gen_vs_int8_gen"]
        ax = axes[row, 2]
        ax.imshow(rdm_intgen, cmap="RdBu_r", vmin=0, vmax=2)
        precision_rho = all_results[name]["analysis_B_precision_effect"]["rdm_spearman"]["rho"]
        ax.set_title(f"INT8 GENERATION RDM (Paper #6)\n(vs fp16 gen: ρ={precision_rho:.3f})", fontsize=10)
        ax.set_xticks(range(len(emos2)))
        ax.set_yticks(range(len(emos2)))
        ax.set_xticklabels(emos2, rotation=90, fontsize=6)
        ax.set_yticklabels(emos2, fontsize=6)

    plt.suptitle("Method × Precision 2×2 Decomposition: RDM Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_precision_rdms.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved method_precision_rdms.png")

    # ── Bar chart summary ──
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary_rows))
    width = 0.27
    ax.bar(x - width, [r["method_rho"] for r in summary_rows], width,
           label="Method (comp vs gen, fp16)", color="#3498db")
    ax.bar(x, [r["precision_rho"] for r in summary_rows], width,
           label="Precision (fp16 vs INT8, gen)", color="#2ecc71")
    ax.bar(x + width, [r["conflated_rho"] for r in summary_rows], width,
           label="Conflated (Task 2 — comp vs INT8 gen)", color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in summary_rows], rotation=15, ha="right")
    ax.set_ylabel("RDM Spearman ρ")
    ax.set_title("Finding #8: Method vs Precision Effects", fontweight="bold")
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="ρ=0.7 threshold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim([-0.1, 1.05])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_precision_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved method_precision_bars.png")

    # ── Save JSON ──
    # Strip non-serializable rdm tuples
    for name in all_results:
        all_results[name].pop("_rdms", None)

    results_file = OUTPUT_DIR / "method_precision_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"  Saved {results_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
