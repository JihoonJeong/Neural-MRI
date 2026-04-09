"""Paper #5 Finding #8 — 3-Layer Decomposition (Final)

4 vector sets per model enable clean measurement of all 3 layers:
  A. fp16 comprehension       (Paper #5)
  B. fp16 generation (loose)  (Paper #5 — paper5_extract.py generation mode)
  C. fp16 generation (matched)(new — protocol-matched to Paper #6)
  D. INT8 generation (matched)(Paper #6)

Layer 1 — Coarse method dissociation (comp vs gen):
    A vs C  → method effect with no sub-parameter confound
    A vs B  → method effect with sub-parameter confound (for comparison)

Layer 2 — Sub-parameter sensitivity (within "generation"):
    B vs C  → same method category, same precision, different sub-parameters
    Direct measurement of how much 8 hyperparameters affect emotion geometry

Layer 3 — True precision effect (matched protocol):
    C vs D  → same method, same protocol, only precision differs (fp16 vs INT8)
    The clean precision comparison Luca originally requested

Models: Mistral 7B Instruct, Llama 3.1 8B Instruct

Usage:
    cd backend && uv run python ../paper5/analysis/finding8_3layer.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

OUTPUT_DIR = Path(__file__).parent / "finding8_3layer_output"
OUTPUT_DIR.mkdir(exist_ok=True)

ROOT = Path(__file__).resolve().parent.parent.parent
P5_DIR = ROOT / "scripts" / "paper5_output"
P5_MATCHED_DIR = ROOT / "scripts" / "paper5_output_protocol_matched"
P6_DIR = Path("/Users/jihoon/Projects/ludex/research/emotion_benchmark")

MODELS = {
    "Mistral 7B Instruct": {
        "p5_key": "mistralai_Mistral-7B-Instruct-v0.3",
        "p5_matched": P5_MATCHED_DIR / "mistralai_Mistral-7B-Instruct-v0.3" / "emotion_vectors_v2.pt",
        "p6_int8": P6_DIR / "emotion_pilot_v2_output" / "emotion_vectors_v2.pt",
    },
    "Llama 3.1 8B Instruct": {
        "p5_key": "meta-llama_Llama-3.1-8B-Instruct",
        "p5_matched": P5_MATCHED_DIR / "meta-llama_Llama-3.1-8B-Instruct" / "emotion_vectors_v2.pt",
        "p6_int8": P6_DIR / "llama31_inst_gen_output" / "emotion_vectors_v2.pt",
    },
}


# ── Loaders ──────────────────────────────────────────────────────────────


def load_p5_comp(model_dir: str) -> dict[str, np.ndarray]:
    d = P5_DIR / model_dir
    meta = json.loads((d / "metadata.json").read_text())
    pt = torch.load(d / "vectors_comprehension.pt", map_location="cpu", weights_only=False)
    best = str(meta["best_layer"])
    return {e: v.numpy() for e, v in pt["vectors"][best].items()}


def load_p5_gen_loose(model_dir: str) -> dict[str, np.ndarray]:
    d = P5_DIR / model_dir
    pt = torch.load(d / "vectors_generation.pt", map_location="cpu", weights_only=False)
    return {e: v.numpy() for e, v in pt["vectors"].items()}


def load_pt_v2(path: Path) -> dict[str, np.ndarray]:
    """Load Paper #6 v2 format (also used by protocol-matched extraction)."""
    pt = torch.load(path, map_location="cpu", weights_only=False)
    return {e: v.numpy() for e, v in pt["vectors"].items()}


# ── Analysis primitives ──────────────────────────────────────────────────


def post_center(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Subtract the mean of the vector set (equalizes baseline confound)."""
    keys = sorted(vectors.keys())
    mat = np.stack([vectors[k] for k in keys])
    mean = mat.mean(axis=0)
    return {k: vectors[k] - mean for k in keys}


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


def compare(set_a: dict, set_b: dict, label: str) -> dict:
    """Post-center, restrict to common emotions, compute RDM Spearman + per-emotion cos."""
    a_c = post_center(set_a)
    b_c = post_center(set_b)
    common = sorted(set(a_c.keys()) & set(b_c.keys()))

    a_sub = {e: a_c[e] for e in common}
    b_sub = {e: b_c[e] for e in common}

    rdm_a, _ = compute_rdm(a_sub)
    rdm_b, _ = compute_rdm(b_sub)
    rho, p = stats.spearmanr(upper_tri(rdm_a), upper_tri(rdm_b))

    # Per-emotion cosine
    pe_cos = {}
    for e in common:
        v1 = a_sub[e] / (np.linalg.norm(a_sub[e]) + 1e-8)
        v2 = b_sub[e] / (np.linalg.norm(b_sub[e]) + 1e-8)
        pe_cos[e] = float(np.dot(v1, v2))

    return {
        "label": label,
        "n_common": len(common),
        "rdm_spearman_rho": round(float(rho), 4),
        "rdm_spearman_p": float(f"{p:.2e}"),
        "per_emotion_cosine_mean": round(float(np.mean(list(pe_cos.values()))), 4),
        "per_emotion_cosine_min": round(float(min(pe_cos.values())), 4),
        "per_emotion_cosine_max": round(float(max(pe_cos.values())), 4),
        "per_emotion_cosines": {k: round(v, 4) for k, v in pe_cos.items()},
        "rdm_a": rdm_a,
        "rdm_b": rdm_b,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 75)
    print("Paper #5 Finding #8 — 3-Layer Decomposition (Final)")
    print("=" * 75)
    print()
    print("Vector sets per model:")
    print("  A = fp16 comprehension     (Paper #5)")
    print("  B = fp16 gen loose         (Paper #5, paper5_extract.py)")
    print("  C = fp16 gen matched       (new, Paper #6 protocol)")
    print("  D = INT8 gen matched       (Paper #6)")
    print()

    all_results = {}
    summary_table = []

    for name, paths in MODELS.items():
        print("─" * 75)
        print(f"  {name}")
        print("─" * 75)

        # Load all 4 vector sets
        a = load_p5_comp(paths["p5_key"])
        b = load_p5_gen_loose(paths["p5_key"])
        c = load_pt_v2(paths["p5_matched"])
        d = load_pt_v2(paths["p6_int8"])

        print(f"  A fp16 comp:        {len(a)} emotions")
        print(f"  B fp16 gen loose:   {len(b)} emotions")
        print(f"  C fp16 gen matched: {len(c)} emotions")
        print(f"  D INT8 gen matched: {len(d)} emotions")

        # ── Layer 1: Coarse method dissociation ──
        print("\n  [Layer 1] Coarse method dissociation (comp vs gen)")
        l1_clean = compare(a, c, "A vs C: comp vs matched-gen (clean)")
        l1_loose = compare(a, b, "A vs B: comp vs loose-gen (with sub-param confound)")
        print(f"      A vs C (clean):  ρ = {l1_clean['rdm_spearman_rho']:+.4f} "
              f"(p={l1_clean['rdm_spearman_p']:.2e})")
        print(f"      A vs B (loose):  ρ = {l1_loose['rdm_spearman_rho']:+.4f} "
              f"(p={l1_loose['rdm_spearman_p']:.2e})")

        # ── Layer 2: Sub-parameter sensitivity ──
        print("\n  [Layer 2] Sub-parameter sensitivity (gen loose vs gen matched)")
        l2 = compare(b, c, "B vs C: same method/precision, different sub-parameters")
        print(f"      B vs C: ρ = {l2['rdm_spearman_rho']:+.4f} "
              f"(p={l2['rdm_spearman_p']:.2e})")
        print(f"      → Within 'generation' category, 8 hyperparameters cause this much divergence")

        # ── Layer 3: True precision effect ──
        print("\n  [Layer 3] True precision effect (matched protocol, fp16 vs INT8)")
        l3 = compare(c, d, "C vs D: matched protocol, only precision differs")
        print(f"      C vs D: ρ = {l3['rdm_spearman_rho']:+.4f} "
              f"(p={l3['rdm_spearman_p']:.2e})")

        # Interpret precision effect
        if l3["rdm_spearman_rho"] > 0.9:
            verdict = "NEGLIGIBLE — quantization preserves emotion geometry"
        elif l3["rdm_spearman_rho"] > 0.7:
            verdict = "MILD — coarse structure preserved, fine details differ"
        elif l3["rdm_spearman_rho"] > 0.5:
            verdict = "MODERATE — measurable but not destructive"
        elif l3["rdm_spearman_rho"] > 0.3:
            verdict = "STRONG — significant geometry change"
        else:
            verdict = "SEVERE — quantization fundamentally alters emotion geometry"
        print(f"      → Precision verdict: {verdict}")

        # ── Conflated reference ──
        print("\n  [Reference] Original Task 2 conflated comparison")
        l_conflated = compare(a, d, "A vs D: comp fp16 vs gen INT8 (Task 2 mistake)")
        print(f"      A vs D: ρ = {l_conflated['rdm_spearman_rho']:+.4f}")

        all_results[name] = {
            "layer_1_method_clean": {
                "label": l1_clean["label"],
                "n_common": l1_clean["n_common"],
                "rdm_spearman_rho": l1_clean["rdm_spearman_rho"],
                "rdm_spearman_p": l1_clean["rdm_spearman_p"],
                "per_emotion_cosine_mean": l1_clean["per_emotion_cosine_mean"],
            },
            "layer_1_method_loose": {
                "label": l1_loose["label"],
                "rdm_spearman_rho": l1_loose["rdm_spearman_rho"],
                "rdm_spearman_p": l1_loose["rdm_spearman_p"],
            },
            "layer_2_subparam_sensitivity": {
                "label": l2["label"],
                "n_common": l2["n_common"],
                "rdm_spearman_rho": l2["rdm_spearman_rho"],
                "rdm_spearman_p": l2["rdm_spearman_p"],
                "per_emotion_cosine_mean": l2["per_emotion_cosine_mean"],
            },
            "layer_3_precision_clean": {
                "label": l3["label"],
                "n_common": l3["n_common"],
                "rdm_spearman_rho": l3["rdm_spearman_rho"],
                "rdm_spearman_p": l3["rdm_spearman_p"],
                "per_emotion_cosine_mean": l3["per_emotion_cosine_mean"],
                "verdict": verdict,
            },
            "reference_conflated": {
                "label": l_conflated["label"],
                "rdm_spearman_rho": l_conflated["rdm_spearman_rho"],
            },
        }

        summary_table.append({
            "model": name,
            "L1_clean": l1_clean["rdm_spearman_rho"],
            "L1_loose": l1_loose["rdm_spearman_rho"],
            "L2_subparam": l2["rdm_spearman_rho"],
            "L3_precision": l3["rdm_spearman_rho"],
            "conflated": l_conflated["rdm_spearman_rho"],
        })

        # Store rdms for plotting (will strip before json save)
        all_results[name]["_rdms"] = {
            "L1_clean": (l1_clean["rdm_a"], l1_clean["rdm_b"]),
            "L2": (l2["rdm_a"], l2["rdm_b"]),
            "L3": (l3["rdm_a"], l3["rdm_b"]),
        }
        print()

    # ── Summary table ──
    print("=" * 75)
    print("SUMMARY — 3-Layer Decomposition")
    print("=" * 75)
    print()
    print(f"  {'Model':<25s} {'L1 clean':>10s} {'L1 loose':>10s} {'L2 sub':>10s} {'L3 prec':>10s} {'(Task 2)':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in summary_table:
        print(f"  {r['model']:<25s} {r['L1_clean']:>+10.4f} {r['L1_loose']:>+10.4f} "
              f"{r['L2_subparam']:>+10.4f} {r['L3_precision']:>+10.4f} {r['conflated']:>+10.4f}")

    # Means
    means = {
        k: float(np.mean([r[k] for r in summary_table]))
        for k in ["L1_clean", "L1_loose", "L2_subparam", "L3_precision", "conflated"]
    }
    print(f"  {'mean':<25s} {means['L1_clean']:>+10.4f} {means['L1_loose']:>+10.4f} "
          f"{means['L2_subparam']:>+10.4f} {means['L3_precision']:>+10.4f} {means['conflated']:>+10.4f}")

    # ── Interpretation ──
    print("\n" + "=" * 75)
    print("INTERPRETATION")
    print("=" * 75)

    print(f"\n  Layer 1 (method dissociation, clean):  ρ = {means['L1_clean']:+.3f}")
    if abs(means['L1_clean']) < 0.1:
        print("    → Comp and gen produce nearly orthogonal emotion representations")
        print("    → Paper #6 finding REPLICATED in 7B+ models with clean protocol")
    elif means['L1_clean'] < 0.3:
        print("    → Strong but not complete dissociation")
    else:
        print("    → Partial dissociation")

    print(f"\n  Layer 2 (sub-parameter sensitivity):   ρ = {means['L2_subparam']:+.3f}")
    if means['L2_subparam'] < 0.3:
        print("    → 8 hyperparameters cause MAJOR divergence within 'generation' category")
        print("    → 'Generation method' is not a single procedure but an 8-D hyperparameter space")
    elif means['L2_subparam'] < 0.7:
        print("    → Sub-parameters cause meaningful but partial divergence")
    else:
        print("    → Sub-parameters have only mild effect")

    print(f"\n  Layer 3 (TRUE precision effect):       ρ = {means['L3_precision']:+.3f}")
    if means['L3_precision'] > 0.9:
        print("    → INT8 quantization PRESERVES emotion geometry")
        print("    → Paper #6 4-bit results compatible with fp16 (when method matches)")
    elif means['L3_precision'] > 0.7:
        print("    → Mild precision effect — coarse preserved, fine details differ")
    elif means['L3_precision'] > 0.5:
        print("    → Moderate precision effect — measurable but not destructive")
    elif means['L3_precision'] > 0.3:
        print("    → STRONG precision effect — geometry significantly altered")
    else:
        print("    → SEVERE precision effect — quantization fundamentally changes geometry")

    # ── Plots ──
    print("\n" + "=" * 75)
    print("Generating figures...")
    print("=" * 75)

    # Bar chart: 3 layers + conflated reference
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_table))
    width = 0.18
    ax.bar(x - 2*width, [r["L1_clean"] for r in summary_table], width,
           label="L1: Method (comp vs matched-gen)", color="#3498db")
    ax.bar(x - width, [r["L1_loose"] for r in summary_table], width,
           label="L1 loose: comp vs loose-gen", color="#85c1e9")
    ax.bar(x, [r["L2_subparam"] for r in summary_table], width,
           label="L2: Sub-params (loose vs matched gen)", color="#f39c12")
    ax.bar(x + width, [r["L3_precision"] for r in summary_table], width,
           label="L3: TRUE precision (fp16 vs INT8 gen)", color="#2ecc71")
    ax.bar(x + 2*width, [r["conflated"] for r in summary_table], width,
           label="(Reference: Task 2 conflated)", color="#e74c3c", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in summary_table], rotation=10, ha="right")
    ax.set_ylabel("RDM Spearman ρ")
    ax.set_title("Finding #8: 3-Layer Decomposition", fontweight="bold", fontsize=12)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="ρ=0.7")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim([-0.15, 1.05])
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finding8_3layer_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved finding8_3layer_bars.png")

    # Heatmap matrix: 4×4 set comparisons per model
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    set_labels = ["A:fp16 comp", "B:fp16 gen\n(loose)", "C:fp16 gen\n(matched)", "D:INT8 gen\n(matched)"]

    for ax_idx, (name, paths) in enumerate(MODELS.items()):
        a = load_p5_comp(paths["p5_key"])
        b = load_p5_gen_loose(paths["p5_key"])
        c = load_pt_v2(paths["p5_matched"])
        d = load_pt_v2(paths["p6_int8"])
        sets = [a, b, c, d]
        n = 4
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                comp = compare(sets[i], sets[j], f"{i} vs {j}")
                mat[i, j] = comp["rdm_spearman_rho"]
                mat[j, i] = comp["rdm_spearman_rho"]

        ax = axes[ax_idx]
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(set_labels, fontsize=9)
        ax.set_yticklabels(set_labels, fontsize=9)
        for i in range(n):
            for j in range(n):
                color = "white" if mat[i, j] < 0.4 or mat[i, j] > 0.85 else "black"
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")
        ax.set_title(name, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    plt.suptitle("Finding #8: 4-Way Vector Set Comparison Matrix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finding8_4way_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved finding8_4way_matrix.png")

    # ── Save JSON ──
    for name in all_results:
        all_results[name].pop("_rdms", None)

    all_results["_summary"] = {
        "layer_means": {k: round(v, 4) for k, v in means.items()},
        "per_model": summary_table,
    }

    results_file = OUTPUT_DIR / "finding8_3layer_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"  Saved {results_file}")

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
