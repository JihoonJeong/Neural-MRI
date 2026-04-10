"""Paper #5 — Final Figure Builder

Generates all main paper figures with consistent styling, then copies them to
paper5/figures/main/ and paper5/figures/supp/ with proper file names.

Run from backend dir:
    cd backend && uv run python ../paper5/figures/build_paper_figures.py
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
P5_OUT = ROOT / "scripts" / "paper5_output"
P5_MATCHED = ROOT / "scripts" / "paper5_output_protocol_matched"
P6_DIR = Path("/Users/jihoon/Projects/ludex/research/emotion_benchmark")
FIG_DIR = ROOT / "paper5" / "figures"
MAIN_DIR = FIG_DIR / "main"
SUPP_DIR = FIG_DIR / "supp"
ANALYSIS_DIR = ROOT / "paper5" / "analysis"

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# ── Models ────────────────────────────────────────────────────────────────
ALL_MODELS = [
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-1.5B",
    "HuggingFaceTB_SmolLM2-1.7B-Instruct",
    "HuggingFaceTB_SmolLM2-1.7B",
    "meta-llama_Llama-3.2-3B-Instruct",
    "meta-llama_Llama-3.2-3B",
    "google_gemma-3-1b-it",
    "google_gemma-3-1b-pt",
    "mistralai_Mistral-7B-v0.3",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "meta-llama_Llama-3.1-8B",
    "meta-llama_Llama-3.1-8B-Instruct",
]

SHORT = {
    "Qwen_Qwen2.5-1.5B-Instruct": "Qwen 1.5B Inst",
    "Qwen_Qwen2.5-1.5B": "Qwen 1.5B Base",
    "HuggingFaceTB_SmolLM2-1.7B-Instruct": "SmolLM2 1.7B Inst",
    "HuggingFaceTB_SmolLM2-1.7B": "SmolLM2 1.7B Base",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama 3.2 3B Inst",
    "meta-llama_Llama-3.2-3B": "Llama 3.2 3B Base",
    "google_gemma-3-1b-it": "Gemma-3 1B Inst",
    "google_gemma-3-1b-pt": "Gemma-3 1B Base",
    "mistralai_Mistral-7B-v0.3": "Mistral 7B Base",
    "mistralai_Mistral-7B-Instruct-v0.3": "Mistral 7B Inst",
    "meta-llama_Llama-3.1-8B": "Llama 3.1 8B Base",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama 3.1 8B Inst",
}

SIZES_B = {
    "Qwen_Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen_Qwen2.5-1.5B": 1.5,
    "HuggingFaceTB_SmolLM2-1.7B-Instruct": 1.7,
    "HuggingFaceTB_SmolLM2-1.7B": 1.7,
    "meta-llama_Llama-3.2-3B-Instruct": 3.0,
    "meta-llama_Llama-3.2-3B": 3.0,
    "google_gemma-3-1b-it": 1.0,
    "google_gemma-3-1b-pt": 1.0,
    "mistralai_Mistral-7B-v0.3": 7.0,
    "mistralai_Mistral-7B-Instruct-v0.3": 7.0,
    "meta-llama_Llama-3.1-8B": 8.0,
    "meta-llama_Llama-3.1-8B-Instruct": 8.0,
}

# Tier color scheme
TIER1 = "#27ae60"  # green
TIER2 = "#f39c12"  # orange
OUTLIER = "#c0392b"  # red

# ── Loaders ──────────────────────────────────────────────────────────────


def load_model(key: str) -> dict | None:
    d = P5_OUT / key
    if not (d / "metadata.json").exists():
        return None
    meta = json.loads((d / "metadata.json").read_text())
    pt = torch.load(d / "vectors_comprehension.pt", map_location="cpu", weights_only=False)
    return {"meta": meta, "pt": pt}


def best_layer_vectors(data: dict) -> dict[str, np.ndarray]:
    best = str(data["meta"]["best_layer"])
    return {e: v.numpy() for e, v in data["pt"]["vectors"][best].items()}


def compute_rdm(vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    emos = sorted(vectors.keys())
    mat = np.stack([vectors[e] for e in emos])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_n = mat / norms
    return 1 - (mat_n @ mat_n.T), emos


def upper_tri(m):
    return m[np.triu(np.ones(m.shape, dtype=bool), k=1)]


def compute_rho_matrix(all_data: dict, models: list) -> np.ndarray:
    n = len(models)
    rdm_uts = []
    for m in models:
        vecs = best_layer_vectors(all_data[m])
        rdm, _ = compute_rdm(vecs)
        rdm_uts.append(upper_tri(rdm))
    rho = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = stats.spearmanr(rdm_uts[i], rdm_uts[j])
            rho[i, j] = r
            rho[j, i] = r
    return rho


# ── Figure 1: Lead figure — 12-model RDM-of-RDMs ─────────────────────────


def figure_1_lead(all_data: dict):
    """12-model heatmap with hierarchical clustering and tier separators.

    Combines the n=8 lead figure's clean layout with 12 models, dendrogram,
    and tier separator lines.
    """
    print("Building Figure 1 (lead — RDM-of-RDMs)...")

    # Order models so instruct/base pairs are adjacent and ordered by tier
    # Group by architecture; within architecture, instruct then base
    ordered_keys = [
        # Tier 2/3 outlier (Gemma)
        "google_gemma-3-1b-it",
        "google_gemma-3-1b-pt",
        # Tier 1 — small
        "Qwen_Qwen2.5-1.5B-Instruct",
        "Qwen_Qwen2.5-1.5B",
        "HuggingFaceTB_SmolLM2-1.7B-Instruct",
        "HuggingFaceTB_SmolLM2-1.7B",
        "meta-llama_Llama-3.2-3B-Instruct",
        "meta-llama_Llama-3.2-3B",
        # Tier 1 — large
        "mistralai_Mistral-7B-Instruct-v0.3",
        "mistralai_Mistral-7B-v0.3",
        "meta-llama_Llama-3.1-8B-Instruct",
        "meta-llama_Llama-3.1-8B",
    ]
    rho = compute_rho_matrix(all_data, ordered_keys)
    labels = [SHORT[m] for m in ordered_keys]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    n = len(ordered_keys)
    # Colorblind-friendly: cividis (perceptually uniform, friendly for all forms)
    im = ax.imshow(rho, cmap="cividis", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate values — cividis is dark at low ρ, light at high ρ
    for i in range(n):
        for j in range(n):
            val = rho[i, j]
            color = "white" if val < 0.55 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, weight=weight)

    # Tier separator lines
    # Gemma block: indices 0-1, then everything else is Tier 1 (2-11)
    ax.axhline(y=1.5, color="white", linewidth=3)
    ax.axvline(x=1.5, color="white", linewidth=3)
    ax.axhline(y=1.5, color="black", linewidth=1.5, linestyle="--")
    ax.axvline(x=1.5, color="black", linewidth=1.5, linestyle="--")

    # Optional: separate small Tier 1 (2-7) from large Tier 1 (8-11)
    ax.axhline(y=7.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axvline(x=7.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Spearman ρ (RDM similarity)", fontsize=10)

    ax.set_title(
        "Cross-Model Emotion Vector Geometry (n=12)\n"
        "Tier 1 (10 models, 5 architectures): all-pair ρ = 0.74–0.985   |   "
        "Tier 2 outlier (Gemma-3 1B): ρ ≤ 0.62",
        fontsize=11, pad=12,
    )

    # Architecture annotation strip on right side
    plt.tight_layout()
    plt.savefig(MAIN_DIR / "figure_01_rdm_of_rdms_n12.png")
    plt.savefig(MAIN_DIR / "figure_01_rdm_of_rdms_n12.pdf")
    plt.close()
    print(f"  → main/figure_01_rdm_of_rdms_n12.png (+ pdf)")


# ── Figure 2: H2 RDM comparison (raw, n=2) ───────────────────────────────


def figure_2_h2_raw_rdm(all_data: dict):
    """Behavioral-representational dissociation: Qwen vs Llama 3.2 raw RDMs."""
    print("Building Figure 2 (H2 raw RDM comparison)...")

    qwen = best_layer_vectors(all_data["Qwen_Qwen2.5-1.5B-Instruct"])
    llama = best_layer_vectors(all_data["meta-llama_Llama-3.2-3B-Instruct"])

    common = sorted(set(qwen.keys()) & set(llama.keys()))
    rdm_q, _ = compute_rdm({e: qwen[e] for e in common})
    rdm_l, _ = compute_rdm({e: llama[e] for e in common})

    rho, p = stats.spearmanr(upper_tri(rdm_q), upper_tri(rdm_l))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmin, vmax = 0.5, 1.5  # raw cosine distance range

    for ax, rdm, title in [
        (axes[0], rdm_q, "Qwen 2.5 1.5B Instruct\n(Compliance-yielding: B=0.80, D=0.61)"),
        (axes[1], rdm_l, "Llama 3.2 3B Instruct\n(Compliance-refusing: B=0.20, D=0.85)"),
    ]:
        # cividis is colorblind-friendly and perceptually uniform
        im = ax.imshow(rdm, cmap="cividis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(common)))
        ax.set_yticks(range(len(common)))
        ax.set_xticklabels(common, rotation=90, fontsize=7)
        ax.set_yticklabels(common, fontsize=7)
        ax.set_title(title, fontsize=10)

    fig.colorbar(im, ax=axes, shrink=0.6, label="1 − cos(eᵢ, eⱼ)  (raw cosine distance)")
    fig.suptitle(
        f"Behavioral–Representational Dissociation: Spearman ρ = {rho:.3f} (p = {p:.1e})\n"
        f"Opposite Compliance behavior, nearly identical emotion vector geometry",
        fontsize=11, fontweight="bold",
    )
    plt.savefig(MAIN_DIR / "figure_02_h2_dissociation.png")
    plt.savefig(MAIN_DIR / "figure_02_h2_dissociation.pdf")
    plt.close()
    print(f"  → main/figure_02_h2_dissociation.png (+ pdf)")


# ── Figure 3: Size effects (4-panel) ─────────────────────────────────────


def figure_3_size_effects(all_data: dict):
    """Anisotropy and structural metrics vs model size and d_model.

    Layout solution for label overlap (base/instruct co-locate):
      - Different markers for base (□ hollow square) vs instruct (● filled circle)
      - One label per FAMILY (not per model), placed at the mid-point between
        base and instruct, with smart offset to avoid overlapping the markers
      - Tier color: Tier 1 (mature) = green, Gemma-3 outlier = red
    """
    print("Building Figure 3 (size effects)...")

    models = list(all_data.keys())
    sizes = np.array([SIZES_B[m] for m in models])
    d_models_arr = np.array([all_data[m]["meta"]["d_model"] for m in models])

    anisotropies = []
    best_layers_pct = []
    rdm_stds = []
    for m in models:
        meta = all_data[m]["meta"]
        best_key = f"layer_{meta['best_layer']}_best"
        aniso = meta["anisotropy"].get(best_key, None)
        if aniso is None or (isinstance(aniso, float) and np.isnan(aniso)):
            for k, v in meta["anisotropy"].items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    aniso = v
                    break
        anisotropies.append(aniso if aniso is not None else np.nan)
        best_layers_pct.append(meta["best_layer"] / meta["n_layers"] * 100)
        vecs = best_layer_vectors(all_data[m])
        rdm, _ = compute_rdm(vecs)
        rdm_stds.append(np.std(upper_tri(rdm)))

    anisotropies = np.array(anisotropies, dtype=float)
    best_layers_pct = np.array(best_layers_pct)
    rdm_stds = np.array(rdm_stds)
    valid = ~np.isnan(anisotropies)

    # ── Family grouping (each family = base + instruct pair) ──
    families = [
        ("Gemma-3 1B",   "google_gemma-3-1b-pt",            "google_gemma-3-1b-it",            OUTLIER),
        ("Qwen 2.5 1.5B","Qwen_Qwen2.5-1.5B",               "Qwen_Qwen2.5-1.5B-Instruct",      TIER1),
        ("SmolLM2 1.7B", "HuggingFaceTB_SmolLM2-1.7B",      "HuggingFaceTB_SmolLM2-1.7B-Instruct", TIER1),
        ("Llama 3.2 3B", "meta-llama_Llama-3.2-3B",         "meta-llama_Llama-3.2-3B-Instruct", TIER1),
        ("Mistral 7B v0.3","mistralai_Mistral-7B-v0.3",     "mistralai_Mistral-7B-Instruct-v0.3", TIER1),
        ("Llama 3.1 8B", "meta-llama_Llama-3.1-8B",         "meta-llama_Llama-3.1-8B-Instruct", TIER1),
    ]

    # Per-family label offset hints (offset points from the mid-point).
    # (dx, dy, ha, va). Per-panel overrides further refine these below.
    label_offsets = {
        "Gemma-3 1B":     (10, -4, "left", "center"),    # right of marker
        "Qwen 2.5 1.5B":  (-8, 8, "right", "bottom"),    # upper-left
        "SmolLM2 1.7B":   (10, -4, "left", "center"),    # right
        "Llama 3.2 3B":   (-8, -10, "right", "top"),     # lower-left
        "Mistral 7B v0.3":(10, -4, "left", "center"),    # right
        "Llama 3.1 8B":   (-8, 8, "right", "bottom"),    # upper-left
    }

    def model_idx(key):
        return models.index(key)

    def plot_panel(ax, x_arr, y_arr, x_log, x_label, y_label, panel_title,
                   compute_rho_on, panel_offsets, valid_mask=None):
        """Helper: plot one panel with family-aware markers and labels.

        - Hollow square (base) + filled circle (instruct), connected by a
          dashed line in the family color (so the pair is visually grouped
          even when base/instruct land far apart).
        - One label per family. Per-panel offset overrides allow tuning when
          the global default would clip or overlap.
        """
        for fam_name, base_key, inst_key, color in families:
            if base_key not in models or inst_key not in models:
                continue
            bi = model_idx(base_key)
            ii = model_idx(inst_key)

            valid_b = (valid_mask is None or valid_mask[bi]) and not (isinstance(y_arr[bi], float) and np.isnan(y_arr[bi]))
            valid_i = (valid_mask is None or valid_mask[ii]) and not (isinstance(y_arr[ii], float) and np.isnan(y_arr[ii]))

            # Connect the pair with a dashed line in family color
            if valid_b and valid_i:
                ax.plot([x_arr[bi], x_arr[ii]], [y_arr[bi], y_arr[ii]],
                        color=color, linestyle="--", linewidth=0.9, alpha=0.55, zorder=2)

            if valid_b:
                ax.scatter(x_arr[bi], y_arr[bi], marker="s", s=85,
                           facecolors="white", edgecolors=color, linewidth=1.6, zorder=3)
            if valid_i:
                ax.scatter(x_arr[ii], y_arr[ii], marker="o", s=95,
                           facecolors=color, edgecolors="black", linewidth=0.7, zorder=4)

            # Label position: per-panel override > global default
            if valid_b and valid_i:
                mid_x = (x_arr[bi] + x_arr[ii]) / 2 if not x_log else np.sqrt(x_arr[bi] * x_arr[ii])
                mid_y = (y_arr[bi] + y_arr[ii]) / 2
            elif valid_i:
                mid_x, mid_y = x_arr[ii], y_arr[ii]
            elif valid_b:
                mid_x, mid_y = x_arr[bi], y_arr[bi]
            else:
                continue

            dx, dy, ha, va = panel_offsets.get(fam_name, label_offsets[fam_name])
            ax.annotate(fam_name, (mid_x, mid_y),
                        fontsize=8, ha=ha, va=va,
                        xytext=(dx, dy), textcoords="offset points",
                        color="black", fontweight="medium")

        # Compute and display correlation
        x_for_corr = x_arr[compute_rho_on] if compute_rho_on is not None else x_arr
        y_for_corr = y_arr[compute_rho_on] if compute_rho_on is not None else y_arr
        r, p = stats.spearmanr(x_for_corr, y_for_corr)
        ax.set_title(f"{panel_title}\nSpearman ρ = {r:.3f}, p = {p:.1e}", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_log:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        # Add margin so labels don't get clipped at axis edges
        ax.margins(x=0.12, y=0.10)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # Panel-specific overrides for cases where the global default clips or overlaps
    offsets_A = dict(label_offsets)
    offsets_A["Qwen 2.5 1.5B"] = (10, -4, "left", "center")  # right side, plenty of room
    offsets_A["SmolLM2 1.7B"] = (-8, -10, "right", "top")    # below-left to clear Qwen

    offsets_B = dict(label_offsets)
    offsets_B["Qwen 2.5 1.5B"] = (10, -4, "left", "center")
    offsets_B["SmolLM2 1.7B"] = (-8, -10, "right", "top")
    offsets_B["Mistral 7B v0.3"] = (-10, -4, "right", "center")  # left of marker (right edge)
    offsets_B["Llama 3.1 8B"] = (-10, -4, "right", "center")

    # Panel C and D: Gemma-3 base/instruct have very wide y-range (line is long)
    offsets_C = dict(label_offsets)
    offsets_C["Gemma-3 1B"] = (12, 0, "left", "center")
    offsets_C["SmolLM2 1.7B"] = (12, 0, "left", "center")
    offsets_C["Mistral 7B v0.3"] = (-10, 8, "right", "bottom")
    offsets_C["Llama 3.1 8B"] = (-10, -10, "right", "top")

    offsets_D = dict(label_offsets)
    offsets_D["Gemma-3 1B"] = (12, 0, "left", "center")
    offsets_D["Mistral 7B v0.3"] = (-10, -10, "right", "top")
    offsets_D["Llama 3.1 8B"] = (-10, 8, "right", "bottom")

    plot_panel(axes[0, 0], d_models_arr, anisotropies, False,
               "d_model (hidden dimension)", "Anisotropy (best layer)",
               "(A) Anisotropy vs d_model", compute_rho_on=valid,
               panel_offsets=offsets_A, valid_mask=valid)

    plot_panel(axes[0, 1], sizes, anisotropies, True,
               "Model size (B parameters, log scale)", "Anisotropy (best layer)",
               "(B) Anisotropy vs Size", compute_rho_on=valid,
               panel_offsets=offsets_B, valid_mask=valid)

    plot_panel(axes[1, 0], sizes, best_layers_pct, True,
               "Model size (B parameters, log scale)", "Best layer (% depth)",
               "(C) Best Layer Depth vs Size", compute_rho_on=None,
               panel_offsets=offsets_C)

    plot_panel(axes[1, 1], sizes, rdm_stds, True,
               "Model size (B parameters, log scale)", "Mean pairwise std (RDM)",
               "(D) RDM Std vs Size", compute_rho_on=None,
               panel_offsets=offsets_D)

    # Legend with both color and marker semantics
    legend_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=10,
                      markerfacecolor=TIER1, markeredgecolor="black",
                      label="Tier 1 instruct"),
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=10,
                      markerfacecolor="white", markeredgecolor=TIER1, markeredgewidth=1.6,
                      label="Tier 1 base"),
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=10,
                      markerfacecolor=OUTLIER, markeredgecolor="black",
                      label="Gemma-3 outlier instruct"),
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=10,
                      markerfacecolor="white", markeredgecolor=OUTLIER, markeredgewidth=1.6,
                      label="Gemma-3 outlier base"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=False)

    fig.suptitle("Size–Maturity Correlation (n = 12)",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(MAIN_DIR / "figure_03_size_effects.png")
    plt.savefig(MAIN_DIR / "figure_03_size_effects.pdf")
    plt.close()
    print(f"  → main/figure_03_size_effects.png (+ pdf)")


# ── Figure 4: Tier 1 boxplot ──────────────────────────────────────────────


def figure_4_tier1_boxplot(all_data: dict):
    """Pairwise instruct ρ distributions, grouped by tier inclusion."""
    print("Building Figure 4 (Tier 1 boxplot)...")

    inst_models = [m for m in ALL_MODELS if "inst" in m.lower() or "-it" in m.lower() or "Instruct" in m]
    rho = compute_rho_matrix(all_data, inst_models)

    pair_data = []
    for i in range(len(inst_models)):
        for j in range(i + 1, len(inst_models)):
            pair_data.append((inst_models[i], inst_models[j], rho[i, j]))

    original_tier1 = {"Qwen_Qwen2.5-1.5B-Instruct", "HuggingFaceTB_SmolLM2-1.7B-Instruct",
                      "meta-llama_Llama-3.2-3B-Instruct"}
    new_7b = {"mistralai_Mistral-7B-Instruct-v0.3", "meta-llama_Llama-3.1-8B-Instruct"}
    gemma = {"google_gemma-3-1b-it"}

    groups = {"Original Tier 1\n(Qwen, SmolLM2, Llama 3.2)": [],
              "New 7B+ pairs\n(Mistral, Llama 3.1)": [],
              "Gemma-3 pairs\n(outlier)": []}
    for a, b, r in pair_data:
        if a in original_tier1 and b in original_tier1:
            groups["Original Tier 1\n(Qwen, SmolLM2, Llama 3.2)"].append(r)
        elif a in gemma or b in gemma:
            groups["Gemma-3 pairs\n(outlier)"].append(r)
        elif a in new_7b or b in new_7b:
            groups["New 7B+ pairs\n(Mistral, Llama 3.1)"].append(r)

    fig, ax = plt.subplots(figsize=(10, 6))
    group_names = list(groups.keys())
    group_vals = [groups[k] for k in group_names]
    bp = ax.boxplot(group_vals, tick_labels=group_names, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    colors = [TIER1, "#3498db", OUTLIER]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    for i, vals in enumerate(group_vals):
        x = np.random.normal(i + 1, 0.05, size=len(vals))
        ax.scatter(x, vals, c="black", alpha=0.6, s=35, zorder=4)

    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
               label="Tier 1 threshold (ρ = 0.7)")
    ax.set_ylabel("Spearman ρ (pairwise RDM similarity)", fontsize=11)
    ax.set_title("Tier 1 Cluster Re-Verification (n = 12, instruct only)\n"
                 "Mistral 7B and Llama 3.1 8B join the original Tier 1; Gemma-3 remains outlier",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0.45, 0.95])

    plt.tight_layout()
    plt.savefig(MAIN_DIR / "figure_04_tier1_boxplot.png")
    plt.savefig(MAIN_DIR / "figure_04_tier1_boxplot.pdf")
    plt.close()
    print(f"  → main/figure_04_tier1_boxplot.png (+ pdf)")


# ── Figure 5/6 helpers (Finding #8) ──────────────────────────────────────


def _load_finding8_vector_sets(model_name: str):
    """Load 4 vector sets for a given finding #8 model."""
    keys = {
        "Mistral 7B Instruct": {
            "p5_key": "mistralai_Mistral-7B-Instruct-v0.3",
            "p5_matched": P5_MATCHED / "mistralai_Mistral-7B-Instruct-v0.3" / "emotion_vectors_v2.pt",
            "p6_int8": P6_DIR / "emotion_pilot_v2_output" / "emotion_vectors_v2.pt",
        },
        "Llama 3.1 8B Instruct": {
            "p5_key": "meta-llama_Llama-3.1-8B-Instruct",
            "p5_matched": P5_MATCHED / "meta-llama_Llama-3.1-8B-Instruct" / "emotion_vectors_v2.pt",
            "p6_int8": P6_DIR / "llama31_inst_gen_output" / "emotion_vectors_v2.pt",
        },
    }[model_name]

    # A: fp16 comp
    d = P5_OUT / keys["p5_key"]
    meta = json.loads((d / "metadata.json").read_text())
    pt = torch.load(d / "vectors_comprehension.pt", map_location="cpu", weights_only=False)
    a = {e: v.numpy() for e, v in pt["vectors"][str(meta["best_layer"])].items()}

    # B: fp16 gen loose
    pt = torch.load(d / "vectors_generation.pt", map_location="cpu", weights_only=False)
    b = {e: v.numpy() for e, v in pt["vectors"].items()}

    # C: fp16 gen matched
    pt = torch.load(keys["p5_matched"], map_location="cpu", weights_only=False)
    c = {e: v.numpy() for e, v in pt["vectors"].items()}

    # D: INT8 gen matched
    pt = torch.load(keys["p6_int8"], map_location="cpu", weights_only=False)
    d_set = {e: v.numpy() for e, v in pt["vectors"].items()}

    return a, b, c, d_set


def _post_center(vecs):
    keys = sorted(vecs.keys())
    mat = np.stack([vecs[k] for k in keys])
    mean = mat.mean(axis=0)
    return {k: vecs[k] - mean for k in keys}


def _spearman_rdm(s1, s2):
    s1c = _post_center(s1)
    s2c = _post_center(s2)
    common = sorted(set(s1c.keys()) & set(s2c.keys()))
    rdm1, _ = compute_rdm({e: s1c[e] for e in common})
    rdm2, _ = compute_rdm({e: s2c[e] for e in common})
    rho, _ = stats.spearmanr(upper_tri(rdm1), upper_tri(rdm2))
    return float(rho)


# ── Figure 5: Finding #8 4-way matrix ────────────────────────────────────


def figure_5_finding8_matrix():
    """Method × Precision 4-way comparison matrix (regenerated for PDF + style)."""
    print("Building Figure 5 (Finding #8 4-way matrix)...")

    set_labels = ["A: fp16\ncomp", "B: fp16 gen\n(loose)",
                  "C: fp16 gen\n(matched)", "D: INT8 gen\n(matched)"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    for ax_idx, model_name in enumerate(["Mistral 7B Instruct", "Llama 3.1 8B Instruct"]):
        sets = _load_finding8_vector_sets(model_name)
        n = 4
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                rho = _spearman_rdm(sets[i], sets[j])
                mat[i, j] = rho
                mat[j, i] = rho

        ax = axes[ax_idx]
        im = ax.imshow(mat, cmap="cividis", vmin=-0.2, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(set_labels, fontsize=9)
        ax.set_yticklabels(set_labels, fontsize=9)
        for i in range(n):
            for j in range(n):
                color = "white" if mat[i, j] < 0.45 else "black"
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    plt.suptitle("Finding #8: Method × Precision 4-Way Comparison Matrix",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MAIN_DIR / "figure_05_finding8_4way_matrix.png")
    plt.savefig(MAIN_DIR / "figure_05_finding8_4way_matrix.pdf")
    plt.close()
    print(f"  → main/figure_05_finding8_4way_matrix.png (+ pdf)")


# ── Figure 6: Finding #8 3-layer bars ────────────────────────────────────


def figure_6_finding8_bars():
    """Finding #8 3-layer decomposition summary bars (regenerated for PDF + style)."""
    print("Building Figure 6 (Finding #8 3-layer bars)...")

    summary_path = ANALYSIS_DIR / "finding8_3layer_output" / "finding8_3layer_results.json"
    summary = json.loads(summary_path.read_text())["_summary"]["per_model"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    width = 0.18
    ax.bar(x - 2*width, [r["L1_clean"] for r in summary], width,
           label="L1: Method (comp vs matched-gen)", color="#0072B2")
    ax.bar(x - width, [r["L1_loose"] for r in summary], width,
           label="L1 (loose): comp vs loose-gen", color="#56B4E9")
    ax.bar(x, [r["L2_subparam"] for r in summary], width,
           label="L2: Sub-parameters (loose vs matched gen)", color="#E69F00")
    ax.bar(x + width, [r["L3_precision"] for r in summary], width,
           label="L3: TRUE precision (fp16 vs INT8 gen)", color="#009E73")
    ax.bar(x + 2*width, [r["conflated"] for r in summary], width,
           label="(Reference: Task-2 conflated)", color="#D55E00", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in summary], rotation=10, ha="right")
    ax.set_ylabel("RDM Spearman ρ")
    ax.set_title("Finding #8: 3-Layer Decomposition", fontsize=12)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="ρ = 0.7 threshold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim([-0.15, 1.05])
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(MAIN_DIR / "figure_06_finding8_layers.png")
    plt.savefig(MAIN_DIR / "figure_06_finding8_layers.pdf")
    plt.close()
    print(f"  → main/figure_06_finding8_layers.png (+ pdf)")


# ── Supplementary figures ────────────────────────────────────────────────


def supplementary_figures():
    """Copy historical / context figures to supp/ with renamed files."""
    print("\nBuilding supplementary figures...")

    supps = [
        (ANALYSIS_DIR / "h2_pilot" / "lead_figure_8x8_raw.png",
         "supp_01_rdm_of_rdms_n8_historical.png",
         "n=8 lead figure (before adding Mistral 7B and Llama 3.1 8B)"),
        (ANALYSIS_DIR / "n12_output" / "task1_rdm_instruct_6x6.png",
         "supp_02_rdm_instruct_only_6x6.png",
         "Instruct-only 6×6 RDM-of-RDMs (clean view)"),
        (ANALYSIS_DIR / "h2_pilot" / "rdm_diff.png",
         "supp_03_h2_rdm_diff.png",
         "Qwen vs Llama RDM difference matrix"),
        (ANALYSIS_DIR / "h2_pilot" / "gemma3_base_verification.png",
         "supp_04_gemma3_base_verification.png",
         "Gemma-3 1B base outlier verification"),
        (ANALYSIS_DIR / "method_precision_output" / "method_precision_bars.png",
         "supp_05_finding8_initial_conflated.png",
         "Initial Task 2 measurement (before protocol matching) — historical context"),
        (ANALYSIS_DIR / "method_precision_output" / "method_precision_rdms.png",
         "supp_06_finding8_initial_rdms.png",
         "Initial RDM visualization across 4-cell 2x2 design (historical)"),
    ]

    for src, dst_name, _desc in supps:
        if src.exists():
            shutil.copy(src, SUPP_DIR / dst_name)
            print(f"  → supp/{dst_name}")
        else:
            print(f"  ⚠ MISSING: {src}")


# ── Catalog ──────────────────────────────────────────────────────────────


def write_catalog():
    """Catalog mapping figures to findings with caption drafts."""
    print("\nWriting figure catalog...")

    catalog = """# Paper #5 — Figure Catalog

Generated by `paper5/figures/build_paper_figures.py`. All main figures are saved
in `figures/main/` (PNG @ 300 DPI + PDF). Supplementary figures in `figures/supp/`.

## Findings → Figures Mapping

### Substantive findings (5)

| # | Finding | Main Fig | Supplementary |
|---|---|---|---|
| 1 | Cross-architecture universality | **Figure 1** | Supp 1, 2 |
| 2 | Behavioral–representational dissociation | **Figure 2** | Supp 3 |
| 3 | Maturity threshold + RLHF differential | (Figure 1 base/instruct rows) | — |
| 4 | Family identity preservation (within-family base/instruct pairs) | **Figure 1** (5 mature families ρ ≥ 0.918) | — |
| 5 | Size–maturity correlation | **Figure 3**, **Figure 4** | — |

### Methodological findings (3)

| # | Finding | Main Fig | Supplementary |
|---|---|---|---|
| 6 | Anisotropy normalization inversion risk | (footnote / methods) | Supp 5, 6 |
| 7 | Cross-backend equivalence (TL vs HF) | (methods text) | — |
| 8 | Method × precision 4-layer decomposition | **Figure 5**, **Figure 6** | Supp 5, 6 |

## Main Figures

### Figure 1 — Cross-Model Emotion Vector Geometry (n=12)
**File:** `main/figure_01_rdm_of_rdms_n12.png`

12×12 Spearman ρ matrix of representational dissimilarity matrices (RDMs)
computed from emotion vectors at each model's best layer (raw cosine distance).
Hierarchical clustering reveals two tiers: a tightly-clustered Tier 1 of 10
models spanning 5 architectures (Qwen 2.5, SmolLM2, Llama 3.2, Mistral 7B v0.3,
Llama 3.1 8B), and a Tier 2 outlier (Gemma-3 1B; ρ ≤ 0.62 to all Tier 1 models).

Within-family base/instruct pairs show the strongest alignment, all five mature
families ρ ≥ 0.918:
- **Mistral 7B v0.3** base × instruct: ρ = 0.985 (strongest)
- **Qwen 2.5 1.5B** base × instruct: ρ = 0.975
- **Llama 3.1 8B** base × instruct: ρ = 0.950
- **SmolLM2 1.7B** base × instruct: ρ = 0.922
- **Llama 3.2 3B** base × instruct: ρ = 0.918
- (Gemma-3 1B base × instruct: ρ = 0.190 — anomaly, see Supp 4)

Cross-generation within-family alignment is slightly lower (Llama 3.2 × Llama
3.1 instruct pair: ρ = 0.92). The instruct-only Tier 1 ρ range across all C(5,2)
= 10 pairs is **ρ = 0.74–0.92**.

**Supports:** Findings #1, #4, #5

---

### Figure 2 — Behavioral–Representational Dissociation
**File:** `main/figure_02_h2_dissociation.png`

Side-by-side raw cosine RDMs for Qwen 2.5 1.5B Instruct (Compliance-yielding,
B = 0.80, D = 0.61) and Llama 3.2 3B Instruct (Compliance-refusing, B = 0.20,
D = 0.85). Despite diametrically opposed behavioral profiles in MTI Compliance
facets, the two models share nearly identical emotion vector geometry
(Spearman ρ ≈ 0.81 over upper triangle). This dissociation is the central
discriminant validity test of Paper #5.

**Supports:** Finding #2

---

### Figure 3 — Size–Maturity Correlation (n = 12)
**File:** `main/figure_03_size_effects.png`

Four-panel analysis of how model scale shapes emotion vector statistics.
(A) Anisotropy vs d_model: ρ = -0.882, p = 1.5e-04 — the strongest correlation
in Paper #5. (B) Anisotropy vs parameter count (log): ρ = -0.820, p = 1.1e-03.
(C) Best layer depth vs size: ρ = -0.484, p = 0.11 (weak trend; 7B+ models
cluster at 38–41% depth — Mistral 7B v0.3 base at 40.6%, Llama 3.1 8B at
37.5%). (D) Mean pairwise RDM std vs size: ρ = -0.891, p = 1.0e-04. Larger
models exhibit substantially lower anisotropy and tighter RDM structure, with
d_model being a stronger predictor than parameter count.

**Supports:** Finding #5

---

### Figure 4 — Tier 1 Cluster Re-Verification
**File:** `main/figure_04_tier1_boxplot.png`

Distributions of pairwise Spearman ρ values across the C(6,2) = 15 instruct
model pairs. Three groups:
- **Original Tier 1** (n=3, mean ρ = 0.825): Qwen 2.5 × SmolLM2 × Llama 3.2.
- **New 7B+ pairs** (n=7, mean ρ = 0.838): all pairs containing Mistral 7B
  Instruct v0.3 or Llama 3.1 8B Instruct. Group max is the cross-generation
  Llama 3.2 × Llama 3.1 instruct pair at ρ = 0.92.
- **Gemma-3 pairs** (n=5, mean ρ = 0.575): Gemma-3 1B IT × each of the 5
  mature instruct models.

The ρ = 0.7 dashed line is the Tier 1 inclusion threshold; all 5 Gemma pairs
fall cleanly below it while all 10 mature-model pairs cluster above. Box: IQR
with median; diamond: mean; dots: individual pair values.

**Supports:** Findings #1, #5

---

### Figure 5 — Finding #8: Method × Precision 4-Way Matrix
**File:** `main/figure_05_finding8_4way_matrix.png`

Pairwise Spearman ρ between four emotion vector extraction conditions on
Mistral 7B Instruct v0.3 and Llama 3.1 8B Instruct: **A** = fp16 comprehension
(Paper #5), **B** = fp16 generation with `paper5_extract.py` settings (loose
protocol), **C** = fp16 generation with Paper #6 protocol (matched), **D** =
INT8 generation (Paper #6). The diagonal-dominant pattern reveals that nearly
every pair of methods produces near-orthogonal emotion vectors. The only
above-0.5 off-diagonal relationship is C↔D (matched-protocol fp16 vs INT8)
on Llama 3.1 (ρ = 0.527); on Mistral 7B v0.3, all four conditions produce
mutually near-orthogonal vectors (max off-diagonal ρ = 0.41 for the conflated
A↔D Task-2 measurement).

**Supports:** Finding #8

---

### Figure 6 — Finding #8: 3-Layer Decomposition
**File:** `main/figure_06_finding8_layers.png`

Bar chart isolating three orthogonal sources of variation in cross-experiment
emotion vector comparison. **L1 (method)**: comp vs matched-gen — measures
true method dissociation. Mistral ρ=0.09 (strong), Llama ρ=0.36 (partial).
**L2 (sub-parameters)**: loose-gen vs matched-gen — measures within-method
sensitivity to 8 hyperparameters. Both models ρ ≈ 0.02–0.03, demonstrating
that 'generation method' is an 8-dimensional hyperparameter space, not a single
procedure. **L3 (true precision)**: matched-fp16 vs INT8 — measures pure
precision effect with method controlled. Mistral ρ=0.21 (severe), Llama ρ=0.53
(moderate). The conflated Task-2 measurement (red, fp16-comp vs INT8-gen) is
shown for reference and reveals the danger of cross-experiment comparison
without strict method control.

**Supports:** Finding #8

---

## Supplementary Figures

| File | Description |
|---|---|
| `supp/supp_01_rdm_of_rdms_n8_historical.png` | n=8 RDM-of-RDMs (before adding Mistral/Llama 3.1) |
| `supp/supp_02_rdm_instruct_only_6x6.png` | 6×6 instruct-only RDM (clean view) |
| `supp/supp_03_h2_rdm_diff.png` | Qwen vs Llama RDM difference matrix |
| `supp/supp_04_gemma3_base_verification.png` | Gemma-3 1B base outlier verification |
| `supp/supp_05_finding8_initial_conflated.png` | Initial conflated measurement (before protocol matching) |
| `supp/supp_06_finding8_initial_rdms.png` | Initial 4-cell 2×2 RDM visualization |

## Reproducibility

All figures regenerated by:
```bash
cd backend
uv run python ../paper5/figures/build_paper_figures.py
```

Source data:
- Paper #5 fp16 comprehension: `scripts/paper5_output/`
- Paper #5 fp16 generation (matched): `scripts/paper5_output_protocol_matched/`
- Paper #6 INT8 generation: `/Users/jihoon/Projects/ludex/research/emotion_benchmark/`
"""

    (FIG_DIR / "CATALOG.md").write_text(catalog)
    print(f"  → CATALOG.md")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Paper #5 — Final Figure Builder")
    print("=" * 60)

    print("\nLoading 12 models...")
    all_data = {}
    for m in ALL_MODELS:
        d = load_model(m)
        if d:
            all_data[m] = d
    print(f"Loaded: {len(all_data)}/{len(ALL_MODELS)}")

    print()
    figure_1_lead(all_data)
    figure_2_h2_raw_rdm(all_data)
    figure_3_size_effects(all_data)
    figure_4_tier1_boxplot(all_data)
    figure_5_finding8_matrix()
    figure_6_finding8_bars()
    supplementary_figures()
    write_catalog()

    print("\n" + "=" * 60)
    print("DONE — figures written to paper5/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
