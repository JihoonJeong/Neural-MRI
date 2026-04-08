"""Paper #5 Finding #7: Precision Confound — fp16 vs 4-bit NF4 comparison.

Compares emotion vectors extracted at fp16 (cloud GPU) vs 4-bit NF4 (Paper #6).
Run on Mac after cloud extraction is pushed.

Usage:
    python paper5/analysis/precision_confound/compare_fp16_vs_nf4.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FP16_DIR = ROOT / "scripts" / "paper5_output"
# Paper #6 NF4 data location (adjust if needed)
NF4_DIR = Path("/Users/jihoon/Projects/ludex/research/emotion_benchmark")

MODELS = {
    "Mistral 7B Instruct": {
        "fp16": "mistralai_Mistral-7B-Instruct-v0.3",
        "nf4_gen": "emotion_pilot_v2_output",  # Paper #6 generation vectors
        "nf4_comp": "comprehension_output/comprehension_mistralai_Mistral-7B-Instruct-v0.3.json",
    },
}


def load_fp16_vectors(model_dir: str) -> dict[str, np.ndarray] | None:
    """Load fp16 vectors from paper5_extract output."""
    d = FP16_DIR / model_dir
    if not (d / "vectors_comprehension.pt").exists():
        return None
    meta = json.loads((d / "metadata.json").read_text())
    pt = torch.load(
        d / "vectors_comprehension.pt", map_location="cpu", weights_only=False
    )
    best = str(meta["best_layer"])
    return {e: v.numpy() for e, v in pt["vectors"][best].items()}


def raw_rdm(vectors: dict[str, np.ndarray]):
    emos = sorted(vectors.keys())
    mat = np.stack([vectors[e] for e in emos])
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1
    cos = (mat / n) @ (mat / n).T
    return 1 - cos, emos


def upper_tri(m):
    return m[np.triu(np.ones(m.shape, dtype=bool), k=1)]


def main():
    print("=" * 60)
    print("Paper #5 Finding #7: fp16 vs NF4 Precision Confound")
    print("=" * 60)

    for name, paths in MODELS.items():
        print(f"\n--- {name} ---")

        fp16_vecs = load_fp16_vectors(paths["fp16"])
        if fp16_vecs is None:
            print(f"  fp16 data not found at {FP16_DIR / paths['fp16']}")
            print("  Run cloud extraction first.")
            continue

        print(f"  fp16: {len(fp16_vecs)} emotions loaded")

        # TODO: Load NF4 vectors from Paper #6 data
        # The exact format depends on what's in ludex/emotion_benchmark
        # This placeholder shows the comparison logic
        nf4_comp_path = NF4_DIR / paths["nf4_comp"]
        if nf4_comp_path.exists():
            nf4_data = json.loads(nf4_comp_path.read_text())
            print(f"  NF4 comprehension: {nf4_data.get('mean_pairwise_cosine', 'N/A')}")
            print("  (NF4 has summary only, not raw vectors — need Ray to re-extract)")
        else:
            print(f"  NF4 data not found at {nf4_comp_path}")

        # When both are available:
        # rdm_fp16, emos = raw_rdm(fp16_vecs)
        # rdm_nf4, _ = raw_rdm(nf4_vecs)
        # rho, p = stats.spearmanr(upper_tri(rdm_fp16), upper_tri(rdm_nf4))
        # print(f"  RDM ρ (fp16 vs NF4): {rho:.4f}")

    print("\n  Note: Full comparison requires NF4 raw vectors from Paper #6.")
    print("  Ray needs to re-extract with vector saving, or we use the")
    print("  fp16 extraction as the clean baseline for Paper #5.")


if __name__ == "__main__":
    main()
