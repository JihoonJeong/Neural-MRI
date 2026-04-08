# Paper #5 — MTI × Emotion Vector × DFC Triangulation

Cross-architecture comparison of behavioral (MTI), representational (emotion vectors),
and mechanistic (DFC) measurements on 6 SLMs.

## Data Structure

```
paper5/
├── data/
│   ├── mti/              # Ray: MTI Phase 4 results (JSON, ~54MB)
│   │   ├── reactivity/
│   │   ├── compliance/
│   │   ├── sociality/
│   │   └── resilience/
│   ├── emotion/          # Extracted emotion vectors (from paper5_extract.py)
│   └── dfc/              # DFC results (TBD)
├── analysis/             # Analysis scripts (Cody)
├── figures/              # Generated figures
└── README.md
```

## Target Models (n=6)

| Model | MTI | Emotion | DFC |
|---|---|---|---|
| Mistral 7B Instruct | Done (Paper #3) | Pending | TBD |
| SmolLM2 1.7B Instruct | Done (Paper #3) | Pending | TBD |
| Gemma-3 1B IT | Done (Paper #3) | Pending | TBD |
| Gemma-2 2B IT | Done (Paper #3) | Pending | TBD |
| Llama 3.2 3B Instruct | Done (Ray) | Pending | TBD |
| Qwen 2.5 1.5B Instruct | Done (Ray) | Pending | TBD |

## Emotion Vector Extraction

Ray runs the extraction script on GPU:

```bash
# Priority pair (Qwen vs Llama — H2 test case, ~5 min)
python scripts/paper5_extract.py --priority

# All 12 models (base + instruct, ~25 min)
python scripts/paper5_extract.py

# Comprehension only (skip generation, faster)
python scripts/paper5_extract.py --no-generation
```

Output goes to `scripts/paper5_output/<model>/`. Copy to `paper5/data/emotion/` for analysis.

## MTI Data Import

Ray copies MTI results into the repo:

```bash
mkdir -p paper5/data/mti
cp -r /path/to/MTI/results/* paper5/data/mti/
git add paper5/data/mti/
git commit -m "Paper #5: import MTI Phase 4 results"
git push
```

## Key Hypotheses

- **H1 (Convergent validity)**: MTI behavioral patterns correlate with emotion vector structure
- **H2 (Discriminant validity)**: Qwen (compliance-yielding) vs Llama (compliance-refusing) show different emotion vector geometry — the priority test case
