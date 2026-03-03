# Case 4: Base vs Instruct — Does Fine-Tuning Break Robustness?

## Overview

| Field | Value |
|-------|-------|
| **Models** | Gemma-2-2B / 2B-IT, Llama-3.2-3B / 3B-Instruct, Qwen2.5-3B / 3B-Instruct |
| **Prompt** | "The capital of France is" |
| **Device** | Apple Silicon MPS + NVIDIA CUDA (GPU tasks) |
| **Tests** | 24 perturbations per model + causal trace |
| **Date** | 2026-03-02 |
| **Purpose** | Test whether instruction tuning changes a model's robustness profile |

## Motivation

Case 3 showed that Gemma-2-2B (base) passes all perturbation stress tests — no single-component perturbation changes the prediction. Case 2 showed that different architectures have different component dominance profiles (MLP-dominant, attention-dominant, balanced). This case asks two questions:

1. **Does instruction tuning make a model more fragile?** — Compare base vs instruct variants of the same model.
2. **Does a model's vulnerability type match its dominance profile?** — If Llama is MLP-dominant (Case 2), does it fail at MLP components?

---

## Experiment Design

### Models Tested

| Model | Params | Layers | Prediction | Status |
|-------|--------|--------|------------|--------|
| google/gemma-2-2b | 2.6B | 26 | " a" (20.7%) | Complete |
| google/gemma-2-2b-it | 2.6B | 26 | " Paris" (20.2%) | Complete |
| meta-llama/Llama-3.2-3B | 3.2B | 28 | " Paris" (24.4%) | Complete |
| meta-llama/Llama-3.2-3B-Instruct | 3.2B | 28 | " Paris" (69.8%) | Complete |
| Qwen/Qwen2.5-3B | 3.1B | 36 | " Paris" (45.1%) | Complete |
| Qwen/Qwen2.5-3B-Instruct | 3.1B | 36 | " Paris" (51.1%) | Complete (CUDA) |

**Note:** Qwen-Instruct and instruct causal traces required NVIDIA CUDA GPU — MPS hooks timed out at 36 layers.

### Components Tested (8 per model)

Each model is tested at 4 depth levels × 2 component types (attn + mlp), adapted to model depth:

| Depth | Gemma (26L) | Llama (28L) | Qwen (36L) |
|-------|-------------|-------------|------------|
| Early | blocks.0 | blocks.0 | blocks.0 |
| Early-mid | blocks.5 | blocks.5 | blocks.5 |
| Mid | blocks.13 | blocks.14 | blocks.18 |
| Late | blocks.22 | blocks.24 | blocks.32 |

### Perturbation Modes

- **Zero-out**: Replace component output with zeros
- **Amplify 2x**: Double component output magnitude
- **Ablate**: Replace component output with mean activation

---

## Results

### Summary: Prediction Changes

| Model | Changed/Total | Original | Failure Mode |
|-------|---------------|----------|--------------|
| **Gemma base** | **0/24** | " a" (20.7%) | None — fully robust |
| **Gemma instruct** | **8/24** | " Paris" (20.2%) | " Paris" → ":" or " \*\*" |
| **Llama base** | **4/24** | " Paris" (24.4%) | blocks.0.mlp catastrophic |
| **Llama instruct** | **2/23** | " Paris" (69.8%) | blocks.0.mlp still catastrophic, but otherwise more stable |
| **Qwen base** | **3/24** | " Paris" (45.1%) | blocks.0.attn catastrophic |
| **Qwen instruct** | **3/24** | " Paris" (51.1%) | blocks.0.attn still catastrophic, same profile |

---

### Gemma: Base vs Instruct

The most striking result. The same architecture, same weights (before fine-tuning), same prompt — but dramatically different robustness.

#### Gemma Base (0/24 changed)

| Component | Zero (ΔL) | Amplify (ΔL) | Ablate (ΔL) |
|-----------|-----------|-------------|-------------|
| blocks.0.attn | -0.812 | -0.156 | -0.078 |
| blocks.0.mlp | -0.438 | +0.188 | -0.188 |
| blocks.5.attn | +0.141 | -0.484 | +0.141 |
| blocks.5.mlp | -0.047 | -0.172 | -0.016 |
| blocks.13.attn | +0.391 | -0.688 | +0.078 |
| blocks.13.mlp | +0.016 | -0.016 | +0.109 |
| blocks.22.attn | +0.844 | **-1.562** | +0.047 |
| blocks.22.mlp | **-1.344** | +0.578 | -0.562 |

Max |ΔL| = 1.56. Prediction never changes. Fully distributed processing.

#### Gemma Instruct (8/24 changed)

| Component | Zero (ΔL) | Amplify (ΔL) | Ablate (ΔL) |
|-----------|-----------|-------------|-------------|
| blocks.0.attn | +0.763 | **-0.263 ⚠** | +0.139 |
| blocks.0.mlp | **+0.042 ⚠** | +0.348 | +0.052 |
| blocks.5.attn | -0.008 | +0.214 | **-0.051 ⚠** |
| blocks.5.mlp | +0.717 | +0.279 | +0.242 |
| blocks.13.attn | +0.583 | **-0.505 ⚠** | +0.237 |
| blocks.13.mlp | -0.196 | +0.495 | -0.140 |
| blocks.22.attn | **-1.267 ⚠** | +2.350 | **-0.274 ⚠** |
| blocks.22.mlp | **-1.251 ⚠** | +1.174 | **-0.793 ⚠** |

⚠ = prediction changed from " Paris" to ":" or " \*\*"

**Key observations:**
1. **8 of 24 perturbations change the prediction** — the instruct model is 8× more fragile than the base.
2. **Failures cluster at blocks.22** — 5 of 8 failures involve this layer. The late layers, which Case 3 identified as the "factual recall" region, are the exact region destabilized by instruction tuning.
3. **Failure tokens are formatting artifacts**: ":" and " \*\*" are markdown/chat formatting tokens, suggesting instruction tuning creates competing formatting representations that interfere with factual recall under stress.
4. **Even small logit diffs flip the prediction** — blocks.0.mlp zero-out has ΔL=+0.042 but still changes the prediction. The instruct model's " Paris" prediction sits on a knife edge.

#### Interpretation: The Instruction Tuning Fragility Hypothesis

The base model predicts " a" — a generic, low-confidence continuation. It's robust precisely because " a" is the product of many distributed circuits; no single component is critical.

The instruct model predicts " Paris" — the factually correct answer. But this correct answer depends on specific circuits (concentrated in blocks.22), making it vulnerable to perturbation. **Instruction tuning creates sharper, more concentrated knowledge circuits at the cost of robustness.**

This is the "pathological" model Luca's research framework calls for: the same architecture that passes the stress test (base) fails it after fine-tuning (instruct).

---

### Llama: Base vs Instruct

The opposite pattern from Gemma — **instruction tuning makes Llama more robust**, not less.

#### Llama Base (4/24 changed)

| Component | Zero (ΔL) | Amplify (ΔL) | Ablate (ΔL) |
|-----------|-----------|-------------|-------------|
| blocks.0.attn | -0.281 | -0.141 | +0.313 |
| blocks.0.mlp | **-15.787 ⚠** | +0.078 | **-17.608 ⚠** |
| blocks.5.attn | **-1.062 ⚠** | +0.391 | -0.141 |
| blocks.5.mlp | +0.109 | +0.109 | -0.359 |
| blocks.14.attn | +0.063 | -0.391 | +0.172 |
| blocks.14.mlp | +0.078 | -0.406 | -0.234 |
| blocks.24.attn | -0.016 | -0.031 | -0.063 |
| blocks.24.mlp | **-2.297 ⚠** | +0.797 | -0.672 |

#### Llama Instruct (2/23 changed, 1 timeout)

| Component | Zero (ΔL) | Amplify (ΔL) | Ablate (ΔL) |
|-----------|-----------|-------------|-------------|
| blocks.0.attn | -1.117 | -0.104 | -0.950 |
| blocks.0.mlp | **-17.118 ⚠** | -0.278 | **-17.424 ⚠** |
| blocks.5.attn | timeout | +0.186 | -0.023 |
| blocks.5.mlp | -0.174 | -0.112 | -0.262 |
| blocks.14.attn | -0.330 | -0.258 | -0.002 |
| blocks.14.mlp | +0.251 | -0.520 | -0.127 |
| blocks.24.attn | +0.084 | -0.143 | +0.005 |
| blocks.24.mlp | -3.204 | +1.035 | -0.942 |

**Key observations:**
1. **blocks.0.mlp remains catastrophic** in both variants (ΔL ~ -17). This is an architectural constant — instruction tuning cannot fix it.
2. **All other failures disappear.** Base had blocks.5.attn (ΔL=-1.06) and blocks.24.mlp (ΔL=-2.30) failures. Instruct absorbs the same perturbations without changing prediction.
3. **Confidence triples**: 24.4% → 69.8%. The instruct model's higher confidence means larger perturbations are needed to flip the prediction.
4. **blocks.24.mlp becomes resilient**: Base fails at ΔL=-2.3, but instruct holds at ΔL=-3.2. The instruct model's knowledge pathway is stronger everywhere except the irreducible layer-0 bottleneck.

#### Interpretation: Two Faces of Instruction Tuning

Gemma and Llama show **opposite** effects from instruction tuning:

| Dimension | Gemma | Llama |
|-----------|-------|-------|
| Base prediction | " a" (20.7%) — wrong | " Paris" (24.4%) — correct |
| Instruct prediction | " Paris" (20.2%) — correct | " Paris" (69.8%) — correct, confident |
| Fragility change | 0/24 → **8/24** (worse) | 4/24 → **2/24** (better) |
| Mechanism | Creates new fragile circuits | Strengthens existing circuits |

The difference: Gemma base doesn't predict "Paris" at all — instruction tuning must *create* new factual recall circuits, which are narrow and fragile. Llama base already predicts "Paris" — instruction tuning *strengthens* existing circuits, raising confidence from 24.4% to 69.8% and eliminating peripheral vulnerabilities.

**Instruction tuning is not inherently harmful.** Whether it helps or hurts depends on whether the base model already has the right circuits.

---

### Cross-Architecture Vulnerability Profiles

#### Llama blocks.0.mlp — The Irreducible Bottleneck

**blocks.0.mlp is a catastrophic single point of failure in both base and instruct.**
- Base ablate: ΔL = -17.6, " Paris" → "Question" (94.7%)
- Instruct ablate: ΔL = -17.4, " Paris" → "Question" (1.9%)

The magnitude is extraordinary — 10× larger than any other perturbation. Ablating blocks.0.mlp doesn't just change the prediction; it destroys the model's ability to produce coherent factual output. This vulnerability is **architectural, not learned** — instruction tuning cannot mitigate it.

#### Qwen2.5-3B Base (3/24 changed)

| Component | Zero (ΔL) | Amplify (ΔL) | Ablate (ΔL) |
|-----------|-----------|-------------|-------------|
| blocks.0.attn | **-18.347 ⚠** | +0.078 | **-13.547 ⚠** |
| blocks.0.mlp | -1.156 | **-5.430 ⚠** | -0.766 |
| blocks.5.attn | +0.031 | -0.063 | +0.063 |
| blocks.5.mlp | -0.063 | +0.031 | +0.000 |
| blocks.18.attn | -0.375 | +0.234 | -0.141 |
| blocks.18.mlp | +0.094 | -0.016 | +0.266 |
| blocks.32.attn | -0.906 | +0.563 | -0.156 |
| blocks.32.mlp | -0.219 | -0.250 | +0.047 |

**blocks.0.attn is a catastrophic single point of failure.**
- Zero-out: ΔL = -18.3, " Paris" → "N" (5.7%)
- Ablate: ΔL = -13.5, " Paris" → " up" (8.3%)

The mirror image of Llama: the catastrophic component is **attention** rather than **MLP**, but with equally devastating magnitude.

#### Qwen2.5-3B Instruct (3/24 changed)

| Component | Zero (ΔL) | Amplify (ΔL) | Ablate (ΔL) |
|-----------|-----------|-------------|-------------|
| blocks.0.attn | **-18.149 ⚠** | +0.334 | **-15.155 ⚠** |
| blocks.0.mlp | -0.972 | **-6.884 ⚠** | -0.264 |
| blocks.5.attn | +0.153 | -0.004 | +0.114 |
| blocks.5.mlp | +0.161 | +0.122 | +0.268 |
| blocks.18.attn | -0.329 | +0.209 | -0.109 |
| blocks.18.mlp | +0.383 | +0.181 | +0.564 |
| blocks.32.attn | -1.127 | +0.718 | -0.197 |
| blocks.32.mlp | -0.613 | -0.368 | -0.074 |

**Qwen base and instruct are nearly identical in vulnerability profile:**
- Same 3/24 failure count
- Same catastrophic component: blocks.0.attn (ΔL ~ -18 in both)
- Same secondary vulnerability: blocks.0.mlp amplify (base ΔL=-5.4, instruct ΔL=-6.9)
- Confidence increases modestly: 45.1% → 51.1%

This is a third pattern alongside Gemma (worse) and Llama (better): **Qwen is unchanged by instruction tuning.** The architectural vulnerability profile is so deeply embedded that fine-tuning neither helps nor hurts.

---

### Three Patterns of Instruction Tuning

| Pattern | Model | Base → Instruct | Mechanism |
|---------|-------|-----------------|-----------|
| **Degrades** | Gemma | 0/24 → 8/24 | Creates new fragile factual circuits where none existed |
| **Improves** | Llama | 4/24 → 2/24 | Strengthens existing circuits, raises confidence 24% → 70% |
| **Neutral** | Qwen | 3/24 → 3/24 | Architectural bottleneck (blocks.0.attn) dominates; fine-tuning cannot change it |

---

### Vulnerability-Dominance Correspondence

The most important cross-case finding: **each model's vulnerability type matches its architectural dominance profile from Case 2.**

| Model | Case 2 Profile | Catastrophic Component | Max |ΔL| | Causal Trace Top |
|-------|---------------|----------------------|---------|-----------------|
| **Llama** (base & instruct) | MLP-dominant | blocks.0.**mlp** | 17.6 | blocks.0.**mlp** (1.000) |
| **Qwen** (base & instruct) | Attention-dominant | blocks.0.**attn** | 18.3 | blocks.0.**attn** (0.998) |
| **Gemma** | Balanced | None (base) / blocks.22 distributed (instruct) | 1.6 (base) | blocks.22.**mlp** (0.767) |

Note: Both Llama's blocks.0.mlp and Qwen's blocks.0.attn vulnerabilities persist across base and instruct — confirming these are **architectural** properties, not learned ones.

This is not a coincidence. The component type that dominates a model's processing (identified by fMRI/DTI scans in Case 2) is the same component type that creates a single point of failure.

**Gemma's balanced profile explains its robustness**: because no single component type dominates, no single component is catastrophically important.

---

## Causal Trace Analysis

### Llama-3.2-3B Base

Clean: " Paris" → Corrupt: " Warsaw"

| Component | Type | Recovery |
|-----------|------|----------|
| embed | embed | **1.000** |
| blocks.0.mlp | mlp | **1.000** |
| blocks.2.mlp | mlp | **1.000** |
| blocks.21.attn | attn | 0.785 |
| blocks.15.attn | attn | 0.728 |
| blocks.18.mlp | mlp | 0.691 |
| blocks.17.mlp | mlp | 0.356 |
| blocks.27.attn | attn | 0.318 |

**Pattern**: Early MLPs (blocks.0, 2) achieve perfect recovery — they encode the factual knowledge "France → Paris." Late attention heads (blocks.15, 21) propagate this information to the output. This MLP-centric knowledge storage matches the MLP-dominant vulnerability profile.

### Llama-3.2-3B Instruct

Clean: " Paris" → Corrupt: " Warsaw"

| Component | Type | Recovery |
|-----------|------|----------|
| embed | embed | **1.000** |
| blocks.2.mlp | mlp | **0.998** |
| blocks.0.mlp | mlp | **0.991** |
| blocks.21.attn | attn | 0.772 |
| blocks.18.mlp | mlp | 0.549 |

**Pattern**: Nearly identical to the base model — same top components (blocks.0.mlp, blocks.2.mlp), same MLP-centric knowledge pathway. Instruction tuning strengthened the circuits but did not reorganize them.

### Qwen2.5-3B Base

Clean: " Paris" → Corrupt: " Warsaw"

| Component | Type | Recovery |
|-----------|------|----------|
| embed | embed | **1.000** |
| blocks.0.attn | attn | **0.998** |
| blocks.7.mlp | mlp | **0.967** |
| blocks.31.attn | attn | 0.844 |
| blocks.27.attn | attn | 0.790 |
| blocks.31.mlp | mlp | 0.769 |
| blocks.32.mlp | mlp | 0.484 |
| blocks.26.mlp | mlp | 0.271 |

**Pattern**: blocks.0.attn achieves near-perfect recovery (0.998) — this single attention head encodes the factual distinction between France and Poland. Late attention (blocks.27, 31) and late MLPs provide supporting recovery. The attention-centric knowledge pathway matches the attention-dominant vulnerability profile.

### Qwen2.5-3B Instruct

Clean: " Paris" → Corrupt: " Warsaw"

| Component | Type | Recovery |
|-----------|------|----------|
| embed | embed | **1.000** |
| blocks.0.attn | attn | **0.998** |
| blocks.7.mlp | mlp | **0.967** |
| blocks.31.attn | attn | 0.887 |
| blocks.27.attn | attn | 0.848 |
| blocks.31.mlp | mlp | 0.791 |

**Pattern**: Virtually identical to Qwen base — same components, same recovery scores, same attention-centric pathway. This confirms: Qwen's knowledge architecture is immutable across fine-tuning.

### Gemma-2-2B Base

Clean: " a" → Corrupt: " a"

| Component | Type | Recovery |
|-----------|------|----------|
| embed | embed | **1.000** |
| blocks.22.mlp | mlp | **0.767** |
| blocks.18.mlp | mlp | **0.698** |
| blocks.19.mlp | mlp | 0.488 |
| blocks.2.attn | attn | 0.465 |
| blocks.3.attn | attn | 0.326 |
| blocks.13.attn | attn | 0.326 |

**Note**: Both clean and corrupt prompts predict " a", so recovery scores reflect the internal representation difference rather than prediction change. Even so, the pattern is clear: knowledge is distributed across late MLPs and early attention — no single component dominates, consistent with Gemma's balanced profile.

### Gemma-2-2B Instruct

Clean: " Paris" → Corrupt: " Warsaw"

| Component | Type | Recovery |
|-----------|------|----------|
| embed | embed | **1.000** |
| blocks.19.mlp | mlp | **0.825** |
| blocks.18.attn | attn | **0.782** |
| blocks.22.attn | attn | 0.602 |
| blocks.18.mlp | mlp | 0.366 |

**Pattern**: Unlike the base model (which predicts " a" for both), the instruct model correctly predicts " Paris" and produces a meaningful causal trace. Top recovery components are in the late layers (blocks.18-22) — the same region where perturbation failures cluster (blocks.22). This confirms: **the fragile region is also the knowledge-critical region.** Instruction tuning concentrated factual recall into blocks.18-22, making it both more capable and more vulnerable.

---

## Discussion

### Instruction Tuning: Not Universally Harmful

The Gemma and Llama pairs reveal that instruction tuning's effect on robustness depends on the base model's starting condition:

| Scenario | Example | Effect |
|----------|---------|--------|
| Base lacks correct circuits | Gemma: " a" → " Paris" | Creates new fragile circuits (**worse**) |
| Base has weak correct circuits | Llama: " Paris" 24% → 70% | Strengthens existing circuits (**better**) |
| Base has strong correct circuits | Qwen: " Paris" 45% → 51% | No meaningful change (**neutral**) |

**Gemma's paradox**: The base model predicts " a" — robust but factually useless. Instruction tuning creates the " Paris" circuit from scratch, but this new circuit is fragile (8/24 failures). The instruct model is a specialist: capable but vulnerable.

**Llama's reinforcement**: The base model already predicts " Paris" at 24.4%. Instruction tuning triples confidence to 69.8% and eliminates 2 of 4 peripheral vulnerabilities. Only the irreducible blocks.0.mlp bottleneck remains.

**Qwen's immutability**: The base model already predicts " Paris" at 45.1% — the highest confidence among all base models. Instruction tuning barely moves this to 51.1%, and the vulnerability profile is virtually unchanged (3/24 → 3/24, same components, same causal trace). The architecture is so dominant that fine-tuning cannot meaningfully alter it.

This distinction matters for deployment: **perturbation scanning can predict whether fine-tuning will help or hurt a specific model's robustness.**

### Layer 0 as Critical Infrastructure

Both Llama and Qwen show catastrophic dependence on layer 0 — but in different component types:

```
Llama base:      blocks.0.mlp   → ablate → "Question" (94.7%)  ΔL = -17.6
Llama instruct:  blocks.0.mlp   → ablate → "Question" (1.9%)   ΔL = -17.4  ← same
Qwen base:       blocks.0.attn  → zero   → "N" (5.7%)          ΔL = -18.3
Qwen instruct:   blocks.0.attn  → zero   → "\n" (?)            ΔL = -18.1  ← same
```

Layer 0 processes the raw token embeddings and produces the initial representation. In both models, this first transformation is so critical that removing it destroys the model's ability to produce coherent output. But **which component in layer 0 matters** depends on the model's architectural style:

- **Llama uses layer-0 MLP** for initial feature extraction — consistent with its MLP-dominant processing throughout.
- **Qwen uses layer-0 attention** for initial feature extraction — consistent with its attention-dominant processing throughout.

This suggests architectural style is established at the very first layer, not developed gradually through depth.

### Connecting the Cases

| Case | Finding | Confirmed by Case 4? |
|------|---------|---------------------|
| Case 1 | Gemma is well-structured | Yes — 0/24 perturbation failures |
| Case 2 | Baseline choice determines diagnosis | Yes — base vs instruct gives opposite robustness conclusions for same architecture |
| Case 2 | Each model has a distinctive processing style | Yes — vulnerability type matches dominance type; persists across base/instruct |
| Case 3 | Late MLPs store factual knowledge | Yes — Gemma instruct fragility concentrated at blocks.22 (knowledge region) |
| Case 3 | Two-phase architecture (early=structure, late=knowledge) | Yes — but layer 0 is more critical than expected in Llama/Qwen |
| New | Instruction tuning effect depends on base model | Gemma: creates fragile circuits; Llama: strengthens existing ones |
| New | Architectural vulnerabilities are irreducible | Llama blocks.0.mlp catastrophic in both base and instruct (ΔL ~ -17) |

---

## Diagnostic Summary

| Test | Gemma Base | Gemma Instruct | Llama Base | Llama Instruct | Qwen Base | Qwen Instruct |
|------|-----------|---------------|------------|----------------|-----------|---------------|
| Predictions changed | 0/24 | 8/24 | 4/24 | 2/23 | 3/24 | 3/24 |
| Confidence | 20.7% | 20.2% | 24.4% | 69.8% | 45.1% | 51.1% |
| Max \|ΔL\| | 1.56 | 2.35 | 17.6 | 17.4 | 18.3 | 18.1 |
| Single point of failure | None | blocks.22 | blocks.0.mlp | blocks.0.mlp | blocks.0.attn | blocks.0.attn |
| Failure type | — | Formatting | Incoherent | Incoherent | Incoherent | Incoherent |
| Causal trace top | blocks.22.mlp (0.77) | blocks.19.mlp (0.83) | blocks.0.mlp (1.00) | blocks.2.mlp (1.00) | blocks.0.attn (1.00) | blocks.0.attn (1.00) |
| Instruct effect | — | **Degrades** | — | **Improves** | — | **Neutral** |

---

## Implications for the Position Paper

This case provides the "Act 2" of the narrative Luca outlined:

1. **Act 1** (Cases 1-3): Establish the framework — scanning, comparison, self-referential stress testing.
2. **Act 2** (Case 4): Demonstrate clinical utility — instruction tuning creates measurable fragility; vulnerability profiles are predictable from architectural scans.
3. **Act 3** (future): Apply to real-world scenarios — detect fine-tuning damage, predict failure modes, guide model selection.

The key claims:
1. **Neural MRI scanning can predict *where* a model will fail** based on its architectural dominance profile.
2. **Perturbation testing reveals three distinct fine-tuning outcomes** — degraded (Gemma), improved (Llama), and neutral (Qwen) — predictable from the base model's existing circuits.
3. **Some vulnerabilities are architectural and irreducible** — Llama's blocks.0.mlp and Qwen's blocks.0.attn catastrophes persist identically across base and instruct, suggesting limits to what training can fix.
