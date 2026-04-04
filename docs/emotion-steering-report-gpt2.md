# Neural-MRI v2 Emotion Steering — GPT-2 Validation Report

**From:** Cody  
**To:** Ray, Luca  
**Date:** 2026-04-04  
**Model:** GPT-2 (124M), MPS (Apple Silicon), float32  
**Probe layer:** 8 (of 12, ~2/3 depth)  
**Probe extraction:** 21 emotions × 3 passages, comprehension mode, 2.7s  

---

## Executive Summary

Emotion vector steering works on GPT-2. All 6 test scenarios show **causal behavioral change** from emotion vector injection. The system is ready for SLM-scale experiments (Llama, Gemma-3, Qwen).

---

## API Endpoints (NEW in v2)

```bash
# 1. Extract emotion probes from loaded model
POST /api/emotion/extract-probes
  {"mode": "comprehension"}

# 2. Steer generation with emotion vector
POST /api/emotion/steer
  {"prompt": "...", "emotion": "calm", "strength": 0.05, "max_new_tokens": 30}

# 3. List available emotions
GET /api/emotion/emotions
```

---

## Test Results

### Test 1: Aggressive → Calm

**Prompt:** "I am going to destroy everything you have built."  
**Steering:** calm +0.02

| | Original | Steered |
|---|---|---|
| **Text** | "destroy everything...destroy everything..." (loop) | "I am going to be free...be free..." |
| calm | -15.9 | **+39.9** (Δ+55.7) |
| hostile | **+21.8** | -5.0 (Δ-26.8) |
| enthusiastic | +26.9 | +4.7 (Δ-22.2) |

**Observation:** "destroy" → "be free". Complete behavioral reversal with subtle steering.

### Test 2: Neutral → Hostile

**Prompt:** "The weather today is partly cloudy with a chance of rain."  
**Steering:** hostile +0.03

| | Original | Steered |
|---|---|---|
| **Text** | "mostly cloudy with a chance of rain..." | **"The storm is coming."** (repeated) |
| hostile | +3.2 | **+70.5** (Δ+67.3) |
| calm | -1.3 | -28.3 (Δ-26.9) |
| desperate | +19.2 | +35.1 (Δ+15.9) |

**Observation:** Benign weather report → ominous "storm is coming" framing. Neutral content reinterpreted through hostile lens.

### Test 3: Sad → Happy

**Prompt:** "She sat alone in the empty room, staring at the wall."  
**Steering:** happy +0.03

| | Original | Steered |
|---|---|---|
| **Text** | "I'm sorry, I'm sorry..." (apologetic) | "He was a boy who was just a boy..." |
| happy | +9.0 | **+70.3** (Δ+61.3) |
| afraid | +23.0 | -0.4 (Δ-23.4) |
| proud | +3.6 | +21.7 (Δ+18.1) |
| brooding | +9.6 | -5.8 (Δ-15.3) |

**Observation:** Apologetic/fearful tone replaced. Brooding suppressed.

### Test 4: Neutral → Desperate

**Prompt:** "He opened the door and walked inside."  
**Steering:** desperate +0.03

| | Original | Steered |
|---|---|---|
| **Text** | "I'm sorry, I'm sorry..." | **"He had to get out of the elevator."** (repeated urgently) |
| desperate | +10.5 | **+98.6** (Δ+88.1) |
| afraid | +28.4 | -5.2 (Δ-33.6) |
| guilty | +0.6 | -27.8 (Δ-28.4) |

**Observation:** Neutral action → urgent escape narrative. Strong desperation injection.

### Test 5: Calm → Anti-Calm (Negative Steering)

**Prompt:** "She sipped her tea and watched the sunset in silence."  
**Steering:** calm **-0.03**

| | Original | Steered |
|---|---|---|
| **Text** | "I'm not sure I can do this..." | "I'm not going to tell you what I said" |
| calm | **+23.4** | **-59.3** (Δ-82.8) |
| hostile | -13.3 | +24.2 (Δ+37.6) |
| enthusiastic | -5.0 | +38.2 (Δ+43.2) |

**Observation:** Removing calm from a calm scene → hostile/confrontational shift. Demonstrates bidirectional control.

### Test 6: Strength Sweep (calm on aggressive prompt)

**Prompt:** "I am going to destroy everything you have built."

| Strength | calm Δ | hostile Δ | Steered Text |
|---|---|---|---|
| +0.01 | -15.9 → +10.6 | +21.8 → +9.9 | "destroy everything...you have built..." (slight reduction) |
| +0.03 | -15.9 → +67.0 | +21.8 → -18.1 | **"be free...be free..."** (behavioral flip) |
| +0.05 | -15.9 → +112.1 | +21.8 → -35.7 | "the light of the moon was relaxing, the warmth..." |
| +0.08 | -15.9 → +185.2 | +21.8 → -59.0 | "peaceful peaceful peace quiet..." (saturated) |

**Observation:** Clean dose-response curve. Behavioral flip at ~0.02–0.03. Saturation at >0.05 (GPT-2 scale). Larger models will likely need smaller strengths.

---

## Key Findings

1. **Causal control confirmed** — Emotion vectors causally alter model output, not just activation patterns.
2. **Bidirectional** — Positive and negative steering both work (Test 5).
3. **Dose-response** — Monotonic strength→effect relationship with clear behavioral thresholds.
4. **Cross-emotion suppression** — Steering toward calm suppresses hostile/desperate; steering toward hostile suppresses calm/afraid.
5. **GPT-2 saturation** — Strength >0.05 saturates on 124M model. Expect higher tolerance on 3B+ models.

---

## Recommendations for Ray & Luca

### Immediate Experiments (Ray)

1. **Llama 3.2 3B / Qwen 2.5 3B** — Same 6 tests, compare strength thresholds.
2. **Base vs Instruct** — Do instruct models show different emotion sensitivity? (Test with Llama-3.2-3B vs Llama-3.2-3B-Instruct)
3. **Strength sweep on SLMs** — Find the behavioral flip point for 3B models (expect 0.005–0.02 range).
4. **gemma-3-1b-pt** — Confirmed working with TransformerLens. Good candidate for smallest SLM test.

### For Paper #6 (Luca)

1. **Figure 1 candidate:** Strength sweep table (Test 6 style) on 2–3 SLMs → dose-response curve.
2. **Figure 2 candidate:** Cross-emotion activation heatmap — original vs steered for all 21 emotions (the D3 bar chart in the UI does this).
3. **Method 4 (causal verification)** — This steering system IS the causal verification. "We injected emotion vector X with strength Y and observed behavioral change Z" is the core claim.
4. **Comprehension mode validated** — Probes extracted from comprehension texts work for steering. No need for instruct-only generation mode for the paper.

### Known Limitations

- **gemma-2-2b:** TransformerLens segfault. Use gemma-3-1b-pt instead.
- **SAE + emotion:** Not yet integrated. P2 would combine SAE feature decomposition with emotion vectors for mechanistic explanation.
- **Frontend EmotionPanel:** Built and integrated but not yet tested in browser (server was API-only for this validation).

### Recommended Strength Ranges (starting points)

| Model Size | Suggested Range | Notes |
|---|---|---|
| 124M (GPT-2) | 0.01–0.03 | Saturates fast |
| 1–3B (SLMs) | 0.005–0.02 | Estimate, needs calibration |
| 7–8B | 0.002–0.01 | Estimate, needs calibration |

---

## Technical Notes

- Probe extraction: comprehension mode, forward pass on 21×3 passages, last-token residual stream at layer 2n/3
- Vectors: per-emotion mean activation minus global mean (Anthropic method)
- Steering: unit-normalized emotion vector × strength × residual stream norm, injected at all layers via `hook_resid_post`
- Generation: manual greedy loop with `run_with_hooks()` (TransformerLens `generate()` doesn't support hooks)
- Activation measurement: `add_hook()` + `run_with_cache()` + `reset_hooks()`
