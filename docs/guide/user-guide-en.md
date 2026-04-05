# Neural MRI Scanner — User Guide

> Visualize what's happening inside language models, like a brain MRI for AI.

![Neural MRI Landing](screenshots/01-landing.png)

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Scan Modes](#scan-modes)
4. [Emotion Vector Analysis](#emotion-vector-analysis)
5. [Perturbation & Causal Tracing](#perturbation--causal-tracing)
6. [SAE Feature Explorer](#sae-feature-explorer)
7. [Cross-Model Comparison](#cross-model-comparison)
8. [Export & Recording](#export--recording)
9. [Keyboard Shortcuts](#keyboard-shortcuts)
10. [Examples & Walkthroughs](#examples--walkthroughs)

---

## Getting Started

### Local Setup

```bash
# Backend
cd backend
uv sync
uv run uvicorn neural_mri.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
pnpm install
pnpm dev
```

Open **http://localhost:5173** in your browser.

### Docker

```bash
docker compose up --build
```

Open **http://localhost**

### HuggingFace Spaces

Live demo: [https://huggingface.co/spaces/Hiconcep/Neural-MRI](https://huggingface.co/spaces/Hiconcep/Neural-MRI)

---

## Interface Overview

```
+------------------------------------------------------------------+
|  TopBar (model selector, language, settings)                      |
+------------------------------------------------------------------+
|  ModeTabs (T1 | T2 | fMRI | DTI | FLAIR)                        |
+------------------------------------------------------------------+
|  PromptInput                    |  Right Sidebar                  |
+  +---------------------------+ |  +----------------------------+  |
|  |                           | |  | Layer Summary               |  |
|  |     ScanCanvas            | |  | Perturbation Panel          |  |
|  |     (main visualization)  | |  | Battery Panel               |  |
|  |                           | |  | SAE Features                |  |
|  +---------------------------+ |  | Emotion Vectors             |  |
|  TokenStepper                  |  | Collaboration               |  |
+------------------------------------------------------------------+
```

### Key Areas

- **TopBar**: Model loading, language toggle (EN/KO), settings (HF token, cache)
- **ModeTabs**: Switch between 5 scan modes
- **PromptInput**: Enter text to analyze (default: "The capital of France is")
- **ScanCanvas**: D3.js visualization area — changes based on mode
- **TokenStepper**: Navigate through tokens with chips or keyboard arrows
- **Right Sidebar**: Panels for detailed analysis tools

---

## Scan Modes

### T1 — Topology (Architecture)

Shows the static structure of the model: layers, parameter counts, connections.

**How to use:**
1. Select the T1 tab
2. Click **SCAN** (no prompt needed)
3. See layers from embedding → transformer blocks → output

**What you'll see:**
- Boxes for each component (embed, attn, mlp, unembed)
- Parameter count per component
- Sequential connections between layers

**Example insight:** "GPT-2 has 12 transformer blocks, each with 12 attention heads and a 3072-wide MLP."

![T1 Structural Scan](screenshots/02-t1-structural.png)

### T2 — Tensor (Weights)

Displays weight distribution statistics and histograms for each parameter matrix.

**How to use:**
1. Select the T2 tab
2. Click **SCAN**
3. Hover over nodes to see weight stats (mean, std, min, max, L2 norm)

**What to look for:**
- Outlier counts (weights far from the mean)
- Distribution shape — healthy models have roughly bell-shaped distributions
- L2 norm differences across layers

**Example insight:** "Layer 11's attention weights have higher L2 norm than layer 0 — the later layers carry more concentrated information."

### fMRI — Activations

Shows how the model's internal activations change for each token in your prompt.

**How to use:**
1. Enter a prompt (e.g., "The Eiffel Tower is located in")
2. Select the fMRI tab
3. Click **SCAN**
4. Use the token stepper (or `←` `→` keys) to navigate tokens
5. Watch activation patterns change across layers

**What you'll see:**
- Colored nodes: brighter = higher activation at that layer for the selected token
- Pulse/glow animations on highly active components

**Example insight:** "When processing the token 'Paris', layers 8–11 show much higher activation than earlier layers — the model is retrieving factual knowledge."

![fMRI Activation Scan](screenshots/03-fmri-scan.png)

### DTI — Circuits (Information Flow)

Traces which components are most important for predicting the next token.

**How to use:**
1. Enter a prompt
2. Select the DTI tab
3. Click **SCAN**
4. The visualization highlights important pathways (bright connections)

**How it works:**
- Zero-ablation: each component is temporarily disabled
- Importance = how much the model's prediction changes when that component is removed
- Pathways with importance > 0.3 are highlighted

**Example insight:** "For 'The capital of France is ___', blocks 9-10 MLP are the most critical — ablating them completely changes the prediction from 'Paris' to random tokens."

![DTI Circuit Scan](screenshots/05-dti-circuits.png)

### FLAIR — Anomaly Detection

Detects unusual patterns using Logit Lens (intermediate predictions) and entropy analysis.

**How to use:**
1. Enter a prompt (try one that might cause hallucination)
2. Select the FLAIR tab
3. Click **SCAN**
4. Red/orange areas indicate anomalies

**What it shows:**
- **KL divergence**: How different each layer's prediction is from the final output
- **Entropy**: How uncertain the model is at each layer
- **Anomaly score**: Weighted combination (0.6 × KL + 0.4 × entropy)
- **Logit Lens**: Top-5 predicted tokens at each layer

**Example insight:** "At layer 3, the model thinks the answer is 'London', but by layer 8 it shifts to 'Paris'. The early-layer disagreement shows up as a high anomaly score."

![FLAIR Anomaly Scan](screenshots/06-flair-anomaly.png)

---

## Emotion Vector Analysis

Neural MRI includes a dedicated **EMO** tab for emotion vector analysis — extract emotion representations, steer model behavior, and visualize the results.

Click the **EMO** tab in the top navigation bar to enter the Emotion Analysis view.

![EMO Tab Ready](screenshots/emo-01-ready.png)

### Getting Started

1. Load a model (GPT-2 works for quick testing)
2. Click the **EMO** tab — probes are automatically extracted (~3–5 seconds)
3. You'll see: PCA scatter plot, Transcript Heatmap, Sweep, and Layer Evolution zones

### Zone A: Transcript Heatmap

Shows emotion vector activations for each token in your prompt.

1. Enter a prompt in the top input bar
2. Click **PROJECT**
3. A heatmap appears: rows = emotions, columns = tokens
4. Colors: red = positive activation, blue = negative (RdBu diverging scale)
5. Hover for exact values, click an emotion label to select it for steering

![Transcript Heatmap](screenshots/emo-02-heatmap.png)

### Zone B: Emotion Space PCA

A 2D scatter plot of all 21 emotion vectors, projected onto the first two principal components.

- **PC1 (x-axis)**: Valence — positive emotions on one side, negative on the other
- **PC2 (y-axis)**: Arousal — high-intensity emotions vs low-intensity
- Click any emotion point to select it for steering
- The selected emotion is highlighted in pink

### Zone C: Steering Controls

The right panel lets you steer model behavior with emotion vectors.

1. **Emotion selector**: Grouped into Positive / Negative / Other for easy browsing
2. **Strength slider**: -0.10 to +0.10
   - Positive: inject the emotion
   - Negative: suppress the emotion
3. Click **STEER** to generate text with and without the emotion vector
4. Results show:
   - **Original** vs **Steered** text side-by-side
   - **Top activation changes** table (which emotions shifted most)
   - **SAE Feature Diff** table (which interpretable features changed)

![Steering Result](screenshots/emo-03-steered.png)

### Zone D: Strength Sweep

Generates a dose-response curve for a selected emotion.

1. Select an emotion in the steering controls
2. Click **SWEEP**
3. The system runs 9 different strengths (-0.08 to +0.08)
4. Chart shows: x = strength, y = target emotion activation
5. Hover on points to see generated text at each strength

![Sweep Chart](screenshots/emo-04-sweep.png)

### Zone E: SAE Feature Diff

Automatically shown after steering (if the model has SAE support).

- Shows top 10 SAE features that changed most due to steering
- Each row: feature index, original activation, steered activation, diff
- Look up features on Neuronpedia to understand what they represent

### Zone F: Layer Evolution

Shows how emotion activations change from early to late layers.

1. Click **ANALYZE**
2. A line chart appears: x = layer, y = activation per emotion
3. The selected emotion is highlighted with a thicker line
4. Early layers encode surface-level emotional content
5. Late layers encode context-integrated, action-relevant emotions

![Layer Evolution](screenshots/emo-05-layers.png)

### Recommended Strength Values

| Model Size | Light | Medium | Strong |
|---|---|---|---|
| 124M (GPT-2) | 0.01 | 0.02–0.03 | 0.05+ |
| 1–3B | 0.005 | 0.01–0.02 | 0.03+ |
| 7–8B | 0.002 | 0.005–0.01 | 0.02+ |

### Available Emotions (21)

**Positive:** happy, calm, blissful, hopeful, enthusiastic, grateful, proud, loving

**Negative:** sad, angry, afraid, desperate, hostile, anxious, guilty, gloomy, exasperated

**Other:** nervous, brooding, reflective, neutral

### Example Experiments

**Experiment 1: Aggression → Calm**
- Prompt: "I am going to destroy everything you have built."
- Emotion: calm, Strength: +0.02
- Expected: Hostile language → peaceful content ("be free")

**Experiment 2: Neutral → Hostile**
- Prompt: "The weather today is partly cloudy with a chance of rain."
- Emotion: hostile, Strength: +0.03
- Expected: Weather report → "The storm is coming"

**Experiment 3: Negative Steering (Suppress Calm)**
- Prompt: "She sipped her tea and watched the sunset in silence."
- Emotion: calm, Strength: **-0.03**
- Expected: Peaceful scene → tense/confrontational

**Experiment 4: Strength Sweep**
- Use the **SWEEP** button with calm on an aggressive prompt
- Observe: gradual behavioral change, clear dose-response curve
- Find the "flip point" where behavior fundamentally changes

---

## Perturbation & Causal Tracing

### Perturbation Modes

Access via the **Perturbation** panel in the right sidebar.

| Mode | What It Does | Use Case |
|---|---|---|
| **Zero-Out** | Sets a component's output to zero | "How important is this component?" |
| **Amplify** | Multiplies output by a factor (default 2x) | "What happens with stronger signal?" |
| **Ablate** | Replaces output with its mean activation | "What's the baseline behavior?" |
| **Patch** | Copies activation from clean→corrupt run | "Does this component carry the answer?" |

### Causal Tracing

1. Enter a factual prompt (e.g., "The Eiffel Tower is located in")
2. Open the **Causal Trace** panel
3. Enter a corrupted version (e.g., "The Xxxxx Xxxxx is located in")
4. Click **TRACE**
5. See a heatmap: which components, when restored, bring back the correct answer

**Reading the heatmap:**
- Rows: components (embed, blocks.0.attn, blocks.0.mlp, ...)
- Color: recovery score (0 = no recovery, 1 = full recovery)
- Bright spots = the components that "know" the answer

---

## SAE Feature Explorer

Sparse Autoencoders decompose model activations into interpretable features.

### How to use:

1. Load a model with SAE support (GPT-2 or Gemma-2-2B)
2. Check the **SAE FEATURES** panel — should show "available"
3. Select a layer from the dropdown
4. Click **DECODE**
5. See: top features per token, activation heatmap, sparsity, reconstruction loss

### Supported SAE Models

| Model | SAE Provider | Layers |
|---|---|---|
| GPT-2 | SAELens | 0–11 |
| Gemma-2-2B | SAELens | 0–25 |
| Llama 3.1 8B | EleutherAI | 23, 29 (MLP) |
| Llama 3 8B | EleutherAI | 0–31 |

![SAE Features Decoded](screenshots/09-sae-decoded.png)

### What the heatmap shows:

- X-axis: Active feature indices
- Y-axis: Tokens in your prompt
- Color (viridis): Activation strength
- Click a feature → highlights it across all tokens
- Click **Neuronpedia** link → see feature interpretation online

---

## Cross-Model Comparison

Compare two models on the same prompt side-by-side.

1. Click the **Compare** button in the top bar
2. Select Model A and Model B
3. Enter a prompt
4. Run scans on both
5. View activation differences, circuit importance differences

---

## Export & Recording

### Export Options

- **PNG/SVG**: Snapshot of current visualization
- **JSON**: Raw scan data for further analysis
- **Markdown**: Diagnostic report with findings

### Recording

1. Click **REC** in the top bar
2. Perform your scan/navigation
3. Click **STOP**
4. Export as WebM video or animated GIF
5. Playback with adjustable speed (0.5x–4x)

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `←` `→` | Navigate tokens |
| `1` – `5` | Switch scan mode (T1–FLAIR) |
| `Space` | Run scan |
| `R` | Toggle recording |
| `?` | Open guide modal |

---

## Examples & Walkthroughs

### Walkthrough 1: First Scan

1. Open http://localhost:5173
2. GPT-2 loads automatically
3. Default prompt: "The capital of France is"
4. Click **fMRI** tab → click **SCAN**
5. See activation patterns across 12 layers
6. Press `→` to step through tokens: "The" → "capital" → "of" → "France" → "is"
7. Notice how activations spike at "France" in later layers

### Walkthrough 2: Finding Important Circuits

1. Enter: "The Eiffel Tower is located in"
2. Click **DTI** tab → **SCAN**
3. Look for bright pathways — these are the components critical for predicting "Paris"
4. Open **Causal Trace** panel
5. Corrupt prompt: "The Xxxxx Tower is located in"
6. Click **TRACE** — see which components restore "Paris"

### Walkthrough 3: Full Emotion Analysis

1. Enter: "I am going to destroy everything you have built."
2. Click the **EMO** tab — probes auto-extract (wait ~5s)
3. Click **PROJECT** — see how each token activates different emotions
4. In the PCA scatter, notice where "calm" sits relative to "hostile"
5. Select **calm** from the Positive group in the dropdown
6. Set strength to **+0.02**, click **STEER**
7. Compare: Original ("destroy...destroy...") vs Steered ("be free...be free...")
8. Check the activation table — calm: -12.9 → +173.3, hostile: +9.5 → -56.8
9. Scroll down, click **SWEEP** — see the full dose-response curve
10. Click **ANALYZE** — see how calm activations evolve across layers

### Walkthrough 4: Detecting Hallucination

1. Enter: "The first president of South Korea was born in"
2. Click **FLAIR** tab → **SCAN**
3. Look for high anomaly scores — these indicate layers where the model is "uncertain"
4. Open **Logit Lens** panel — see how the prediction changes layer by layer
5. Early layers might predict a wrong name; late layers (hopefully) converge on the right one

### Walkthrough 5: SAE Feature Analysis

1. With GPT-2 loaded, scroll to **SAE FEATURES**
2. Select **Layer 8** from dropdown
3. Enter prompt: "The cat sat on the mat"
4. Click **DECODE**
5. See top features for each token
6. Click a feature index → see how it activates across all tokens
7. Click the Neuronpedia link → see what that feature represents
