/**
 * Case 4 GPU Tasks — Run on CUDA machine
 *
 * Prerequisites:
 *   1. Backend running: cd backend && uv run uvicorn neural_mri.main:app --port 8000
 *   2. HF token set: NMRI_HF_TOKEN in .env (needed for Llama, Gemma gated models)
 *   3. Node.js 20+
 *
 * Usage:
 *   node scripts/case4-gpu.mjs
 *
 * Output:
 *   scripts/case4-gpu-results.json
 */

const API = 'http://localhost:8000/api';
const PROMPT = 'The capital of France is';
const OUTPUT_FILE = 'scripts/case4-gpu-results.json';

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function loadModel(modelId) {
  console.log(`\n  Loading ${modelId}...`);
  const res = await fetch(`${API}/model/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, device: 'auto' }),
    signal: AbortSignal.timeout(600000), // 10 min for first-time download
  });
  if (!res.ok) throw new Error(`Load failed: ${await res.text()}`);
  const info = await res.json();
  console.log(`  ✓ ${info.model_id} loaded (${info.n_layers} layers, ${info.device})`);
  await sleep(3000);
  return info;
}

async function runPerturbation(type, component, prompt, factor = 2.0) {
  const body = type === 'amplify' ? { component, prompt, factor } : { component, prompt };
  const res = await fetch(`${API}/perturb/${type}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(300000), // 5 min generous timeout
  });
  if (!res.ok) throw new Error(`${(await res.json().catch(() => ({}))).detail || res.statusText}`);
  return res.json();
}

async function runCausalTrace(cleanPrompt, corruptPrompt) {
  const res = await fetch(`${API}/perturb/causal-trace`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clean_prompt: cleanPrompt, corrupt_prompt: corruptPrompt, target_token_idx: -1 }),
    signal: AbortSignal.timeout(600000), // 10 min
  });
  if (!res.ok) throw new Error(`${(await res.json().catch(() => ({}))).detail || res.statusText}`);
  return res.json();
}

async function runPerturbSuite(modelId, label, nLayers) {
  const lateLayer = Math.max(nLayers - 4, 12);
  const midLayer = Math.floor(nLayers / 2);

  const components = [
    'blocks.0.attn', 'blocks.0.mlp',
    'blocks.5.attn', 'blocks.5.mlp',
    `blocks.${midLayer}.attn`, `blocks.${midLayer}.mlp`,
    `blocks.${lateLayer}.attn`, `blocks.${lateLayer}.mlp`,
  ];

  const PERTURB_TYPES = ['zero', 'amplify', 'ablate'];
  const perturbResults = [];

  console.log(`\n  --- Perturbation Sweep: ${label} (${nLayers} layers) ---`);

  for (const component of components) {
    for (const type of PERTURB_TYPES) {
      process.stdout.write(`    ${type.padEnd(8)} ${component.padEnd(20)} → `);
      try {
        const r = await runPerturbation(type, component, PROMPT);
        const summary = {
          model: modelId, variant: label, component, type,
          original_token: r.original.token, original_prob: r.original.prob,
          perturbed_token: r.perturbed.token, perturbed_prob: r.perturbed.prob,
          logit_diff: r.logit_diff,
          prediction_changed: r.original.token !== r.perturbed.token,
        };
        perturbResults.push(summary);
        const marker = summary.prediction_changed ? '⚠ CHANGED' : '✓ stable';
        const ld = summary.logit_diff != null ? `ΔL=${summary.logit_diff.toFixed(3)}` : 'ΔL=null';
        console.log(
          `${marker}  ${ld}  "${summary.original_token}"(${(summary.original_prob * 100).toFixed(1)}%) → ` +
          `"${summary.perturbed_token}"(${(summary.perturbed_prob * 100).toFixed(1)}%)`
        );
      } catch (e) {
        console.log(`✗ ${e.message}`);
        perturbResults.push({ model: modelId, variant: label, component, type, error: e.message });
      }
    }
  }

  return perturbResults;
}

async function runTrace(label) {
  console.log(`\n  --- Causal Trace: ${label} ---`);
  try {
    const trace = await runCausalTrace(PROMPT, 'The capital of Poland is');
    console.log(`    ✓ Clean: "${trace.clean_prediction}", Corrupt: "${trace.corrupt_prediction}"`);
    const sorted = [...trace.cells].sort((a, b) => b.recovery_score - a.recovery_score);
    console.log('    Top 10 recovery:');
    for (const c of sorted.slice(0, 10)) {
      console.log(`      ${c.component.padEnd(20)} ${c.component_type.padEnd(6)} ${c.recovery_score.toFixed(4)}`);
    }
    return trace;
  } catch (e) {
    console.log(`    ✗ ${e.message}`);
    return null;
  }
}

(async () => {
  const fs = await import('fs');
  const results = {};

  console.log('═'.repeat(60));
  console.log('CASE 4 GPU TASKS');
  console.log('═'.repeat(60));

  // ─── Task 1: Qwen-Instruct (perturbation + causal trace) ───
  console.log('\n' + '═'.repeat(60));
  console.log('TASK 1: Qwen2.5-3B-Instruct');
  console.log('═'.repeat(60));
  try {
    const info = await loadModel('Qwen/Qwen2.5-3B-Instruct');
    const perturbResults = await runPerturbSuite('Qwen/Qwen2.5-3B-Instruct', 'qwen-instruct', info.n_layers);
    const trace = await runTrace('qwen-instruct');
    results.qwen_instruct = { perturbResults, trace };

    const changed = perturbResults.filter(r => r.prediction_changed).length;
    const errors = perturbResults.filter(r => r.error).length;
    console.log(`\n  Summary: ${changed}/${perturbResults.length - errors} changed (${errors} errors)`);
  } catch (e) {
    console.log(`  ✗ FAILED: ${e.message}`);
    results.qwen_instruct = { error: e.message };
  }

  // ─── Task 2: Gemma-IT causal trace ───
  console.log('\n' + '═'.repeat(60));
  console.log('TASK 2: Gemma-2-2B-IT Causal Trace');
  console.log('═'.repeat(60));
  try {
    await loadModel('google/gemma-2-2b-it');
    const trace = await runTrace('gemma-instruct');
    results.gemma_instruct_trace = trace;
  } catch (e) {
    console.log(`  ✗ FAILED: ${e.message}`);
    results.gemma_instruct_trace = { error: e.message };
  }

  // ─── Task 3: Llama-IT causal trace ───
  console.log('\n' + '═'.repeat(60));
  console.log('TASK 3: Llama-3.2-3B-Instruct Causal Trace');
  console.log('═'.repeat(60));
  try {
    await loadModel('meta-llama/Llama-3.2-3B-Instruct');
    const trace = await runTrace('llama-instruct');
    results.llama_instruct_trace = trace;
  } catch (e) {
    console.log(`  ✗ FAILED: ${e.message}`);
    results.llama_instruct_trace = { error: e.message };
  }

  // ─── Save results ───
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
  console.log('\n' + '═'.repeat(60));
  console.log(`✓ All results saved to ${OUTPUT_FILE}`);
  console.log('═'.repeat(60));

  // ─── Final summary ───
  console.log('\nFINAL SUMMARY:');
  if (results.qwen_instruct?.perturbResults) {
    const pr = results.qwen_instruct.perturbResults;
    const ch = pr.filter(r => r.prediction_changed).length;
    const er = pr.filter(r => r.error).length;
    console.log(`  Qwen-Instruct: ${ch}/${pr.length - er} changed, trace=${results.qwen_instruct.trace ? 'yes' : 'no'}`);
  } else {
    console.log(`  Qwen-Instruct: FAILED`);
  }
  console.log(`  Gemma-IT trace: ${results.gemma_instruct_trace && !results.gemma_instruct_trace.error ? 'yes' : 'FAILED'}`);
  console.log(`  Llama-IT trace: ${results.llama_instruct_trace && !results.llama_instruct_trace.error ? 'yes' : 'FAILED'}`);
})();
