/**
 * Case 4 — Llama-3.2-3B-Instruct causal trace only
 * Run with fresh server to avoid OOM
 */
const API = 'http://localhost:8000/api';

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function loadModel(modelId) {
  console.log(`\n  Loading ${modelId}...`);
  const res = await fetch(`${API}/model/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, device: 'auto' }),
    signal: AbortSignal.timeout(600000),
  });
  if (!res.ok) throw new Error(`Load failed: ${await res.text()}`);
  const info = await res.json();
  console.log(`  ✓ ${info.model_id} loaded (${info.n_layers} layers, ${info.device})`);
  await sleep(3000);
  return info;
}

async function runCausalTrace(cleanPrompt, corruptPrompt) {
  const res = await fetch(`${API}/perturb/causal-trace`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clean_prompt: cleanPrompt, corrupt_prompt: corruptPrompt, target_token_idx: -1 }),
    signal: AbortSignal.timeout(600000),
  });
  if (!res.ok) throw new Error(`${(await res.json().catch(() => ({}))).detail || res.statusText}`);
  return res.json();
}

(async () => {
  const fs = await import('fs');

  console.log('═'.repeat(60));
  console.log('Llama-3.2-3B-Instruct Causal Trace');
  console.log('═'.repeat(60));

  try {
    await loadModel('meta-llama/Llama-3.2-3B-Instruct');
    const trace = await runCausalTrace('The capital of France is', 'The capital of Poland is');
    console.log(`  ✓ Clean: "${trace.clean_prediction}", Corrupt: "${trace.corrupt_prediction}"`);
    const sorted = [...trace.cells].sort((a, b) => b.recovery_score - a.recovery_score);
    console.log('  Top 10 recovery:');
    for (const c of sorted.slice(0, 10)) {
      console.log(`    ${c.component.padEnd(20)} ${c.component_type.padEnd(6)} ${c.recovery_score.toFixed(4)}`);
    }
    fs.writeFileSync('scripts/case4-llama-trace.json', JSON.stringify(trace, null, 2));
    console.log('\n✓ Saved to scripts/case4-llama-trace.json');
  } catch (e) {
    console.log(`  ✗ FAILED: ${e.message}`);
  }
})();
