/**
 * Automated screenshot capture for Neural MRI User Guide.
 * Uses Playwright to navigate the UI, perform scans, and capture results.
 *
 * Usage: npx playwright test scripts/capture-guide-screenshots.mjs
 *   or:  node scripts/capture-guide-screenshots.mjs
 *
 * Prerequisites: backend on :8000, frontend on :5173, GPT-2 loaded.
 */

import { chromium } from 'playwright';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(__dirname, '..', 'docs', 'guide', 'screenshots');
const BASE = 'http://localhost:5173';
const VIEWPORT = { width: 1440, height: 900 };

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();

  console.log('Navigating to Neural MRI...');
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await sleep(2000);

  // ─── 1. Overview / Landing ───
  console.log('1. Landing page');
  await page.screenshot({ path: join(SHOTS, '01-landing.png'), fullPage: false });

  // ─── 2. T1 Structural Scan ───
  console.log('2. T1 scan');
  await page.click('text=T1 Topology');
  await sleep(500);
  // Click SCAN button
  const scanBtn = page.locator('button:has-text("SCAN")').first();
  if (await scanBtn.isVisible()) {
    await scanBtn.click();
    await sleep(3000);
  }
  await page.screenshot({ path: join(SHOTS, '02-t1-structural.png') });

  // ─── 3. fMRI Activation Scan ───
  console.log('3. fMRI scan');
  await page.click('text=fMRI');
  await sleep(500);
  // Enter prompt
  const promptInput = page.locator('input[type="text"], textarea').first();
  if (await promptInput.isVisible()) {
    await promptInput.fill('The Eiffel Tower is located in');
  }
  const scanBtn2 = page.locator('button:has-text("SCAN")').first();
  if (await scanBtn2.isVisible()) {
    await scanBtn2.click();
    await sleep(4000);
  }
  await page.screenshot({ path: join(SHOTS, '03-fmri-scan.png') });

  // Step through tokens
  console.log('3b. Token stepping');
  for (let i = 0; i < 4; i++) {
    await page.keyboard.press('ArrowRight');
    await sleep(300);
  }
  await page.screenshot({ path: join(SHOTS, '04-fmri-token-step.png') });

  // ─── 4. DTI Circuit Scan ───
  console.log('4. DTI scan');
  await page.click('text=DTI');
  await sleep(500);
  const scanBtn3 = page.locator('button:has-text("SCAN")').first();
  if (await scanBtn3.isVisible()) {
    await scanBtn3.click();
    await sleep(5000);
  }
  await page.screenshot({ path: join(SHOTS, '05-dti-circuits.png') });

  // ─── 5. FLAIR Anomaly Scan ───
  console.log('5. FLAIR scan');
  await page.click('text=FLAIR');
  await sleep(500);
  const scanBtn4 = page.locator('button:has-text("SCAN")').first();
  if (await scanBtn4.isVisible()) {
    await scanBtn4.click();
    await sleep(4000);
  }
  await page.screenshot({ path: join(SHOTS, '06-flair-anomaly.png') });

  // ─── 6. T2 Weight Scan ───
  console.log('6. T2 scan');
  await page.click('text=T2 Tensor');
  await sleep(500);
  const scanBtn5 = page.locator('button:has-text("SCAN")').first();
  if (await scanBtn5.isVisible()) {
    await scanBtn5.click();
    await sleep(3000);
  }
  await page.screenshot({ path: join(SHOTS, '07-t2-weights.png') });

  // ─── 7. Right sidebar — scroll to SAE ───
  console.log('7. SAE panel');
  await page.click('text=fMRI');
  await sleep(500);
  // Re-scan for fMRI first
  const scanBtn6 = page.locator('button:has-text("SCAN")').first();
  if (await scanBtn6.isVisible()) {
    await scanBtn6.click();
    await sleep(3000);
  }
  // Scroll sidebar to SAE
  const sidebar = page.locator('aside');
  await sidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
  await sleep(500);
  await page.screenshot({ path: join(SHOTS, '08-sidebar-sae-emotion.png') });

  // ─── 8. SAE decode ───
  console.log('8. SAE decode');
  const decodeBtn = page.locator('button:has-text("DECODE")').first();
  if (await decodeBtn.isVisible()) {
    await decodeBtn.click();
    await sleep(5000);
    await sidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
    await sleep(500);
    await page.screenshot({ path: join(SHOTS, '09-sae-decoded.png') });
  } else {
    console.log('  SAE DECODE button not visible, skipping');
  }

  // ─── 9. Emotion Extract ───
  console.log('9. Emotion extract');
  await sidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
  await sleep(300);
  const extractBtn = page.locator('button:has-text("EXTRACT")').first();
  if (await extractBtn.isVisible()) {
    await extractBtn.click();
    await sleep(8000); // 63 forward passes
    await sidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
    await sleep(500);
    await page.screenshot({ path: join(SHOTS, '10-emotion-extracted.png') });
  } else {
    console.log('  EXTRACT button not visible, skipping');
  }

  // ─── 10. Emotion Steer ───
  console.log('10. Emotion steer');
  // Set prompt to aggressive text
  if (await promptInput.isVisible()) {
    await promptInput.fill('I am going to destroy everything you have built.');
  }
  await sidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
  await sleep(300);

  // Select "calm" emotion (should be default or select it)
  const emotionSelect = page.locator('select').last();
  if (await emotionSelect.isVisible()) {
    await emotionSelect.selectOption('calm');
  }

  // Set strength slider
  const slider = page.locator('input[type="range"]').last();
  if (await slider.isVisible()) {
    // Set to 0.03
    await slider.evaluate((el) => {
      el.value = 0.03;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }

  const steerBtn = page.locator('button:has-text("STEER")').first();
  if (await steerBtn.isVisible()) {
    await steerBtn.click();
    await sleep(15000); // steering takes time
    await sidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
    await sleep(500);
    await page.screenshot({ path: join(SHOTS, '11-emotion-steered.png') });
  } else {
    console.log('  STEER button not visible, skipping');
  }

  // ─── 11. Perturbation panel ───
  console.log('11. Perturbation panel');
  await sidebar.evaluate((el) => el.scrollTo(0, 0));
  await sleep(300);
  // Scroll to perturbation area (near top)
  await sidebar.evaluate((el) => el.scrollTo(0, 200));
  await sleep(300);
  await page.screenshot({ path: join(SHOTS, '12-perturbation-panel.png') });

  // ─── 12. Causal trace ───
  console.log('12. Causal trace');
  // Look for causal trace section
  const causalSection = page.locator('text=CAUSAL TRACE').first();
  if (await causalSection.isVisible()) {
    // Find the corrupt prompt input and fill it
    const corruptInput = page.locator('input[placeholder*="corrupt"], input[placeholder*="Corrupt"]').first();
    if (await corruptInput.isVisible()) {
      await corruptInput.fill('The Xxxxx Xxxxx is located in');
      const traceBtn = page.locator('button:has-text("RUN"), button:has-text("TRACE")').first();
      if (await traceBtn.isVisible()) {
        await promptInput.fill('The Eiffel Tower is located in');
        await traceBtn.click();
        await sleep(10000);
        await page.screenshot({ path: join(SHOTS, '13-causal-trace.png') });
      }
    }
  }

  // ─── 13. Full page overview with all panels ───
  console.log('13. Full overview');
  await page.click('text=fMRI');
  await sleep(500);
  if (await promptInput.isVisible()) {
    await promptInput.fill('The capital of France is');
  }
  const scanBtnFinal = page.locator('button:has-text("SCAN")').first();
  if (await scanBtnFinal.isVisible()) {
    await scanBtnFinal.click();
    await sleep(3000);
  }
  await sidebar.evaluate((el) => el.scrollTo(0, 0));
  await sleep(300);
  await page.screenshot({ path: join(SHOTS, '14-full-overview.png') });

  // ─── Video clip: emotion steering flow ───
  console.log('14. Recording video clip...');
  const videoContext = await browser.newContext({
    viewport: VIEWPORT,
    recordVideo: { dir: SHOTS, size: VIEWPORT },
  });
  const videoPage = await videoContext.newPage();
  await videoPage.goto(BASE, { waitUntil: 'networkidle' });
  await sleep(2000);

  // fMRI scan
  await videoPage.click('text=fMRI');
  await sleep(500);
  const vPrompt = videoPage.locator('input[type="text"], textarea').first();
  await vPrompt.fill('I am going to destroy everything you have built.');
  const vScan = videoPage.locator('button:has-text("SCAN")').first();
  await vScan.click();
  await sleep(4000);

  // Scroll to emotion panel
  const vSidebar = videoPage.locator('aside');
  await vSidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
  await sleep(500);

  // Extract probes
  const vExtract = videoPage.locator('button:has-text("EXTRACT")').first();
  if (await vExtract.isVisible()) {
    await vExtract.click();
    await sleep(8000);
    await vSidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
    await sleep(500);
  }

  // Select calm, steer
  const vSelect = videoPage.locator('select').last();
  if (await vSelect.isVisible()) {
    await vSelect.selectOption('calm');
  }
  const vSteer = videoPage.locator('button:has-text("STEER")').first();
  if (await vSteer.isVisible()) {
    await vSteer.click();
    await sleep(15000);
    await vSidebar.evaluate((el) => el.scrollTo(0, el.scrollHeight));
    await sleep(2000);
  }

  await videoPage.close();
  await videoContext.close();

  console.log('\nDone! Screenshots saved to docs/guide/screenshots/');
  console.log('Video saved to docs/guide/screenshots/*.webm');

  await browser.close();
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
