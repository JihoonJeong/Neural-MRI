export type Locale = 'en' | 'ko';

export type TranslationKey =
  // Mode tooltips
  | 'tooltip.T1'
  | 'tooltip.T2'
  | 'tooltip.fMRI'
  | 'tooltip.DTI'
  | 'tooltip.FLAIR'
  // Guide
  | 'guide.title'
  | 'guide.overview.title'
  | 'guide.overview.body'
  | 'guide.T1.title'
  | 'guide.T1.body'
  | 'guide.T2.title'
  | 'guide.T2.body'
  | 'guide.fMRI.title'
  | 'guide.fMRI.body'
  | 'guide.DTI.title'
  | 'guide.DTI.body'
  | 'guide.FLAIR.title'
  | 'guide.FLAIR.body'
  | 'guide.perturbation.title'
  | 'guide.perturbation.body'
  | 'guide.shortcuts.title'
  | 'guide.shortcuts.body'
  // Report modal
  | 'report.title'
  | 'report.technique'
  | 'report.findings'
  | 'report.impression'
  | 'report.recommendation'
  | 'report.patient'
  | 'report.date'
  | 'report.prompt'
  | 'report.button'
  | 'report.generating'
  // Battery
  | 'battery.title'
  | 'battery.detailTitle'
  | 'battery.run'
  | 'battery.running'
  | 'battery.passed'
  | 'battery.failed'
  | 'battery.total'
  | 'battery.viewDetails'
  | 'battery.noResult'
  | 'battery.peak'
  | 'battery.activeLayers'
  | 'battery.compare'
  | 'battery.model'
  | 'battery.includeSae'
  | 'battery.autoLayer'
  | 'battery.saeFeatures'
  | 'battery.crossTestSae'
  | 'report.explain'
  // Compare
  | 'compare.title'
  | 'compare.promptA'
  | 'compare.promptB'
  | 'compare.diff'
  | 'compare.scan'
  | 'compare.stronger'
  // Export
  | 'export.button'
  | 'export.png'
  | 'export.svg'
  | 'export.json'
  | 'export.report'
  | 'export.gif'
  | 'export.webm'
  // Recording
  | 'recording.rec'
  | 'recording.stop'
  | 'recording.play'
  | 'recording.pause'
  | 'recording.save'
  | 'recording.load'
  | 'recording.speed'
  // SAE
  | 'sae.title'
  | 'sae.noSae'
  | 'sae.layer'
  | 'sae.scan'
  | 'sae.scanning'
  | 'sae.features'
  | 'sae.heatmap'
  | 'sae.reconstruction'
  | 'sae.sparsity'
  | 'sae.neuronpedia'
  // Layout
  | 'layout.vertical'
  | 'layout.brain'
  | 'layout.network'
  | 'layout.radial'
  // Cross-model
  | 'crossModel.title'
  | 'crossModel.selectModel'
  | 'crossModel.compare'
  // Causal Trace
  | 'causalTrace.title'
  | 'causalTrace.clean'
  | 'causalTrace.corruptPlaceholder'
  | 'causalTrace.trace'
  | 'causalTrace.tracing'
  // Attention
  | 'attention.title'
  | 'attention.needDTI'
  // Logit Lens
  | 'logitLens.title'
  | 'logitLens.needFLAIR'
  // Templates
  | 'templates.title'
  | 'templates.button'
  // Settings
  | 'settings.button'
  | 'settings.title'
  | 'settings.token'
  | 'settings.tokenPlaceholder'
  | 'settings.tokenSet'
  | 'settings.tokenClear'
  | 'settings.tokenValid'
  | 'settings.tokenInvalid'
  | 'settings.tokenNotSet'
  | 'settings.tokenFromEnv'
  | 'settings.tokenFromSession'
  | 'settings.device'
  | 'settings.deviceAuto'
  | 'settings.cache'
  | 'settings.cacheEntries'
  | 'settings.cacheClear'
  // Model Picker
  | 'modelPicker.searchPlaceholder'
  | 'modelPicker.recommended'
  | 'modelPicker.recent'
  | 'modelPicker.results'
  | 'modelPicker.tlCompat'
  | 'modelPicker.tlUnknown'
  | 'modelPicker.tlIncompat'
  | 'modelPicker.noResults'
  | 'modelPicker.tlOnly';

export const translations: Record<Locale, Record<TranslationKey, string>> = {
  en: {
    // ── Tooltips ──
    'tooltip.T1': 'Visualizes model architecture — layer types and parameter counts',
    'tooltip.T2': 'Shows weight distribution and L2 norms per layer',
    'tooltip.fMRI': 'Tracks token-level activation patterns through layers',
    'tooltip.DTI': 'Traces computational circuits for the final prediction',
    'tooltip.FLAIR': 'Detects anomalous layers via Logit Lens divergence',

    // ── Guide ──
    'guide.title': 'NEURAL MRI \u2014 USER GUIDE',

    'guide.overview.title': 'OVERVIEW',
    'guide.overview.body':
      'Neural MRI visualizes the internals of a language model like a medical MRI scanner.\n\n' +
      'The canvas on the left shows the model\'s layers as nodes:\n' +
      '\u2022 Left column = Attention layers\n' +
      '\u2022 Right column = MLP (Feed-Forward) layers\n' +
      '\u2022 Top = Embedding, Bottom = Output (Unembed)\n' +
      '\u2022 Lines between nodes = data flow connections\n\n' +
      'Select a scan mode from the tabs above, enter a prompt, and press SCAN to begin.\n' +
      'Click any node to inspect it and run perturbation experiments.',

    'guide.T1.title': 'T1 TOPOLOGY',
    'guide.T1.body':
      'Like a structural MRI, T1 shows the model\'s "anatomy."\n\n' +
      '\u2022 Node size = number of parameters (larger = more parameters)\n' +
      '\u2022 Node brightness = relative parameter count\n' +
      '\u2022 Layout shows the sequential layer structure\n\n' +
      'Use T1 to understand the model\'s architecture before running dynamic scans.',

    'guide.T2.title': 'T2 TENSOR',
    'guide.T2.body':
      'Like a T2-weighted MRI that reveals tissue density, T2 shows weight magnitudes.\n\n' +
      '\u2022 Bright nodes = high L2 norm (large weight values)\n' +
      '\u2022 Dim nodes = low L2 norm (smaller weights)\n' +
      '\u2022 Right panel bars show exact values per layer\n\n' +
      'Unusually bright or dim layers may indicate undertrained or overtrained components.',

    'guide.fMRI.title': 'fMRI ACTIVATION',
    'guide.fMRI.body':
      'Like functional MRI that shows brain activity, fMRI tracks which layers activate for each token.\n\n' +
      '\u2022 Bright/pulsing nodes = strongly activated for the selected token\n' +
      '\u2022 Dim nodes = low activation\n' +
      '\u2022 Use \u2190 \u2192 arrow keys or click token chips to step through tokens\n' +
      '\u2022 Glow effect appears on nodes with activation > 0.5\n\n' +
      'Watch how activation patterns shift across tokens to see which layers process which concepts.',

    'guide.DTI.title': 'DTI CIRCUITS',
    'guide.DTI.body':
      'Like Diffusion Tensor Imaging that traces nerve fibers, DTI reveals the computational circuits.\n\n' +
      '\u2022 Thick animated lines = critical pathway connections\n' +
      '\u2022 Bright nodes = components important for the final prediction\n' +
      '\u2022 Dim/thin connections = less important paths\n' +
      '\u2022 Importance is measured by zero-ablation: how much removing a component changes the output\n\n' +
      'This shows which layers actually contribute to the model\'s answer.',

    'guide.FLAIR.title': 'FLAIR ANOMALY',
    'guide.FLAIR.body':
      'Like FLAIR MRI that highlights lesions, FLAIR detects anomalous layers using Logit Lens.\n\n' +
      '\u2022 Red pulsing nodes = high anomaly score (intermediate prediction differs from final)\n' +
      '\u2022 Dark nodes = prediction has converged (normal)\n' +
      '\u2022 Expected pattern: early layers red (divergent) \u2192 later layers dark (converged)\n' +
      '\u2022 Anomaly = 0.6 \u00d7 KL divergence + 0.4 \u00d7 Entropy\n\n' +
      'If a late layer suddenly shows high anomaly, it may indicate unusual processing at that depth.',

    'guide.perturbation.title': 'PERTURBATION',
    'guide.perturbation.body':
      'Click a node and use the perturbation buttons to run causal experiments:\n\n' +
      '\u2022 Zero-out: Set the component\'s output to zero. Large KL = this component is critical.\n' +
      '\u2022 Amplify 2x: Double the component\'s output. Tests sensitivity to magnitude.\n' +
      '\u2022 Ablate: Replace output with its mean activation. Removes specific information while keeping the baseline.\n\n' +
      'The Comparison panel shows:\n' +
      '\u2022 Original vs Perturbed top prediction\n' +
      '\u2022 Logit diff: negative (red) = prediction weakened, positive (green) = strengthened\n' +
      '\u2022 KL divergence: how much the full distribution changed (higher = bigger impact)\n' +
      '\u2022 Top-3 predictions before and after\n\n' +
      'Embedding and Output layers cannot be perturbed.',

    'guide.shortcuts.title': 'KEYBOARD SHORTCUTS',
    'guide.shortcuts.body':
      '\u2190 \u2192    Step through tokens (fMRI / DTI / FLAIR)\n' +
      '1-5       Switch scan mode (T1/T2/fMRI/DTI/FLAIR)\n' +
      'R         Toggle recording\n' +
      'Space     Play/pause playback\n' +
      'L         Cycle layout mode\n' +
      'C         Toggle multi-prompt compare\n' +
      'Shift+C   Toggle cross-model compare\n' +
      'H         Toggle this guide\n' +
      'ESC       Close this guide',

    // ── Report ──
    'report.title': 'DIAGNOSTIC REPORT',
    'report.technique': 'TECHNIQUE',
    'report.findings': 'FINDINGS',
    'report.impression': 'IMPRESSION',
    'report.recommendation': 'RECOMMENDATION',
    'report.patient': 'Patient',
    'report.date': 'Date',
    'report.prompt': 'Prompt',
    'report.button': 'REPORT',
    'report.generating': 'GENERATING...',

    // ── Battery ──
    'battery.title': 'FUNCTIONAL BATTERY',
    'battery.detailTitle': 'FUNCTIONAL TEST BATTERY',
    'battery.run': 'RUN',
    'battery.running': 'RUNNING...',
    'battery.passed': 'PASSED',
    'battery.failed': 'FAILED',
    'battery.total': 'total',
    'battery.viewDetails': 'View details',
    'battery.noResult': 'Run battery to test model capabilities',
    'battery.peak': 'Peak',
    'battery.activeLayers': 'Active layers',
    'battery.compare': 'Compare',
    'battery.model': 'Model',
    'battery.includeSae': '+ SAE',
    'battery.autoLayer': 'Auto',
    'battery.saeFeatures': 'SAE Features',
    'battery.crossTestSae': 'CROSS-TEST SAE ANALYSIS',
    'report.explain': 'Explanation',
    // ── Compare ──
    'compare.title': 'COMPARE',
    'compare.promptA': 'Prompt A',
    'compare.promptB': 'Prompt B',
    'compare.diff': 'ACTIVATION DIFF',
    'compare.scan': 'COMPARE',
    'compare.stronger': 'stronger',
    // ── Export ──
    'export.button': 'EXPORT',
    'export.png': 'PNG Image',
    'export.svg': 'SVG Image',
    'export.json': 'Scan Data (JSON)',
    'export.report': 'Report (Markdown)',
    'export.gif': 'GIF Animation',
    'export.webm': 'WebM Video',
    // ── Recording ──
    'recording.rec': 'REC',
    'recording.stop': 'STOP',
    'recording.play': 'PLAY',
    'recording.pause': 'PAUSE',
    'recording.save': 'SAVE',
    'recording.load': 'LOAD',
    'recording.speed': 'Speed',
    // ── SAE ──
    'sae.title': 'SAE FEATURES',
    'sae.noSae': 'No SAE available for this model',
    'sae.layer': 'Layer',
    'sae.scan': 'DECODE',
    'sae.scanning': 'DECODING...',
    'sae.features': 'Top Features',
    'sae.heatmap': 'Activation Heatmap',
    'sae.reconstruction': 'Recon. Loss',
    'sae.sparsity': 'Sparsity',
    'sae.neuronpedia': 'Neuronpedia',
    // ── Layout ──
    'layout.vertical': 'Stack View',
    'layout.brain': 'Brain View',
    'layout.network': 'Network View',
    'layout.radial': 'Radial View',
    // ── Cross-model ──
    'crossModel.title': 'X-MODEL',
    'crossModel.selectModel': 'Select model...',
    'crossModel.compare': 'COMPARE',
    // ── Causal Trace ──
    'causalTrace.title': 'CAUSAL TRACE',
    'causalTrace.clean': 'Clean',
    'causalTrace.corruptPlaceholder': 'Corrupt prompt...',
    'causalTrace.trace': 'TRACE',
    'causalTrace.tracing': 'TRACING...',
    // ── Attention ──
    'attention.title': 'ATTENTION HEADS',
    'attention.needDTI': 'Run DTI scan first',
    // ── Logit Lens ──
    'logitLens.title': 'LOGIT LENS',
    'logitLens.needFLAIR': 'Run FLAIR scan first',
    // ── Templates ──
    'templates.title': 'Prompt Templates',
    'templates.button': '\u25A4',
    // ── Settings ──
    'settings.button': 'SETTINGS',
    'settings.title': 'SETTINGS',
    'settings.token': 'HUGGINGFACE TOKEN',
    'settings.tokenPlaceholder': 'hf_...',
    'settings.tokenSet': 'SET',
    'settings.tokenClear': 'CLEAR',
    'settings.tokenValid': 'Valid',
    'settings.tokenInvalid': 'Invalid',
    'settings.tokenNotSet': 'Not set',
    'settings.tokenFromEnv': 'from env',
    'settings.tokenFromSession': 'session',
    'settings.device': 'DEVICE',
    'settings.deviceAuto': 'Auto',
    'settings.cache': 'SCAN CACHE',
    'settings.cacheEntries': 'entries',
    'settings.cacheClear': 'CLEAR CACHE',
    // ── Model Picker ──
    'modelPicker.searchPlaceholder': 'Search HuggingFace models...',
    'modelPicker.recommended': 'RECOMMENDED',
    'modelPicker.recent': 'RECENT',
    'modelPicker.results': 'SEARCH RESULTS',
    'modelPicker.tlCompat': 'TL',
    'modelPicker.tlUnknown': 'TL?',
    'modelPicker.tlIncompat': 'No TL',
    'modelPicker.noResults': 'No models found',
    'modelPicker.tlOnly': 'TL only',
  },

  ko: {
    // ── Tooltips ──
    'tooltip.T1': '\ubaa8\ub378 \uc544\ud0a4\ud14d\ucc98 \uc2dc\uac01\ud654 \u2014 \ub808\uc774\uc5b4 \uc885\ub958\uc640 \ud30c\ub77c\ubbf8\ud130 \uc218',
    'tooltip.T2': '\ub808\uc774\uc5b4\ubcc4 \uac00\uc911\uce58 \ubd84\ud3ec\uc640 L2 \ud06c\uae30\ub97c \ud45c\uc2dc',
    'tooltip.fMRI': '\ud1a0\ud070\ubcc4 \ud65c\uc131\ud654 \ud328\ud134\uc744 \ub808\uc774\uc5b4\ubcc4\ub85c \ucd94\uc801',
    'tooltip.DTI': '\ucd5c\uc885 \uc608\uce21\uc5d0 \uae30\uc5ec\ud558\ub294 \uc5f0\uc0b0 \ud68c\ub85c\ub97c \ucd94\uc801',
    'tooltip.FLAIR': 'Logit Lens \ubc1c\uc0b0\uc744 \ud1b5\ud574 \uc774\uc0c1 \ub808\uc774\uc5b4 \ud0d0\uc9c0',

    // ── Guide ──
    'guide.title': 'NEURAL MRI \u2014 \uc0ac\uc6a9 \uac00\uc774\ub4dc',

    'guide.overview.title': '\uac1c\uc694',
    'guide.overview.body':
      'Neural MRI\ub294 \uc5b8\uc5b4 \ubaa8\ub378\uc758 \ub0b4\ubd80\ub97c \uc758\ub8cc MRI\ucc98\ub7fc \uc2dc\uac01\ud654\ud558\ub294 \ub3c4\uad6c\uc785\ub2c8\ub2e4.\n\n' +
      '\uc67c\ucabd \uce94\ubc84\uc2a4\ub294 \ubaa8\ub378\uc758 \ub808\uc774\uc5b4\ub97c \ub178\ub4dc\ub85c \ud45c\uc2dc\ud569\ub2c8\ub2e4:\n' +
      '\u2022 \uc67c\ucabd \uc5f4 = Attention \ub808\uc774\uc5b4\n' +
      '\u2022 \uc624\ub978\ucabd \uc5f4 = MLP (Feed-Forward) \ub808\uc774\uc5b4\n' +
      '\u2022 \uc704 = \uc784\ubca0\ub529, \uc544\ub798 = \ucd9c\ub825 (Unembed)\n' +
      '\u2022 \ub178\ub4dc \uc0ac\uc774 \uc120 = \ub370\uc774\ud130 \ud750\ub984 \uc5f0\uacb0\n\n' +
      '\uc704 \ud0ed\uc5d0\uc11c \uc2a4\uce94 \ubaa8\ub4dc\ub97c \uc120\ud0dd\ud558\uace0, \ud504\ub86c\ud504\ud2b8\ub97c \uc785\ub825\ud55c \ud6c4 SCAN\uc744 \ub204\ub974\uba74 \uc2dc\uc791\ub429\ub2c8\ub2e4.\n' +
      '\ub178\ub4dc\ub97c \ud074\ub9ad\ud558\uba74 \uc0c1\uc138 \uc815\ubcf4\ub97c \ubcf4\uace0 perturbation \uc2e4\ud5d8\uc744 \ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4.',

    'guide.T1.title': 'T1 TOPOLOGY',
    'guide.T1.body':
      '\uad6c\uc870\uc801 MRI\ucc98\ub7fc \ubaa8\ub378\uc758 "\ud574\ubd80\ub3c4"\ub97c \ubcf4\uc5ec\uc90d\ub2c8\ub2e4.\n\n' +
      '\u2022 \ub178\ub4dc \ud06c\uae30 = \ud30c\ub77c\ubbf8\ud130 \uc218 (\ud074\uc218\ub85d \ub354 \ub9ce\uc740 \ud30c\ub77c\ubbf8\ud130)\n' +
      '\u2022 \ub178\ub4dc \ubc1d\uae30 = \uc0c1\ub300\uc801 \ud30c\ub77c\ubbf8\ud130 \ube44\uc728\n' +
      '\u2022 \ub808\uc774\uc544\uc6c3\uc740 \uc21c\ucc28\uc801 \ub808\uc774\uc5b4 \uad6c\uc870\ub97c \ud45c\uc2dc\n\n' +
      '\ub3d9\uc801 \uc2a4\uce94 \uc804\uc5d0 \ubaa8\ub378 \uad6c\uc870\ub97c \ud30c\uc545\ud558\ub294 \ub370 \uc0ac\uc6a9\ud558\uc138\uc694.',

    'guide.T2.title': 'T2 TENSOR',
    'guide.T2.body':
      'T2 \uac00\uc911\uce58 MRI\ucc98\ub7fc \uc870\uc9c1 \ubc00\ub3c4\ub97c \ubcf4\uc5ec\uc8fc\ub294 \ubaa8\ub4dc\uc785\ub2c8\ub2e4.\n\n' +
      '\u2022 \ubc1d\uc740 \ub178\ub4dc = \ub192\uc740 L2 \ub178\ub984 (\ud070 \uac00\uc911\uce58 \uac12)\n' +
      '\u2022 \uc5b4\ub450\uc6b4 \ub178\ub4dc = \ub0ae\uc740 L2 \ub178\ub984 (\uc791\uc740 \uac00\uc911\uce58)\n' +
      '\u2022 \uc624\ub978\ucabd \ud328\ub110\uc758 \ubc14\uc5d0\uc11c \ub808\uc774\uc5b4\ubcc4 \uc815\ud655\ud55c \uac12\uc744 \ud655\uc778 \uac00\ub2a5\n\n' +
      '\ube44\uc815\uc0c1\uc801\uc73c\ub85c \ubc1d\uac70\ub098 \uc5b4\ub450\uc6b4 \ub808\uc774\uc5b4\ub294 \ud559\uc2b5 \ubd88\uade0\ud615\uc744 \ub098\ud0c0\ub0bc \uc218 \uc788\uc2b5\ub2c8\ub2e4.',

    'guide.fMRI.title': 'fMRI ACTIVATION',
    'guide.fMRI.body':
      '\uae30\ub2a5\uc801 MRI\ucc98\ub7fc \uac01 \ud1a0\ud070\uc5d0 \ub300\ud574 \uc5b4\ub5a4 \ub808\uc774\uc5b4\uac00 \ud65c\uc131\ud654\ub418\ub294\uc9c0 \ucd94\uc801\ud569\ub2c8\ub2e4.\n\n' +
      '\u2022 \ubc1d\uac8c \ubc18\uc9dd\uc774\ub294 \ub178\ub4dc = \uc120\ud0dd\ub41c \ud1a0\ud070\uc5d0 \ub300\ud574 \uac15\ud558\uac8c \ud65c\uc131\ud654\n' +
      '\u2022 \uc5b4\ub450\uc6b4 \ub178\ub4dc = \ub0ae\uc740 \ud65c\uc131\ud654\n' +
      '\u2022 \u2190 \u2192 \ud654\uc0b4\ud45c \ud0a4 \ub610\ub294 \ud1a0\ud070 \uce69\uc744 \ud074\ub9ad\ud558\uc5ec \ud1a0\ud070 \ud0d0\uc0c9\n' +
      '\u2022 \ud65c\uc131\ud654 > 0.5\uc778 \ub178\ub4dc\uc5d0 \uae00\ub85c\uc6b0 \ud6a8\uacfc \ud45c\uc2dc\n\n' +
      '\ud1a0\ud070\ubcc4\ub85c \ud65c\uc131\ud654 \ud328\ud134\uc774 \uc5b4\ub5bb\uac8c \ubcc0\ud558\ub294\uc9c0 \uad00\ucc30\ud558\uba74 \uc5b4\ub5a4 \ub808\uc774\uc5b4\uac00 \uc5b4\ub5a4 \uac1c\ub150\uc744 \ucc98\ub9ac\ud558\ub294\uc9c0 \uc54c \uc218 \uc788\uc2b5\ub2c8\ub2e4.',

    'guide.DTI.title': 'DTI CIRCUITS',
    'guide.DTI.body':
      '\ud655\uc0b0 \ud150\uc11c \uc601\uc0c1(DTI)\ucc98\ub7fc \uc2e0\uacbd \uc12c\uc720\ub97c \ucd94\uc801\ud558\ub4ef, \uc5f0\uc0b0 \ud68c\ub85c\ub97c \ubcf4\uc5ec\uc90d\ub2c8\ub2e4.\n\n' +
      '\u2022 \uad75\uace0 \uc560\ub2c8\uba54\uc774\uc158\ub418\ub294 \uc120 = \ud575\uc2ec \uacbd\ub85c \uc5f0\uacb0\n' +
      '\u2022 \ubc1d\uc740 \ub178\ub4dc = \ucd5c\uc885 \uc608\uce21\uc5d0 \uc911\uc694\ud55c \ucef4\ud3ec\ub10c\ud2b8\n' +
      '\u2022 \uc5b4\ub450\uc6b4/\uac00\ub294 \uc5f0\uacb0 = \ub35c \uc911\uc694\ud55c \uacbd\ub85c\n' +
      '\u2022 \uc911\uc694\ub3c4\ub294 zero-ablation\uc73c\ub85c \uce21\uc815: \ucef4\ud3ec\ub10c\ud2b8 \uc81c\uac70 \uc2dc \ucd9c\ub825 \ubcc0\ud654\ub7c9\n\n' +
      '\uc5b4\ub5a4 \ub808\uc774\uc5b4\uac00 \ubaa8\ub378\uc758 \ub2f5\ubcc0\uc5d0 \uc2e4\uc81c\ub85c \uae30\uc5ec\ud558\ub294\uc9c0 \ubcf4\uc5ec\uc90d\ub2c8\ub2e4.',

    'guide.FLAIR.title': 'FLAIR ANOMALY',
    'guide.FLAIR.body':
      'FLAIR MRI\uac00 \ubcd1\ubcc0\uc744 \uac15\uc870\ud558\ub4ef, Logit Lens\ub97c \uc0ac\uc6a9\ud558\uc5ec \uc774\uc0c1 \ub808\uc774\uc5b4\ub97c \ud0d0\uc9c0\ud569\ub2c8\ub2e4.\n\n' +
      '\u2022 \ube68\uac04 \ubc18\uc9dd\uc774\ub294 \ub178\ub4dc = \ub192\uc740 \uc774\uc0c1 \uc810\uc218 (\uc911\uac04 \uc608\uce21\uc774 \ucd5c\uc885\uacfc \ub2e4\ub984)\n' +
      '\u2022 \uc5b4\ub450\uc6b4 \ub178\ub4dc = \uc608\uce21\uc774 \uc218\ub834\ub428 (\uc815\uc0c1)\n' +
      '\u2022 \uc815\uc0c1 \ud328\ud134: \ucd08\ubc18 \ub808\uc774\uc5b4 \ube68\uac04\uc0c9 (\ubc1c\uc0b0) \u2192 \ud6c4\ubc18 \ub808\uc774\uc5b4 \uc5b4\ub450\uc6c0 (\uc218\ub834)\n' +
      '\u2022 \uc774\uc0c1 \uc810\uc218 = 0.6 \u00d7 KL \ubc1c\uc0b0 + 0.4 \u00d7 \uc5d4\ud2b8\ub85c\ud53c\n\n' +
      '\ud6c4\ubc18 \ub808\uc774\uc5b4\uc5d0\uc11c \uac11\uc790\uae30 \ub192\uc740 \uc774\uc0c1\uc774 \ub098\ud0c0\ub098\uba74 \ud574\ub2f9 \uae4a\uc774\uc5d0\uc11c \ube44\uc815\uc0c1\uc801 \ucc98\ub9ac\uac00 \uc77c\uc5b4\ub098\uace0 \uc788\uc744 \uc218 \uc788\uc2b5\ub2c8\ub2e4.',

    'guide.perturbation.title': 'PERTURBATION',
    'guide.perturbation.body':
      '\ub178\ub4dc\ub97c \ud074\ub9ad\ud55c \ud6c4 perturbation \ubc84\ud2bc\uc73c\ub85c \uc778\uacfc \uc2e4\ud5d8\uc744 \uc218\ud589\ud569\ub2c8\ub2e4:\n\n' +
      '\u2022 Zero-out: \ucef4\ud3ec\ub10c\ud2b8 \ucd9c\ub825\uc744 0\uc73c\ub85c \uc124\uc815. KL\uc774 \ud06c\uba74 \uc774 \ucef4\ud3ec\ub10c\ud2b8\uac00 \uc911\uc694.\n' +
      '\u2022 Amplify 2x: \ucef4\ud3ec\ub10c\ud2b8 \ucd9c\ub825\uc744 2\ubc30\ub85c. \ud06c\uae30 \ubbfc\uac10\ub3c4 \ud14c\uc2a4\ud2b8.\n' +
      '\u2022 Ablate: \ucd9c\ub825\uc744 \ud3c9\uade0 \ud65c\uc131\ud654\ub85c \ub300\uccb4. \uae30\ubcf8\uac12\uc740 \uc720\uc9c0\ud558\uba74\uc11c \ud2b9\uc815 \uc815\ubcf4\ub9cc \uc81c\uac70.\n\n' +
      'Comparison \ud328\ub110\uc5d0\uc11c \ud655\uc778\ud560 \uc218 \uc788\ub294 \uac83:\n' +
      '\u2022 Original vs Perturbed \uc0c1\uc704 \uc608\uce21\n' +
      '\u2022 Logit diff: \uc74c\uc218(\ube68\uac04\uc0c9) = \uc608\uce21 \uc57d\ud654, \uc591\uc218(\ucd08\ub85d\uc0c9) = \uc608\uce21 \uac15\ud654\n' +
      '\u2022 KL divergence: \uc804\uccb4 \ubd84\ud3ec\uac00 \uc5bc\ub9c8\ub098 \ubcc0\ud588\ub294\uc9c0 (\ub192\uc744\uc218\ub85d \ud070 \uc601\ud5a5)\n' +
      '\u2022 \ubcc0\ud615 \uc804/\ud6c4 Top-3 \uc608\uce21\n\n' +
      'Embedding\uacfc Output \ub808\uc774\uc5b4\ub294 perturbation\uc774 \ubd88\uac00\ub2a5\ud569\ub2c8\ub2e4.',

    'guide.shortcuts.title': 'KEYBOARD SHORTCUTS',
    'guide.shortcuts.body':
      '\u2190 \u2192    \ud1a0\ud070 \ud0d0\uc0c9 (fMRI / DTI / FLAIR)\n' +
      '1-5       \uc2a4\uce94 \ubaa8\ub4dc \uc804\ud658 (T1/T2/fMRI/DTI/FLAIR)\n' +
      'R         \ub179\ud654 \ud1a0\uae00\n' +
      'Space     \uc7ac\uc0dd/\uc77c\uc2dc\uc815\uc9c0\n' +
      'L         \ub808\uc774\uc544\uc6c3 \uc21c\ud658\n' +
      'C         \ub2e4\uc911 \ud504\ub86c\ud504\ud2b8 \ube44\uad50 \ud1a0\uae00\n' +
      'Shift+C   \uad50\ucc28 \ubaa8\ub378 \ube44\uad50 \ud1a0\uae00\n' +
      'H         \uac00\uc774\ub4dc \uc5f4\uae30/\ub2eb\uae30\n' +
      'ESC       \uac00\uc774\ub4dc \ub2eb\uae30',

    // ── Report ──
    'report.title': '\uc9c4\ub2e8 \ub9ac\ud3ec\ud2b8',
    'report.technique': '\uac80\uc0ac \uae30\ubc95',
    'report.findings': '\uc18c\uacac',
    'report.impression': '\uc885\ud569 \ud310\ub3c5',
    'report.recommendation': '\uad8c\uace0 \uc0ac\ud56d',
    'report.patient': '\ud658\uc790',
    'report.date': '\ub0a0\uc9dc',
    'report.prompt': '\ud504\ub86c\ud504\ud2b8',
    'report.button': '\ub9ac\ud3ec\ud2b8',
    'report.generating': '\uc0dd\uc131 \uc911...',

    // ── Battery ──
    'battery.title': '\uae30\ub2a5 \uac80\uc0ac',
    'battery.detailTitle': '\uae30\ub2a5 \uac80\uc0ac \ubc30\ud130\ub9ac',
    'battery.run': '\uc2e4\ud589',
    'battery.running': '\uc2e4\ud589 \uc911...',
    'battery.passed': '\ud1b5\uacfc',
    'battery.failed': '\uc2e4\ud328',
    'battery.total': '\uc804\uccb4',
    'battery.viewDetails': '\uc0c1\uc138 \ubcf4\uae30',
    'battery.noResult': '\ubc30\ud130\ub9ac\ub97c \uc2e4\ud589\ud558\uc5ec \ubaa8\ub378 \ub2a5\ub825\uc744 \ud14c\uc2a4\ud2b8\ud558\uc138\uc694',
    'battery.peak': '\ud53c\ud06c',
    'battery.activeLayers': '\ud65c\uc131 \ub808\uc774\uc5b4',
    'battery.compare': '\ube44\uad50',
    'battery.model': '\ubaa8\ub378',
    'battery.includeSae': '+ SAE',
    'battery.autoLayer': '\uc790\ub3d9',
    'battery.saeFeatures': 'SAE \ud2b9\uc9d5',
    'battery.crossTestSae': '\uad50\ucc28 \ud14c\uc2a4\ud2b8 SAE \ubd84\uc11d',
    'report.explain': '\ud574\uc124',
    // ── Compare ──
    'compare.title': '\ube44\uad50',
    'compare.promptA': '\ud504\ub86c\ud504\ud2b8 A',
    'compare.promptB': '\ud504\ub86c\ud504\ud2b8 B',
    'compare.diff': '\ud65c\uc131\ud654 \ucc28\uc774',
    'compare.scan': '\ube44\uad50 \uc2a4\uce94',
    'compare.stronger': '\ub354 \uac15\ud568',
    // ── Export ──
    'export.button': '\ub0b4\ubcf4\ub0b4\uae30',
    'export.png': 'PNG \uc774\ubbf8\uc9c0',
    'export.svg': 'SVG \uc774\ubbf8\uc9c0',
    'export.json': '\uc2a4\uce94 \ub370\uc774\ud130 (JSON)',
    'export.report': '\ub9ac\ud3ec\ud2b8 (Markdown)',
    'export.gif': 'GIF \uc560\ub2c8\uba54\uc774\uc158',
    'export.webm': 'WebM \ub3d9\uc601\uc0c1',
    // ── Recording ──
    'recording.rec': '\ub179\ud654',
    'recording.stop': '\uc815\uc9c0',
    'recording.play': '\uc7ac\uc0dd',
    'recording.pause': '\uc77c\uc2dc\uc815\uc9c0',
    'recording.save': '\uc800\uc7a5',
    'recording.load': '\ubd88\ub7ec\uc624\uae30',
    'recording.speed': '\uc18d\ub3c4',
    // ── SAE ──
    'sae.title': 'SAE \ud2b9\uc9d5',
    'sae.noSae': '\uc774 \ubaa8\ub378\uc5d0\ub294 SAE\uac00 \uc5c6\uc2b5\ub2c8\ub2e4',
    'sae.layer': '\ub808\uc774\uc5b4',
    'sae.scan': '\ub514\ucf54\ub4dc',
    'sae.scanning': '\ub514\ucf54\ub529 \uc911...',
    'sae.features': '\uc0c1\uc704 \ud2b9\uc9d5',
    'sae.heatmap': '\ud65c\uc131\ud654 \ud788\ud2b8\ub9f5',
    'sae.reconstruction': '\uc7ac\uad6c\uc131 \uc190\uc2e4',
    'sae.sparsity': '\ud76c\uc18c\uc131',
    'sae.neuronpedia': 'Neuronpedia',
    // ── Layout ──
    'layout.vertical': '\uc2a4\ud0dd \ubdf0',
    'layout.brain': '\ub1cc \ubdf0',
    'layout.network': '\ub124\ud2b8\uc6cc\ud06c \ubdf0',
    'layout.radial': '\ubc29\uc0ac\ud615 \ubdf0',
    // ── Cross-model ──
    'crossModel.title': 'X-MODEL',
    'crossModel.selectModel': '\ubaa8\ub378 \uc120\ud0dd...',
    'crossModel.compare': '\ube44\uad50',
    // ── Causal Trace ──
    'causalTrace.title': 'CAUSAL TRACE',
    'causalTrace.clean': '\uc815\uc0c1',
    'causalTrace.corruptPlaceholder': '\uc190\uc0c1 \ud504\ub86c\ud504\ud2b8...',
    'causalTrace.trace': '\ucd94\uc801',
    'causalTrace.tracing': '\ucd94\uc801 \uc911...',
    // ── Attention ──
    'attention.title': 'ATTENTION HEADS',
    'attention.needDTI': 'DTI \uc2a4\uce94\uc744 \uba3c\uc800 \uc2e4\ud589\ud558\uc138\uc694',
    // ── Logit Lens ──
    'logitLens.title': 'LOGIT LENS',
    'logitLens.needFLAIR': 'FLAIR \uc2a4\uce94\uc744 \uba3c\uc800 \uc2e4\ud589\ud558\uc138\uc694',
    // ── Templates ──
    'templates.title': '\ud504\ub86c\ud504\ud2b8 \ud15c\ud50c\ub9bf',
    'templates.button': '\u25A4',
    // ── Settings ──
    'settings.button': '\uc124\uc815',
    'settings.title': '\uc124\uc815',
    'settings.token': 'HUGGINGFACE \ud1a0\ud070',
    'settings.tokenPlaceholder': 'hf_...',
    'settings.tokenSet': '\uc124\uc815',
    'settings.tokenClear': '\uc0ad\uc81c',
    'settings.tokenValid': '\uc720\ud6a8',
    'settings.tokenInvalid': '\ubb34\ud6a8',
    'settings.tokenNotSet': '\ubbf8\uc124\uc815',
    'settings.tokenFromEnv': '\ud658\uacbd\ubcc0\uc218',
    'settings.tokenFromSession': '\uc138\uc158',
    'settings.device': '\uc7a5\uce58',
    'settings.deviceAuto': '\uc790\ub3d9',
    'settings.cache': '\uc2a4\uce94 \uce90\uc2dc',
    'settings.cacheEntries': '\ud56d\ubaa9',
    'settings.cacheClear': '\uce90\uc2dc \uc0ad\uc81c',
    // ── Model Picker ──
    'modelPicker.searchPlaceholder': 'HuggingFace \ubaa8\ub378 \uac80\uc0c9...',
    'modelPicker.recommended': '\ucd94\ucc9c',
    'modelPicker.recent': '\ucd5c\uadfc',
    'modelPicker.results': '\uac80\uc0c9 \uacb0\uacfc',
    'modelPicker.tlCompat': 'TL',
    'modelPicker.tlUnknown': 'TL?',
    'modelPicker.tlIncompat': 'TL \ubd88\uac00',
    'modelPicker.noResults': '\ubaa8\ub378\uc744 \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4',
    'modelPicker.tlOnly': 'TL\ub9cc',
  },
};
