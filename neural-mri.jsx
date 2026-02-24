import { useState, useEffect, useRef, useCallback } from "react";

const MODES = {
  T1: { label: "T1 Topology", desc: "Topology Layer 1 — Architecture, Layers, Parameters", color: "#e0e0e0", accent: "#8899aa" },
  T2: { label: "T2 Tensor", desc: "Tensor Layer 2 — Weight Distribution & Magnitude", color: "#aaccee", accent: "#4488cc" },
  fMRI: { label: "fMRI", desc: "functional Model Resonance Imaging — Activation Patterns", color: "#ff6644", accent: "#ffaa00" },
  DTI: { label: "DTI", desc: "Data Tractography Imaging — Information Flow Pathways", color: "#44ddaa", accent: "#8866ff" },
  FLAIR: { label: "FLAIR", desc: "Feature-Level Anomaly Identification & Reporting", color: "#ff4466", accent: "#ff88aa" },
};

const LAYERS = [
  { id: "embed", label: "Embedding", y: 60, neurons: 8, type: "input" },
  { id: "attn1", label: "Attn L1", y: 130, neurons: 12, type: "attention" },
  { id: "mlp1", label: "MLP L1", y: 200, neurons: 10, type: "mlp" },
  { id: "attn2", label: "Attn L2", y: 270, neurons: 12, type: "attention" },
  { id: "mlp2", label: "MLP L2", y: 340, neurons: 10, type: "mlp" },
  { id: "attn3", label: "Attn L3", y: 410, neurons: 12, type: "attention" },
  { id: "mlp3", label: "MLP L3", y: 480, neurons: 10, type: "mlp" },
  { id: "out", label: "Output", y: 550, neurons: 6, type: "output" },
];

function generateNeurons(layers, width) {
  const neurons = [];
  layers.forEach((layer) => {
    const startX = (width - (layer.neurons - 1) * 42) / 2;
    for (let i = 0; i < layer.neurons; i++) {
      neurons.push({
        id: `${layer.id}-${i}`,
        layerId: layer.id,
        layerType: layer.type,
        x: startX + i * 42,
        y: layer.y,
        baseSize: layer.type === "attention" ? 7 : layer.type === "input" || layer.type === "output" ? 5 : 6,
        activation: Math.random(),
        weight: Math.random() * 2 - 1,
        anomaly: Math.random() > 0.85 ? Math.random() * 0.8 + 0.2 : 0,
      });
    }
  });
  return neurons;
}

function generateConnections(neurons, layers) {
  const conns = [];
  for (let li = 0; li < layers.length - 1; li++) {
    const from = neurons.filter((n) => n.layerId === layers[li].id);
    const to = neurons.filter((n) => n.layerId === layers[li + 1].id);
    from.forEach((f) => {
      const numConns = Math.min(3 + Math.floor(Math.random() * 3), to.length);
      const targets = [...to].sort(() => Math.random() - 0.5).slice(0, numConns);
      targets.forEach((t) => {
        conns.push({
          id: `${f.id}-${t.id}`,
          from: f,
          to: t,
          strength: Math.random(),
          pathway: Math.random() > 0.6,
        });
      });
    });
  }
  return conns;
}

function ScanLines() {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 2, opacity: 0.04 }}>
      {Array.from({ length: 120 }, (_, i) => (
        <div key={i} style={{ height: "1px", background: "#fff", marginBottom: "4px" }} />
      ))}
    </div>
  );
}

function DicomHeader({ mode, prompt, model }) {
  const now = new Date();
  const ts = `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, "0")}.${String(now.getDate()).padStart(2, "0")} ${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  return (
    <div style={{ fontFamily: "'Courier New', monospace", fontSize: "10px", color: "#66aa88", padding: "8px 12px", display: "flex", justifyContent: "space-between", opacity: 0.8, letterSpacing: "0.5px" }}>
      <div>
        <div>MODEL: {model}</div>
        <div>SEQ: {MODES[mode].label.toUpperCase()}</div>
        <div>PROMPT: {prompt.slice(0, 40)}{prompt.length > 40 ? "..." : ""}</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div>Model Resonance Imaging v0.1</div>
        <div>SCAN: {ts}</div>
        <div>FOV: All Layers</div>
      </div>
    </div>
  );
}

function NeuralCanvas({ mode, neurons, connections, stimNode, setStimNode, time }) {
  const width = 560;
  const height = 620;
  const modeConf = MODES[mode];

  const getNeuronColor = useCallback((n) => {
    if (mode === "T1") {
      const v = Math.floor(180 + n.baseSize * 8);
      return `rgb(${v},${v},${v + 10})`;
    }
    if (mode === "T2") {
      const w = Math.abs(n.weight);
      const b = Math.floor(100 + w * 155);
      return `rgb(${Math.floor(60 + w * 40)},${Math.floor(80 + w * 80)},${b})`;
    }
    if (mode === "fMRI") {
      const a = n.activation;
      const pulse = Math.sin(time * 0.03 + n.x * 0.01 + n.y * 0.02) * 0.15 + 0.85;
      const v = a * pulse;
      if (v > 0.6) return `rgb(${Math.floor(200 + v * 55)},${Math.floor(v * 120)},${Math.floor(v * 30)})`;
      if (v > 0.3) return `rgb(${Math.floor(v * 200)},${Math.floor(v * 160)},${Math.floor(40 + v * 60)})`;
      return `rgb(${Math.floor(30 + v * 80)},${Math.floor(30 + v * 100)},${Math.floor(80 + v * 120)})`;
    }
    if (mode === "DTI") {
      const hue = (n.x / width) * 120 + (n.y / height) * 120;
      return `hsl(${hue}, 70%, 55%)`;
    }
    if (mode === "FLAIR") {
      if (n.anomaly > 0.3) {
        const pulse = Math.sin(time * 0.05) * 0.3 + 0.7;
        return `rgba(255,${Math.floor(50 + n.anomaly * 60)},${Math.floor(80 + n.anomaly * 40)},${pulse})`;
      }
      return `rgb(60,65,75)`;
    }
    return "#888";
  }, [mode, time]);

  const getNeuronSize = useCallback((n) => {
    const isStim = stimNode === n.id;
    const base = n.baseSize;
    if (mode === "fMRI") {
      const pulse = Math.sin(time * 0.03 + n.x * 0.02) * 0.3 + 1;
      return base * (0.8 + n.activation * 0.8) * pulse * (isStim ? 1.8 : 1);
    }
    if (mode === "FLAIR" && n.anomaly > 0.3) {
      return base * (1 + n.anomaly * 0.8) * (isStim ? 1.8 : 1);
    }
    return base * (isStim ? 1.6 : 1);
  }, [mode, time, stimNode]);

  const getConnectionStyle = useCallback((c) => {
    if (mode === "T1") return { stroke: "rgba(120,130,140,0.15)", width: 0.5 };
    if (mode === "T2") {
      const w = c.strength;
      return { stroke: `rgba(100,150,200,${w * 0.3})`, width: 0.5 + w };
    }
    if (mode === "fMRI") {
      const a = (c.from.activation + c.to.activation) / 2;
      const pulse = Math.sin(time * 0.02 + c.from.y * 0.01) * 0.3 + 0.7;
      return { stroke: a > 0.5 ? `rgba(255,${Math.floor(100 + a * 80)},50,${a * pulse * 0.5})` : `rgba(50,70,120,${a * 0.15})`, width: a > 0.5 ? a * 2 : 0.3 };
    }
    if (mode === "DTI") {
      if (!c.pathway) return { stroke: "rgba(50,55,65,0.05)", width: 0.2 };
      const hue = ((c.from.x + c.to.x) / (2 * width)) * 200 + 100;
      const flow = (Math.sin(time * 0.04 - c.from.y * 0.015) + 1) / 2;
      return { stroke: `hsla(${hue},80%,60%,${0.2 + flow * 0.5})`, width: 1 + c.strength * 2 };
    }
    if (mode === "FLAIR") {
      if (c.from.anomaly > 0.3 || c.to.anomaly > 0.3) {
        return { stroke: `rgba(255,80,100,${0.2 + Math.sin(time * 0.05) * 0.15})`, width: 1.5 };
      }
      return { stroke: "rgba(50,55,65,0.08)", width: 0.3 };
    }
    return { stroke: "rgba(80,80,80,0.1)", width: 0.5 };
  }, [mode, time]);

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <defs>
        <radialGradient id="glow-fmri" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(255,100,40,0.3)" />
          <stop offset="100%" stopColor="rgba(255,100,40,0)" />
        </radialGradient>
        <radialGradient id="glow-flair" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(255,60,100,0.4)" />
          <stop offset="100%" stopColor="rgba(255,60,100,0)" />
        </radialGradient>
        <filter id="blur-soft">
          <feGaussianBlur stdDeviation="3" />
        </filter>
        <filter id="blur-strong">
          <feGaussianBlur stdDeviation="8" />
        </filter>
      </defs>

      {/* Background glow layer */}
      {mode === "fMRI" && neurons.filter(n => n.activation > 0.6).map(n => (
        <circle key={`glow-${n.id}`} cx={n.x} cy={n.y} r={n.activation * 25} fill="url(#glow-fmri)" filter="url(#blur-strong)" />
      ))}
      {mode === "FLAIR" && neurons.filter(n => n.anomaly > 0.3).map(n => (
        <circle key={`glow-${n.id}`} cx={n.x} cy={n.y} r={20 + n.anomaly * 15} fill="url(#glow-flair)" filter="url(#blur-strong)" />
      ))}

      {/* Connections */}
      {connections.map((c) => {
        const s = getConnectionStyle(c);
        if (mode === "DTI" && c.pathway) {
          const mx = (c.from.x + c.to.x) / 2 + (Math.random() - 0.5) * 20;
          const my = (c.from.y + c.to.y) / 2;
          return <path key={c.id} d={`M${c.from.x},${c.from.y} Q${mx},${my} ${c.to.x},${c.to.y}`} fill="none" stroke={s.stroke} strokeWidth={s.width} strokeLinecap="round" />;
        }
        return <line key={c.id} x1={c.from.x} y1={c.from.y} x2={c.to.x} y2={c.to.y} stroke={s.stroke} strokeWidth={s.width} />;
      })}

      {/* Neurons */}
      {neurons.map((n) => {
        const color = getNeuronColor(n);
        const size = getNeuronSize(n);
        const isStim = stimNode === n.id;
        return (
          <g key={n.id} onClick={() => setStimNode(isStim ? null : n.id)} style={{ cursor: "pointer" }}>
            {(mode === "fMRI" && n.activation > 0.5) && (
              <circle cx={n.x} cy={n.y} r={size * 2.5} fill={color} opacity={0.15} filter="url(#blur-soft)" />
            )}
            <circle cx={n.x} cy={n.y} r={size} fill={color} stroke={isStim ? "#00ffaa" : "none"} strokeWidth={isStim ? 2 : 0} />
            {isStim && (
              <>
                <circle cx={n.x} cy={n.y} r={size + 6} fill="none" stroke="#00ffaa" strokeWidth={1} opacity={0.5 + Math.sin(time * 0.08) * 0.3} />
                <circle cx={n.x} cy={n.y} r={size + 12} fill="none" stroke="#00ffaa" strokeWidth={0.5} opacity={0.2 + Math.sin(time * 0.06) * 0.15} />
              </>
            )}
          </g>
        );
      })}

      {/* Layer labels */}
      {LAYERS.map((l) => (
        <text key={l.id} x={12} y={l.y + 4} fill="rgba(100,180,140,0.5)" fontSize="9" fontFamily="'Courier New', monospace">{l.label}</text>
      ))}
    </svg>
  );
}

function StimPanel({ stimNode, neurons, mode, onPerturb }) {
  const node = neurons.find((n) => n.id === stimNode);
  if (!node) return (
    <div style={{ padding: "16px", color: "#556", fontFamily: "'Courier New', monospace", fontSize: "11px", textAlign: "center", marginTop: "20px" }}>
      <div style={{ color: "#66aa88", marginBottom: "6px" }}>◉ STIMULATION MODE</div>
      <div>Click any neuron to select</div>
      <div style={{ marginTop: "4px", opacity: 0.6 }}>Inspect activation, apply perturbation</div>
    </div>
  );

  return (
    <div style={{ padding: "12px", fontFamily: "'Courier New', monospace", fontSize: "10px", color: "#aab" }}>
      <div style={{ color: "#00ffaa", marginBottom: "8px", fontSize: "11px", letterSpacing: "1px" }}>◉ NODE SELECTED</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px", marginBottom: "12px" }}>
        <span style={{ color: "#66aa88" }}>ID:</span><span>{node.id}</span>
        <span style={{ color: "#66aa88" }}>Layer:</span><span>{node.layerId}</span>
        <span style={{ color: "#66aa88" }}>Type:</span><span>{node.layerType}</span>
        <span style={{ color: "#66aa88" }}>Activation:</span><span style={{ color: node.activation > 0.6 ? "#ff8844" : "#aab" }}>{node.activation.toFixed(4)}</span>
        <span style={{ color: "#66aa88" }}>Weight Mag:</span><span>{Math.abs(node.weight).toFixed(4)}</span>
        <span style={{ color: "#66aa88" }}>Anomaly:</span><span style={{ color: node.anomaly > 0.3 ? "#ff4466" : "#aab" }}>{node.anomaly.toFixed(4)}</span>
      </div>
      <div style={{ color: "#66aa88", marginBottom: "6px", fontSize: "11px" }}>PERTURBATION</div>
      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
        {["Zero-out", "Amplify 2×", "Invert", "Noise ±0.5", "Ablate"].map((op) => (
          <button key={op} onClick={() => onPerturb(op)} style={{
            background: "rgba(0,255,170,0.08)", border: "1px solid rgba(0,255,170,0.25)", color: "#00ffaa",
            padding: "4px 8px", fontSize: "9px", fontFamily: "'Courier New', monospace", cursor: "pointer",
            borderRadius: "2px", transition: "all 0.2s"
          }}
            onMouseEnter={(e) => { e.target.style.background = "rgba(0,255,170,0.2)"; }}
            onMouseLeave={(e) => { e.target.style.background = "rgba(0,255,170,0.08)"; }}
          >{op}</button>
        ))}
      </div>
    </div>
  );
}

function ActivationBar({ neurons, mode }) {
  const layerData = LAYERS.map((l) => {
    const lNeurons = neurons.filter((n) => n.layerId === l.id);
    if (mode === "fMRI") return { label: l.label, value: lNeurons.reduce((s, n) => s + n.activation, 0) / lNeurons.length };
    if (mode === "FLAIR") return { label: l.label, value: lNeurons.reduce((s, n) => s + n.anomaly, 0) / lNeurons.length };
    if (mode === "T2") return { label: l.label, value: lNeurons.reduce((s, n) => s + Math.abs(n.weight), 0) / lNeurons.length };
    return { label: l.label, value: lNeurons.reduce((s, n) => s + n.baseSize / 12, 0) / lNeurons.length };
  });
  const maxVal = Math.max(...layerData.map((d) => d.value), 0.01);
  const barColor = mode === "fMRI" ? "#ff6644" : mode === "FLAIR" ? "#ff4466" : mode === "DTI" ? "#44ddaa" : mode === "T2" ? "#4488cc" : "#8899aa";

  return (
    <div style={{ padding: "12px", fontFamily: "'Courier New', monospace", fontSize: "9px", color: "#66aa88" }}>
      <div style={{ marginBottom: "8px", letterSpacing: "1px" }}>LAYER SUMMARY</div>
      {layerData.map((d) => (
        <div key={d.label} style={{ display: "flex", alignItems: "center", marginBottom: "3px", gap: "8px" }}>
          <span style={{ width: "52px", textAlign: "right", color: "#556" }}>{d.label}</span>
          <div style={{ flex: 1, height: "8px", background: "rgba(255,255,255,0.03)", borderRadius: "1px", overflow: "hidden" }}>
            <div style={{ width: `${(d.value / maxVal) * 100}%`, height: "100%", background: barColor, opacity: 0.7, borderRadius: "1px", transition: "width 0.5s" }} />
          </div>
          <span style={{ width: "36px", textAlign: "right", color: "#778" }}>{d.value.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

function LogPanel({ logs }) {
  return (
    <div style={{
      padding: "8px 12px", fontFamily: "'Courier New', monospace", fontSize: "9px",
      color: "#556", maxHeight: "80px", overflowY: "auto", borderTop: "1px solid rgba(100,170,136,0.15)"
    }}>
      {logs.slice(-6).map((log, i) => (
        <div key={i} style={{ marginBottom: "2px" }}>
          <span style={{ color: "#66aa88" }}>[{log.time}]</span> {log.msg}
        </div>
      ))}
    </div>
  );
}

export default function NeuralMRI() {
  const [mode, setMode] = useState("T1");
  const [prompt, setPrompt] = useState("The capital of France is");
  const [model, setModel] = useState("Llama-3.2-3B");
  const [stimNode, setStimNode] = useState(null);
  const [time, setTime] = useState(0);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(100);
  const [logs, setLogs] = useState([
    { time: "00:00", msg: "Model Resonance Imaging Scanner initialized" },
    { time: "00:01", msg: "Model loaded: Llama-3.2-3B (3.21B params)" },
  ]);

  const canvasWidth = 560;
  const [neurons] = useState(() => generateNeurons(LAYERS, canvasWidth));
  const [connections] = useState(() => generateConnections(neurons, LAYERS));

  useEffect(() => {
    const interval = setInterval(() => setTime((t) => t + 1), 50);
    return () => clearInterval(interval);
  }, []);

  const addLog = useCallback((msg) => {
    const now = new Date();
    const t = `${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
    setLogs((prev) => [...prev, { time: t, msg }]);
  }, []);

  const handleScan = () => {
    setScanning(true);
    setScanProgress(0);
    addLog(`Scanning — Mode: ${MODES[mode].label}, Prompt: "${prompt.slice(0, 30)}..."`);
    let p = 0;
    const interval = setInterval(() => {
      p += Math.random() * 12 + 3;
      if (p >= 100) {
        p = 100;
        clearInterval(interval);
        setScanning(false);
        addLog("Scan complete — Rendering results");
        neurons.forEach((n) => {
          n.activation = Math.random();
          n.weight = Math.random() * 2 - 1;
          n.anomaly = Math.random() > 0.85 ? Math.random() * 0.8 + 0.2 : 0;
        });
      }
      setScanProgress(Math.min(p, 100));
    }, 120);
  };

  const handlePerturb = (op) => {
    addLog(`Perturbation: ${op} on node ${stimNode}`);
    const node = neurons.find((n) => n.id === stimNode);
    if (node) {
      if (op === "Zero-out") node.activation = 0;
      else if (op === "Amplify 2×") node.activation = Math.min(node.activation * 2, 1);
      else if (op === "Invert") node.weight = -node.weight;
      else if (op === "Noise ±0.5") node.activation = Math.max(0, Math.min(1, node.activation + (Math.random() - 0.5)));
      else if (op === "Ablate") { node.activation = 0; node.weight = 0; node.anomaly = 0; }
      addLog(`Node ${stimNode} — New activation: ${node.activation.toFixed(4)}`);
    }
  };

  const handleModeChange = (m) => {
    setMode(m);
    addLog(`Mode switched: ${MODES[m].label}`);
  };

  return (
    <div style={{
      background: "#0a0c10", minHeight: "100vh", color: "#ccc",
      fontFamily: "'Courier New', monospace", display: "flex", flexDirection: "column"
    }}>
      {/* Top Bar */}
      <div style={{
        background: "linear-gradient(180deg, #0f1218 0%, #0a0c10 100%)",
        borderBottom: "1px solid rgba(100,170,136,0.2)", padding: "12px 20px",
        display: "flex", alignItems: "center", gap: "16px"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#00ffaa", boxShadow: "0 0 8px rgba(0,255,170,0.5)" }} />
          <span style={{ fontSize: "14px", fontWeight: "bold", color: "#66aa88", letterSpacing: "2px" }}>NEURAL MRI</span>
        </div>
        <span style={{ fontSize: "10px", color: "#445", letterSpacing: "1px" }}>Model Resonance Imaging</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: "8px", alignItems: "center" }}>
          <select value={model} onChange={(e) => { setModel(e.target.value); addLog(`Model: ${e.target.value}`); }}
            style={{ background: "#12151c", border: "1px solid rgba(100,170,136,0.2)", color: "#66aa88", padding: "4px 8px", fontSize: "10px", fontFamily: "'Courier New', monospace", borderRadius: "2px" }}>
            <option>Llama-3.2-3B</option>
            <option>Qwen-2.5-3B</option>
            <option>Gemma-2-2B</option>
            <option>Phi-3-mini-3.8B</option>
            <option>Mistral-7B-v0.3</option>
          </select>
        </div>
      </div>

      {/* Mode Tabs */}
      <div style={{ display: "flex", borderBottom: "1px solid rgba(100,170,136,0.1)", background: "#0c0e14" }}>
        {Object.entries(MODES).map(([key, m]) => (
          <button key={key} onClick={() => handleModeChange(key)} style={{
            flex: 1, padding: "10px 8px", background: mode === key ? "rgba(100,170,136,0.08)" : "transparent",
            border: "none", borderBottom: mode === key ? `2px solid ${m.color}` : "2px solid transparent",
            color: mode === key ? m.color : "#445", fontSize: "10px", fontFamily: "'Courier New', monospace",
            cursor: "pointer", transition: "all 0.3s", letterSpacing: "0.5px"
          }}>
            <div style={{ fontWeight: mode === key ? "bold" : "normal" }}>{m.label}</div>
            {mode === key && <div style={{ fontSize: "8px", marginTop: "2px", opacity: 0.7 }}>{m.desc}</div>}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left: Canvas */}
        <div style={{ flex: 1, position: "relative", display: "flex", flexDirection: "column" }}>
          <DicomHeader mode={mode} prompt={prompt} model={model} />
          <div style={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "flex-start", position: "relative", overflow: "hidden" }}>
            <ScanLines />
            {scanning && (
              <div style={{ position: "absolute", inset: 0, zIndex: 10, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)" }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ color: "#00ffaa", fontSize: "12px", marginBottom: "8px", letterSpacing: "2px" }}>SCANNING...</div>
                  <div style={{ width: "200px", height: "3px", background: "rgba(255,255,255,0.05)", borderRadius: "2px", overflow: "hidden" }}>
                    <div style={{ width: `${scanProgress}%`, height: "100%", background: "linear-gradient(90deg, #00ffaa, #44ddaa)", transition: "width 0.1s" }} />
                  </div>
                  <div style={{ color: "#445", fontSize: "9px", marginTop: "4px" }}>{Math.floor(scanProgress)}%</div>
                </div>
              </div>
            )}
            <NeuralCanvas mode={mode} neurons={neurons} connections={connections} stimNode={stimNode} setStimNode={setStimNode} time={time} />
          </div>
          {/* Prompt Input */}
          <div style={{ padding: "8px 12px", borderTop: "1px solid rgba(100,170,136,0.1)", display: "flex", gap: "8px", alignItems: "center" }}>
            <span style={{ fontSize: "9px", color: "#66aa88", whiteSpace: "nowrap" }}>PROMPT:</span>
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} style={{
              flex: 1, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(100,170,136,0.15)",
              color: "#aab", padding: "6px 10px", fontSize: "11px", fontFamily: "'Courier New', monospace", borderRadius: "2px", outline: "none"
            }} />
            <button onClick={handleScan} disabled={scanning} style={{
              background: scanning ? "#1a1c22" : "rgba(0,255,170,0.12)", border: "1px solid rgba(0,255,170,0.3)",
              color: "#00ffaa", padding: "6px 16px", fontSize: "10px", fontFamily: "'Courier New', monospace",
              cursor: scanning ? "default" : "pointer", borderRadius: "2px", letterSpacing: "1px"
            }}>
              {scanning ? "SCANNING..." : "▶ SCAN"}
            </button>
          </div>
          <LogPanel logs={logs} />
        </div>

        {/* Right Panel */}
        <div style={{ width: "240px", borderLeft: "1px solid rgba(100,170,136,0.12)", background: "#0b0d12", display: "flex", flexDirection: "column" }}>
          <ActivationBar neurons={neurons} mode={mode} />
          <div style={{ borderTop: "1px solid rgba(100,170,136,0.1)" }}>
            <StimPanel stimNode={stimNode} neurons={neurons} mode={mode} onPerturb={handlePerturb} />
          </div>
          {/* Color Legend */}
          <div style={{ marginTop: "auto", padding: "12px", borderTop: "1px solid rgba(100,170,136,0.1)", fontSize: "9px" }}>
            <div style={{ color: "#66aa88", marginBottom: "6px", letterSpacing: "1px" }}>LEGEND</div>
            {mode === "fMRI" && <>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "3px" }}><div style={{ width: "12px", height: "8px", background: "rgb(50,60,140)", borderRadius: "1px" }} /><span style={{ color: "#556" }}>Low activation</span></div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "3px" }}><div style={{ width: "12px", height: "8px", background: "rgb(180,100,40)", borderRadius: "1px" }} /><span style={{ color: "#556" }}>Medium</span></div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><div style={{ width: "12px", height: "8px", background: "rgb(240,80,30)", borderRadius: "1px" }} /><span style={{ color: "#556" }}>High activation</span></div>
            </>}
            {mode === "T1" && <div style={{ color: "#556" }}>Brightness = parameter count</div>}
            {mode === "T2" && <div style={{ color: "#556" }}>Blue intensity = weight magnitude</div>}
            {mode === "DTI" && <>
              <div style={{ color: "#556" }}>Color = directional encoding</div>
              <div style={{ color: "#556", marginTop: "2px" }}>Animated flow = active pathways</div>
            </>}
            {mode === "FLAIR" && <>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "3px" }}><div style={{ width: "12px", height: "8px", background: "#334", borderRadius: "1px" }} /><span style={{ color: "#556" }}>Normal</span></div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}><div style={{ width: "12px", height: "8px", background: "#ff4466", borderRadius: "1px" }} /><span style={{ color: "#556" }}>Anomaly detected</span></div>
            </>}
          </div>
        </div>
      </div>
    </div>
  );
}
