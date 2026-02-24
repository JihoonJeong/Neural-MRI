# Neural MRI Scanner — Implementation Specification
## Model Resonance Imaging for AI Interpretability

**Project Codename:** NeuralMRI  
**Full Name:** Neural MRI — Model Resonance Imaging  
**Version:** 0.1 (MVP)  
**Date:** 2026-02-24  
**Author:** JJ (Asia2G Capital / ModuLabs)

---

## 1. Executive Summary

Neural MRI Scanner는 오픈소스 LLM 내부를 뇌 MRI처럼 시각화하고, 실시간으로 자극(perturbation)을 가해 변화를 관찰할 수 있는 AI 모델 해석 도구(interpretability tool)다. MRI는 **Model Resonance Imaging**의 약자로, 의료 MRI(Magnetic Resonance Imaging)가 뇌의 내부를 들여다보듯 AI 모델 내부에서 특정 입력에 "공명(resonate)"하는 뉴런과 회로를 찾아 영상화한다는 의미를 담고 있다.

**핵심 아이디어:** 의료 영상(T1, T2, fMRI, DTI, FLAIR)의 멀티모달 스캔 패러다임을 그대로 AI 모델 내부 분석에 매핑한다. 연구자뿐 아니라 엔지니어, 의사결정자도 "이 모델 내부에서 무슨 일이 일어나는지" 직관적으로 파악할 수 있게 한다.

**대상 사용자:**
- AI 엔지니어 (모델 디버깅, 파인튜닝 문제 진단)
- 연구자 (mechanistic interpretability 연구 보조)
- 기술 리더/의사결정자 (모델 행동에 대한 직관 확보)

---

## 2. MRI Modality → AI Interpretability 매핑

이 프로젝트의 핵심 프레임워크. 각 의료 영상 기법이 AI 모델의 어떤 측면을 보여주는지 정의한다. 의료 MRI의 용어 체계를 AI 맥락으로 완전히 재정의하여 프로젝트 고유의 용어 세계관을 구축한다.

### Terminology Map

| 의료 원본 | Neural MRI 재정의 | 풀네임 | 의미 |
|-----------|-------------------|--------|------|
| MRI (Magnetic Resonance Imaging) | **MRI** | **Model Resonance Imaging** | AI 모델 내부 공명 영상 |
| T1-weighted | **T1** | **Topology Layer 1** | 1차 구조 — 정적 아키텍처 토폴로지 |
| T2-weighted | **T2** | **Tensor Layer 2** | 2차 구조 — 텐서(가중치) 분포 |
| fMRI (functional Magnetic Resonance Imaging) | **fMRI** | **functional Model Resonance Imaging** | 기능적 활성화 영상 |
| DTI (Diffusion Tensor Imaging) | **DTI** | **Data Tractography Imaging** | 데이터 흐름 경로 추적 |
| FLAIR (Fluid-Attenuated Inversion Recovery) | **FLAIR** | **Feature-Level Anomaly Identification & Reporting** | 피처 수준 이상 탐지 및 보고 |

### 2.1 T1 — Topology Layer 1 (Model Architecture)

| 항목 | 설명 |
|------|------|
| **의료 원본** | T1-weighted MRI: 조직의 해부학적 구조를 보여줌 |
| **AI 매핑** | 모델의 정적 구조 — 레이어 수, 각 레이어의 뉴런/head 수, 파라미터 카운트 |
| **시각화** | 각 레이어를 노드 클러스터로, 크기는 파라미터 수에 비례. 그레이스케일 톤 |
| **데이터 소스** | `model.config` 에서 직접 추출 (정적) |
| **인터랙션** | 호버 시 레이어 상세 정보 표시 (hidden_size, num_heads, intermediate_size 등) |

### 2.2 T2 — Tensor Layer 2 (Weight Distribution)

| 항목 | 설명 |
|------|------|
| **의료 원본** | T2-weighted MRI: T1과 다른 타이밍으로 다른 조직 대조를 보여줌 |
| **AI 매핑** | 가중치(weight)의 분포, magnitude, 통계적 특성 |
| **시각화** | 각 뉴런/head의 weight magnitude를 블루 스케일 히트맵으로 표현. 밝을수록 큰 가중치 |
| **데이터 소스** | `model.state_dict()`에서 각 레이어의 weight tensor → 통계 (mean, std, max, L2 norm) |
| **인터랙션** | 레이어별/head별 weight 히스토그램 표시. 이상치(outlier) 가중치 하이라이트 |

### 2.3 fMRI — functional Model Resonance Imaging (Activation Patterns)

| 항목 | 설명 |
|------|------|
| **의료 원본** | fMRI: 혈류 변화로 뇌의 활성화 영역을 실시간으로 보여줌 |
| **AI 매핑** | 특정 입력(prompt)에 대한 각 레이어/뉴런의 활성화(activation) 패턴 |
| **시각화** | Cool-to-Hot 컬러맵 (파랑→노랑→빨강). 활성화가 높은 뉴런이 "뜨겁게" 표시. 실시간 펄스 애니메이션 |
| **데이터 소스** | TransformerLens의 `run_with_cache()` → 각 레이어별 activation tensor |
| **인터랙션** | 프롬프트를 바꾸면 activation이 실시간으로 변화. 토큰별 step-through 가능 |
| **핵심 기술** | `hook_resid_post`, `hook_attn_out`, `hook_mlp_out` 에서 캐싱 |

### 2.4 DTI — Data Tractography Imaging (Circuit Tracing)

| 항목 | 설명 |
|------|------|
| **의료 원본** | DTI: 백질의 신경섬유 트랙을 추적하여 뇌 영역 간 연결을 보여줌 |
| **AI 매핑** | 정보가 어떤 경로(attention head → MLP → 다음 레이어)로 흐르는지 추적 |
| **시각화** | 방향별 색상 인코딩(directional color encoding). 유의미한 정보 흐름 경로만 굵은 곡선으로 표시. 흐름 방향 애니메이션 |
| **데이터 소스** | (1) Attention pattern: 각 head의 attention matrix. (2) Attribution patching: 각 컴포넌트의 출력 기여도 |
| **인터랙션** | 특정 출력 토큰 선택 시 해당 토큰에 가장 기여한 경로가 하이라이트됨 |
| **핵심 기술** | TransformerLens의 activation patching, attention pattern 추출 |

### 2.5 FLAIR — Feature-Level Anomaly Identification & Reporting (Bias & Hallucination Detection)

| 항목 | 설명 |
|------|------|
| **의료 원본** | FLAIR: 병변(lesion)을 강조하여 이상 부위를 명확하게 보여줌 |
| **AI 매핑** | 모델의 "문제 지점" — 할루시네이션, 편향, 불확실성이 높은 영역 |
| **시각화** | 정상 영역은 어둡게, 이상 영역은 빨간색/핑크색으로 펄스. 이상 점수에 따른 강도 |
| **데이터 소스** | (1) Logit lens: 중간 레이어의 예측이 최종 예측과 얼마나 다른지. (2) Entropy: 각 위치의 다음 토큰 예측 불확실성. (3) SAE feature 중 알려진 편향/할루시네이션 관련 feature의 활성화 |
| **인터랙션** | 이상 노드 클릭 시 해당 뉴런/feature의 상세 정보, 관련 학습 데이터 패턴 추정 |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (React)                   │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ MRI Canvas │  │ Mode Tabs │  │ Control Panels   │ │
│  │ (D3 / SVG) │  │ T1~FLAIR  │  │ Stim, Perturb,  │ │
│  │            │  │           │  │ Layer Summary    │ │
│  └─────┬─────┘  └───────────┘  └──────────────────┘ │
│        │  WebSocket (real-time activation stream)     │
├────────┼────────────────────────────────────────────┤
│        ▼          Backend (FastAPI + Python)          │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────┐ │
│  │ Model      │  │ Analysis  │  │ Perturbation     │ │
│  │ Manager    │  │ Engine    │  │ Engine           │ │
│  │ (load/     │  │ (Trans-   │  │ (activation      │ │
│  │  swap)     │  │  formerLens│ │  patching, etc.) │ │
│  └───────────┘  └───────────┘  └──────────────────┘ │
│        │                                              │
│  ┌─────▼───────────────────────────────────────────┐ │
│  │  Model Registry (HuggingFace Hub cache)          │ │
│  │  Llama-3.2-3B, Qwen-2.5-3B, Gemma-2-2B, etc.   │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 3.1 Frontend

| 항목 | 기술 |
|------|------|
| **Framework** | React 18+ (Vite) |
| **시각화 엔진** | D3.js (SVG 기반) — 뉴런/연결 렌더링 |
| **실시간 통신** | WebSocket (activation 스트리밍) |
| **상태 관리** | Zustand (경량) |
| **스타일** | Tailwind CSS + CSS Variables (DICOM 테마) |
| **애니메이션** | requestAnimationFrame (캔버스 펄스), CSS transitions (UI) |

### 3.2 Backend

| 항목 | 기술 |
|------|------|
| **서버** | FastAPI (Python 3.11+) |
| **모델 인트로스펙션** | TransformerLens (`HookedTransformer`) |
| **SAE 분석** | SAELens (선택사항, Phase 2) |
| **텐서 연산** | PyTorch 2.x |
| **모델 로딩** | HuggingFace `transformers` + `accelerate` |
| **WebSocket** | `fastapi[websockets]` |
| **시리얼라이즈** | `orjson` (대용량 텐서 데이터 직렬화) |

### 3.3 지원 모델 (MVP)

| 모델 | 파라미터 | TransformerLens 지원 | 우선순위 |
|------|---------|---------------------|---------|
| GPT-2 small (124M) | 124M | ✅ 공식 지원 | P0 (개발/테스트용) |
| GPT-2 medium (355M) | 355M | ✅ 공식 지원 | P0 |
| Pythia-1.4B | 1.4B | ✅ 공식 지원 | P0 |
| Gemma-2-2B | 2B | ✅ 지원 | P1 |
| Llama-3.2-3B | 3.21B | ⚠️ 커뮤니티 지원 | P1 |
| Qwen-2.5-3B | 3B | ⚠️ 커뮤니티/커스텀 | P1 |
| Mistral-7B-v0.3 | 7.24B | ⚠️ 커뮤니티 지원 | P2 (GPU 필요) |
| Phi-3-mini-3.8B | 3.8B | ⚠️ 커스텀 필요 | P2 |

> **참고:** TransformerLens는 GPT-2, Pythia 계열이 가장 안정적. Llama/Qwen 등은 `HookedTransformer.from_pretrained()` 호환성 확인 필요. 미지원 모델은 nnsight로 대체 가능.

---

## 4. API Design

### 4.1 REST Endpoints

```
POST   /api/model/load          모델 로드 (HuggingFace ID 또는 로컬 경로)
GET    /api/model/info           현재 로드된 모델의 구조 정보 (T1 데이터)
DELETE /api/model/unload         모델 언로드 (메모리 해제)

POST   /api/scan/structural      T1 스캔: 정적 구조 데이터 반환
POST   /api/scan/weights         T2 스캔: weight 통계 반환
POST   /api/scan/activation      fMRI 스캔: 프롬프트 기반 activation 반환
POST   /api/scan/circuits        DTI 스캔: attention + attribution 경로 반환
POST   /api/scan/anomaly         FLAIR 스캔: 이상 탐지 결과 반환

POST   /api/perturb/zero         특정 컴포넌트 zero-out
POST   /api/perturb/amplify      특정 컴포넌트 amplify (factor)
POST   /api/perturb/ablate       특정 컴포넌트 ablate (제거)
POST   /api/perturb/inject       특정 위치에 activation 주입
POST   /api/perturb/patch        activation patching (causal tracing)
POST   /api/perturb/reset        perturbation 초기화 (원본 복원)

GET    /api/features/list        SAE feature 목록 (Phase 2)
POST   /api/features/activate    특정 SAE feature 활성화/비활성화 (Phase 2)
```

### 4.2 WebSocket Endpoint

```
WS  /ws/stream

클라이언트 → 서버:
{
  "type": "scan_stream",
  "mode": "fMRI",
  "prompt": "The capital of France is",
  "token_step": true       // true면 토큰별로 스트리밍
}

서버 → 클라이언트:
{
  "type": "activation_frame",
  "token_idx": 3,
  "token": "capital",
  "layers": [
    {
      "layer_id": "blocks.0.attn",
      "type": "attention",
      "activations": [0.12, 0.87, ...],   // 요약된 per-head 값
      "attention_pattern": [[...], ...]     // DTI 모드 시 포함
    },
    ...
  ]
}
```

### 4.3 요청/응답 스키마 예시

#### POST /api/scan/activation

**Request:**
```json
{
  "prompt": "The Eiffel Tower is located in",
  "layers": "all",              // 또는 ["blocks.3.mlp", "blocks.4.attn"]
  "aggregation": "l2_norm",     // "l2_norm" | "max" | "mean" | "raw"
  "include_residual": true,
  "token_positions": "all"      // 또는 [0, 1, 5]  (특정 토큰 위치)
}
```

**Response:**
```json
{
  "model": "gpt2-small",
  "prompt_tokens": ["The", " Eiff", "el", " Tower", " is", " located", " in"],
  "scan_mode": "fMRI",
  "data": {
    "embed": {
      "type": "embedding",
      "shape": [7, 768],
      "activations_summary": [0.45, 0.52, 0.48, 0.61, 0.33, 0.55, 0.41]
    },
    "blocks.0.attn": {
      "type": "attention",
      "num_heads": 12,
      "per_head_activation": [0.12, 0.87, 0.34, ...],
      "attention_patterns": {
        "shape": [12, 7, 7],
        "data_url": "/api/tensor/attn_0_patterns"
      }
    },
    "blocks.0.mlp": {
      "type": "mlp",
      "activation_summary": [0.22, 0.91, 0.45, ...],
      "top_neurons": [
        {"idx": 1247, "activation": 3.82, "label": null},
        {"idx": 892, "activation": 2.91, "label": null}
      ]
    }
  },
  "metadata": {
    "compute_time_ms": 342,
    "gpu_memory_mb": 1240
  }
}
```

#### POST /api/perturb/patch

**Request:**
```json
{
  "prompt": "The Eiffel Tower is located in",
  "target_token_idx": -1,
  "target_component": "blocks.5.mlp",
  "method": "zero",
  "compare_logits": true
}
```

**Response:**
```json
{
  "original_prediction": {
    "token": " Paris",
    "logit": 12.34,
    "prob": 0.87
  },
  "perturbed_prediction": {
    "token": " the",
    "logit": 8.12,
    "prob": 0.23
  },
  "logit_diff": -4.22,
  "affected_components": [
    {"id": "blocks.5.mlp", "impact_score": 0.92},
    {"id": "blocks.6.attn.head_3", "impact_score": 0.45}
  ]
}
```

---

## 5. Frontend Specification

### 5.1 전체 레이아웃

```
┌─ Top Bar ──────────────────────────────────────────────────┐
│ [●] NEURAL MRI  │  Model Resonance Imaging  │  Model: [Dropdown ▾]  │  GPU: 2.1GB/8GB │
├─ Mode Tabs ────────────────────────────────────────────────┤
│ [ T1 Topology ] [ T2 Tensor ] [ fMRI ]                    │
│ [ DTI ] [ FLAIR ]                                          │
├────────────────────────────────┬────────────────────────────┤
│                                │  Layer Summary             │
│   DICOM Header                 │  ├─ Embed:  ████░░ 0.45  │
│   ┌──────────────────────┐     │  ├─ Attn1:  ██████ 0.87  │
│   │                      │     │  ├─ MLP1:   ███░░░ 0.34  │
│   │    Main Scan Canvas   │     │  └─ ...                   │
│   │    (SVG/D3)           │     │                            │
│   │                      │     │  ◉ Stimulation Panel       │
│   │    - neurons          │     │  ID: blocks.3.attn.h7     │
│   │    - connections      │     │  Activation: 0.8721       │
│   │    - flow animations  │     │  [Zero] [Amp] [Inv]       │
│   │                      │     │  [Noise] [Ablate]          │
│   └──────────────────────┘     │                            │
│                                │  Comparison Panel          │
│   PROMPT: [________________]   │  Original: "Paris" (0.87)  │
│   [▶ SCAN] [⏸ PAUSE] [↺ RESET]│  Perturbed: "the" (0.23)  │
│                                │                            │
├── Log Panel ───────────────────┴────────────────────────────┤
│ [00:12] Scan complete — Mode: fMRI, 7 tokens processed     │
│ [00:14] Perturbation: Zero-out on blocks.3.attn.head_7     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 디자인 시스템

**테마: "Medical Dark" — DICOM 뷰어 + 수술실 모니터 미학**

```css
/* Color Palette */
--bg-primary:     #0a0c10;     /* 거의 검정, 약간 블루 */
--bg-secondary:   #0c0e14;     /* 패널 배경 */
--bg-surface:     #12151c;     /* 카드/입력 배경 */
--border:         rgba(100, 170, 136, 0.15);  /* 의료 그린 보더 */
--text-primary:   #66aa88;     /* 의료 그린 텍스트 */
--text-secondary: #556;        /* 회색 보조 텍스트 */
--text-data:      #aabbcc;     /* 데이터 값 */
--accent-active:  #00ffaa;     /* 선택/활성 하이라이트 */
--scan-line:      rgba(255, 255, 255, 0.04);  /* 스캔라인 오버레이 */

/* Mode-specific Colors (T1=Topology, T2=Tensor, fMRI=functional MRI, DTI=Data Tractography, FLAIR=Feature-Level Anomaly) */
--t1-base:     #8899aa;   --t1-accent:  #e0e0e0;
--t2-base:     #4488cc;   --t2-accent:  #aaccee;
--fmri-cold:   #1a2a5a;   --fmri-warm:  #cc8830;  --fmri-hot: #ff4420;
--dti-green:   #44ddaa;   --dti-purple: #8866ff;
--flair-normal:#334;       --flair-hot:  #ff4466;

/* Typography — Monospace only */
--font-primary: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
--font-size-xs:  9px;   /* 로그, 범례 */
--font-size-sm:  10px;  /* 라벨, 탭 */
--font-size-md:  11px;  /* 본문 데이터 */
--font-size-lg:  14px;  /* 타이틀 */
```

**필수 비주얼 요소:**

1. **스캔라인 오버레이** — 캔버스 위에 1px 간격의 수평선. opacity 0.03~0.05. CRT 모니터 느낌
2. **DICOM 헤더** — 캔버스 상단에 의료 영상 스타일의 메타데이터 (모델명, 시퀀스, 날짜/시간, FOV, "Model Resonance Imaging" 표기)
3. **Vignette 효과** — 캔버스 가장자리가 살짝 어두워지는 효과
4. **Pulse 애니메이션** — fMRI 모드에서 활성화된 뉴런의 크기와 밝기가 주기적으로 미세하게 변동
5. **Flow 애니메이션** — DTI 모드에서 연결선을 따라 작은 입자/밝기가 흐르는 효과

### 5.3 Canvas 렌더링 사양

#### 뉴런(노드) 렌더링

```
각 뉴런은 원(circle)으로 표현.

위치 결정:
- Y축: 레이어 순서 (상단 = embedding, 하단 = output)
- X축: 같은 레이어 내 뉴런들이 수평으로 분포
- 레이어 간 간격: 60~80px
- 뉴런 간 간격: 레이어 내 뉴런 수에 따라 자동 조정

크기 결정 (모드별):
- T1: 파라미터 수에 비례 (4~10px 반지름)
- T2: weight magnitude에 비례
- fMRI: base 크기 × (0.5 + activation × 1.0) × pulse_factor
- DTI: 일정 크기, 색상으로 방향 인코딩
- FLAIR: 정상=작게, 이상=크게 + 펄스

색상 결정 (모드별):
- T1: 그레이스케일 (rgb(v,v,v+10), v = 160~220)
- T2: 블루 스케일 (weight 작으면 어두운 남색, 크면 밝은 하늘색)
- fMRI: cool-to-hot colormap
  - activation < 0.3: 어두운 파랑 rgb(30+a*80, 30+a*100, 80+a*120)
  - activation 0.3~0.6: 노랑/주황 rgb(a*200, a*160, 40+a*60)
  - activation > 0.6: 빨강/흰 rgb(200+a*55, a*120, a*30)
- DTI: HSL, hue = (x/width)*120 + (y/height)*120, saturation 70%, lightness 55%
- FLAIR: 정상=rgb(60,65,75), 이상=rgb(255, 50+a*60, 80+a*40) 펄스
```

#### 연결(엣지) 렌더링

```
연결은 레이어 간 정보 흐름을 표현.

모드별 표현:
- T1: 얇은 회색 선 (opacity 0.15, width 0.5)
- T2: weight 크기에 따라 opacity와 두께 변화
- fMRI: 양끝 뉴런의 평균 activation에 따라 색상/두께 변화
  - 높은 activation: 핫 컬러, 굵은 선
  - 낮은 activation: 거의 투명
- DTI: 유의미한 pathway만 표시
  - 곡선(quadratic bezier) 사용
  - 방향에 따른 HSL 색상
  - flow 애니메이션 (sin wave로 opacity 변동)
  - 비-pathway 연결은 거의 투명
- FLAIR: 이상 노드에 연결된 엣지만 빨간색 하이라이트
```

#### 토폴로지 레이아웃 옵션 (Phase 2 이후)

```
MVP: 수직 레이어 스택 (위→아래)
Phase 2: 사용자가 레이아웃 모드를 선택 가능
  - Stack (기본): 수직 레이어 스택
  - Brain: 타원형 뇌 모양으로 감싸서 배치 (코르티컬 매핑 비유)
  - Network: force-directed 그래프 (D3 force simulation)
  - Radial: 중심에서 바깥으로 레이어가 확장
```

### 5.4 인터랙션 사양

#### 뉴런 선택 (Stimulation Mode)

```
1. 뉴런 클릭 → 선택 상태 진입
2. 선택된 뉴런 주위에 동심원 애니메이션 (green glow)
3. 우측 패널에 상세 정보 표시:
   - Node ID (layer.component.index)
   - Layer type (attention / mlp / embedding / output)
   - 현재 모드의 주요 값 (activation, weight, anomaly score)
   - Top-k 연결된 뉴런 (가장 강한 연결)
4. Perturbation 버튼 활성화:
   - Zero-out: 해당 컴포넌트 출력을 0으로
   - Amplify 2×: 출력을 2배로
   - Invert: 출력 부호 반전
   - Noise ±σ: 가우시안 노이즈 추가
   - Ablate: 완전 제거 (zero + gradient 차단)
5. Perturbation 적용 시:
   - 백엔드에 perturbation 요청 → 새로운 activation 수신
   - 캔버스 전체가 0.3초간 재스캔 애니메이션
   - 변화된 부분이 잠시 하이라이트
   - 우측 Comparison Panel에 before/after 표시
```

#### 프롬프트 입력 & 스캔

```
1. 프롬프트 입력 → SCAN 버튼 클릭 (또는 Enter)
2. 스캔 프로그레스 바 표시 (실제 백엔드 처리 시간 반영)
3. WebSocket으로 토큰별 activation 스트리밍
4. 토큰 step-through 가능:
   - 프롬프트 영역에 각 토큰이 칩(chip)으로 표시
   - 토큰 칩 클릭 → 해당 토큰 시점의 activation만 표시
   - ← → 화살표로 토큰 간 이동
   - 자동 재생 (0.5초 간격)
```

#### 모드 전환

```
1. 모드 탭 클릭 → 0.3초 크로스페이드 전환
2. 동일한 토폴로지(뉴런 위치)를 유지하면서 색상/크기/연결 표현만 변경
3. 이는 실제 MRI에서 같은 환자의 T1→fMRI 전환과 동일한 경험
```

### 5.5 반응형 고려사항

```
- 최소 지원 해상도: 1280×720
- 권장 해상도: 1920×1080
- 캔버스 크기: 컨테이너에 맞게 스케일링 (SVG viewBox 사용)
- 모바일: 미지원 (데스크톱 전용 도구)
```

---

## 6. Backend Specification

### 6.1 모델 매니저 (ModelManager)

```python
class ModelManager:
    """모델 로딩, 스왑, 메모리 관리"""

    def load_model(self, model_id: str, device: str = "auto") -> ModelInfo:
        """
        HuggingFace 모델을 TransformerLens HookedTransformer로 로드.
        - model_id: "gpt2", "EleutherAI/pythia-1.4b", "meta-llama/Llama-3.2-3B" 등
        - device: "cpu", "cuda", "mps", "auto"
        - 반환: 모델 메타데이터 (레이어 수, hidden size, head 수 등)
        """

    def unload_model(self) -> None:
        """현재 모델 언로드 + GPU 메모리 해제 (gc + torch.cuda.empty_cache)"""

    def get_model_info(self) -> ModelInfo:
        """현재 로드된 모델의 아키텍처 정보 반환 (T1 데이터)"""

    def get_model(self) -> HookedTransformer:
        """현재 로드된 모델 인스턴스 반환"""
```

### 6.2 분석 엔진 (AnalysisEngine)

```python
class AnalysisEngine:
    """각 스캔 모드에 대한 분석 수행"""

    def scan_structural(self) -> StructuralData:
        """T1: model.cfg에서 정적 구조 추출"""

    def scan_weights(self, layers: list[str] | None = None) -> WeightData:
        """T2: state_dict에서 weight 통계 추출"""

    def scan_activation(self, prompt: str, **kwargs) -> ActivationData:
        """
        fMRI: prompt에 대한 activation 캐시.
        TransformerLens run_with_cache() 사용.

        핵심 구현:
        logits, cache = model.run_with_cache(prompt)

        추출 대상 hook points:
        - hook_embed: 임베딩 레이어
        - blocks.{i}.hook_resid_pre: 각 블록 입력 residual
        - blocks.{i}.attn.hook_result: attention 출력
        - blocks.{i}.hook_mlp_out: MLP 출력
        - blocks.{i}.hook_resid_post: 각 블록 출력 residual

        aggregation 옵션:
        - "l2_norm": L2 norm per position (스칼라)
        - "max": max absolute value
        - "mean": mean absolute value
        - "raw": 전체 텐서 반환 (대용량, 선택적)
        """

    def scan_circuits(self, prompt: str, target_token: int = -1) -> CircuitData:
        """
        DTI: attention pattern + attribution 경로 추출.

        (1) Attention Pattern:
        _, cache = model.run_with_cache(prompt)
        attn_patterns = cache["blocks.{i}.attn.hook_pattern"]
        → shape: [num_heads, seq_len, seq_len]

        (2) Attribution (간이 버전):
        각 head/mlp의 출력을 zero-out 했을 때 target logit 변화량 계산.
        → 절대값이 큰 컴포넌트 = 중요 경로
        """

    def scan_anomaly(self, prompt: str) -> AnomalyData:
        """
        FLAIR: 이상 탐지.

        (1) Logit Lens:
        각 중간 레이어의 residual stream을 unembed하여
        중간 예측 vs 최종 예측의 KL divergence 계산.
        큰 divergence = 해당 레이어에서 "생각이 크게 바뀜" = 잠재적 이상

        (2) Entropy:
        각 위치의 logit에서 softmax → entropy 계산.
        높은 entropy = 모델이 불확실 = 할루시네이션 위험

        (3) 이상 점수:
        anomaly_score = α * normalized_kl_div + β * normalized_entropy
        α = 0.6, β = 0.4 (튜닝 가능)
        """
```

### 6.3 Perturbation 엔진 (PerturbationEngine)

```python
class PerturbationEngine:
    """모델 내부에 자극/변형을 가하고 결과를 비교"""

    def zero_out(self, component: str, prompt: str) -> PerturbResult:
        """
        특정 컴포넌트의 출력을 0으로 만들고 재실행.

        구현:
        def zero_hook(value, hook):
            value[:, :, :] = 0  # 또는 특정 head만
            return value

        model.run_with_hooks(prompt, fwd_hooks=[(component, zero_hook)])
        """

    def amplify(self, component: str, factor: float, prompt: str) -> PerturbResult:
        """출력에 factor를 곱하여 증폭"""

    def ablate(self, component: str, prompt: str) -> PerturbResult:
        """컴포넌트를 완전히 제거 (mean ablation: 평균값으로 대체)"""

    def inject_activation(self, component: str, values: list, prompt: str) -> PerturbResult:
        """특정 activation 값을 직접 주입"""

    def activation_patch(
        self,
        clean_prompt: str,
        corrupt_prompt: str,
        component: str
    ) -> PatchResult:
        """
        Activation Patching (Causal Tracing).

        clean_prompt의 특정 컴포넌트 activation을
        corrupt_prompt 실행 중에 교체하여 복구 정도를 측정.

        구현:
        _, clean_cache = model.run_with_cache(clean_prompt)
        clean_activation = clean_cache[component]

        def patch_hook(value, hook):
            value[:] = clean_activation
            return value

        patched_logits = model.run_with_hooks(
            corrupt_prompt,
            fwd_hooks=[(component, patch_hook)]
        )

        recovery = (patched_logit - corrupt_logit) / (clean_logit - corrupt_logit)
        """

    def compare_results(self, original: Logits, perturbed: Logits) -> ComparisonData:
        """원본과 변형 결과 비교: top-k 예측, logit diff, KL divergence"""
```

### 6.4 데이터 요약 전략

대용량 텐서를 프론트엔드로 전송할 때의 요약 전략:

```
문제: GPT-2 small만 해도 단일 프롬프트에 대한 전체 activation cache가 수백MB.

해결:
1. 기본 응답: 레이어별/head별 요약 통계만 전송 (L2 norm, max, mean → 스칼라 배열)
2. 온디맨드: 사용자가 특정 레이어/head를 선택하면 해당 부분만 상세 데이터 전송
3. 어텐션 패턴: full attention matrix는 요청 시에만 전송 (shape: [heads, seq, seq])
4. 스트리밍: 토큰별 step-through 시 각 토큰의 데이터만 증분 전송
5. 캐싱: 동일 프롬프트에 대한 캐시는 서버 메모리에 보관 (LRU, 최대 5개 프롬프트)
```

---

## 7. Implementation Phases

### Phase 0: Foundation (1~2주)

```
목표: 프로젝트 구조 셋업 + GPT-2 small로 T1/T2 모드 작동

Backend:
- [ ] FastAPI 프로젝트 셋업 (poetry/uv 기반 dependency 관리)
- [ ] ModelManager 구현 (GPT-2 small 로드)
- [ ] scan_structural() 구현 → T1 데이터 반환
- [ ] scan_weights() 구현 → T2 데이터 반환
- [ ] 기본 REST API 엔드포인트 (/model/load, /model/info, /scan/structural, /scan/weights)

Frontend:
- [ ] Vite + React 프로젝트 셋업
- [ ] DICOM 테마 CSS 변수 정의
- [ ] 기본 레이아웃 구현 (Top Bar, Mode Tabs, Canvas, Panels)
- [ ] T1 Canvas 렌더링: 모델 구조를 노드/엣지로 시각화
- [ ] T2 Canvas 렌더링: weight 히트맵
- [ ] Model selector dropdown

테스트:
- [ ] GPT-2 small 로드 → T1 데이터 표시 → T2 모드 전환 검증
```

### Phase 1: Core Scanning (2~3주)

```
목표: fMRI + DTI 모드 작동. 프롬프트 입력 → activation 시각화

Backend:
- [ ] TransformerLens 통합 (HookedTransformer.from_pretrained)
- [ ] scan_activation() 구현 → fMRI 데이터 반환
- [ ] scan_circuits() 구현 → DTI 데이터 반환
- [ ] WebSocket 엔드포인트 (토큰별 activation 스트리밍)
- [ ] 데이터 요약/직렬화 파이프라인 (orjson)

Frontend:
- [ ] fMRI Canvas: cool-to-hot 컬러맵, 펄스 애니메이션
- [ ] DTI Canvas: 곡선 경로, 방향별 색상, flow 애니메이션
- [ ] Prompt 입력 UI + SCAN 버튼 + 프로그레스 바
- [ ] 토큰 step-through UI (토큰 칩 + 화살표 내비게이션)
- [ ] Layer Summary 바 차트 (모드별 적응)
- [ ] WebSocket 연결 + 실시간 업데이트

테스트:
- [ ] "The capital of France is" → fMRI에서 "France" 토큰 시 관련 뉴런 활성화 확인
- [ ] DTI에서 유의미한 information flow 경로 시각화 확인
```

### Phase 2: Perturbation + FLAIR (2~3주)

```
목표: 자극/변형 실험 + 이상 탐지

Backend:
- [ ] PerturbationEngine 전체 구현 (zero, amplify, ablate, inject, patch)
- [ ] scan_anomaly() 구현 (logit lens + entropy)
- [ ] compare_results() 구현 (before/after 비교)
- [ ] Activation patching (causal tracing) 구현

Frontend:
- [ ] FLAIR Canvas: 이상 영역 하이라이트, 펄스 애니메이션
- [ ] Stimulation Panel: 뉴런 클릭 → 상세 정보 + perturbation 버튼
- [ ] Comparison Panel: 원본 vs 변형 결과 나란히 표시
- [ ] Perturbation 적용 시 재스캔 애니메이션
- [ ] Reset 기능 (모든 perturbation 초기화)
- [ ] 스캔라인 오버레이 + vignette 효과

테스트:
- [ ] 특정 attention head zero-out → 예측 변화 확인
- [ ] "The Eiffel Tower is in" → 사실 관련 컴포넌트 ablation → 할루시네이션 유도 확인
- [ ] FLAIR에서 entropy가 높은 위치가 올바르게 하이라이트되는지 확인
```

### Phase 3: Polish + Multi-Model (2주)

```
목표: 다중 모델 지원 + UX 완성

Backend:
- [ ] Pythia-1.4B, Gemma-2-2B 지원 추가 및 테스트
- [ ] Llama-3.2-3B 지원 (TransformerLens 호환성 확인, 필요 시 nnsight 대체)
- [ ] 모델 스왑 시 메모리 관리 최적화
- [ ] API 응답 캐싱 레이어

Frontend:
- [ ] 모드 전환 크로스페이드 애니메이션
- [ ] 뉴런 호버 툴팁
- [ ] 전체 스캔라인 + CRT 미학 완성
- [ ] 성능 최적화 (large graph에서 60fps 유지)
- [ ] 에러/로딩 상태 UX

테스트:
- [ ] 모델 간 스왑 시 메모리 누수 없음 확인
- [ ] 3B 모델에서 전체 스캔 파이프라인 e2e 확인
```

### Phase 4: Advanced Features (향후)

```
- [ ] SAE Feature 탐색기 (SAELens 통합)
- [ ] Brain 레이아웃 모드 (코르티컬 매핑)
- [ ] Multi-prompt 비교 (같은 모델에 다른 입력 시 activation 차이)
- [ ] 시계열 녹화/재생 (스캔 세션 저장)
- [ ] Export: 스캔 결과를 이미지/영상으로 내보내기
- [ ] 협업: 여러 사용자가 같은 스캔 세션을 공유
- [ ] 자동 진단: "이 모델은 이런 문제가 있을 수 있습니다" 보고서 생성
```

---

## 8. Development Environment

### 8.1 필수 요구사항

```
- Python 3.11+
- Node.js 20+
- GPU: NVIDIA GPU with 8GB+ VRAM (권장). CPU 전용도 가능 (GPT-2 small 한정)
- CUDA 12.x (GPU 사용 시)
- 메모리: 16GB+ RAM
```

### 8.2 Backend 의존성

```toml
[project]
name = "neural-mri"
requires-python = ">=3.11"

dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "websockets>=12.0",
    "transformer-lens>=2.0",
    "torch>=2.2",
    "transformers>=4.40",
    "accelerate>=0.28",
    "sae-lens>=3.0",         # Phase 2
    "orjson>=3.9",
    "numpy>=1.26",
    "pydantic>=2.6",
]
```

### 8.3 Frontend 의존성

```json
{
  "dependencies": {
    "react": "^18.3",
    "react-dom": "^18.3",
    "d3": "^7.9",
    "zustand": "^4.5",
    "use-websocket": "^4.8"
  },
  "devDependencies": {
    "vite": "^5.4",
    "@vitejs/plugin-react": "^4.2",
    "tailwindcss": "^3.4",
    "autoprefixer": "^10.4",
    "postcss": "^8.4"
  }
}
```

### 8.4 프로젝트 구조

```
neural-mri/
├── README.md
├── docker-compose.yml
│
├── backend/
│   ├── pyproject.toml
│   ├── neural_mri/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry
│   │   ├── config.py            # 설정 (모델 경로, 캐시, GPU)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes_model.py  # /api/model/* 라우트
│   │   │   ├── routes_scan.py   # /api/scan/* 라우트
│   │   │   ├── routes_perturb.py # /api/perturb/* 라우트
│   │   │   └── ws_stream.py     # WebSocket 핸들러
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── model_manager.py
│   │   │   ├── analysis_engine.py
│   │   │   └── perturbation_engine.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── model.py         # ModelInfo, ModelConfig
│   │   │   ├── scan.py          # ActivationData, CircuitData 등
│   │   │   └── perturb.py       # PerturbResult, PatchResult
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── tensor_summary.py # 텐서 → 요약 변환
│   │       └── serialization.py  # orjson 커스텀 직렬화
│   └── tests/
│       ├── test_model_manager.py
│       ├── test_analysis.py
│       └── test_perturbation.py
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── index.html
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── theme/
│   │   │   ├── variables.css    # DICOM 테마 CSS 변수
│   │   │   └── globals.css
│   │   ├── store/
│   │   │   ├── useModelStore.js
│   │   │   ├── useScanStore.js
│   │   │   └── usePerturbStore.js
│   │   ├── components/
│   │   │   ├── TopBar.jsx
│   │   │   ├── ModeTabs.jsx
│   │   │   ├── DicomHeader.jsx
│   │   │   ├── ScanCanvas/
│   │   │   │   ├── ScanCanvas.jsx       # 메인 SVG 캔버스
│   │   │   │   ├── NeuronRenderer.jsx   # 뉴런 렌더링 로직
│   │   │   │   ├── ConnectionRenderer.jsx # 엣지 렌더링 로직
│   │   │   │   ├── ScanLineOverlay.jsx  # CRT 스캔라인
│   │   │   │   └── colorMaps.js         # 모드별 색상 함수
│   │   │   ├── Panels/
│   │   │   │   ├── LayerSummary.jsx     # 레이어별 막대 차트
│   │   │   │   ├── StimPanel.jsx        # 뉴런 선택 + perturbation
│   │   │   │   ├── ComparisonPanel.jsx  # before/after 비교
│   │   │   │   └── LogPanel.jsx         # 하단 로그
│   │   │   ├── PromptInput.jsx
│   │   │   └── TokenStepper.jsx         # 토큰별 step-through
│   │   ├── hooks/
│   │   │   ├── useWebSocket.js
│   │   │   └── useAnimationFrame.js
│   │   └── api/
│   │       ├── client.js                # REST API 클라이언트
│   │       └── ws.js                    # WebSocket 클라이언트
│   └── public/
│       └── fonts/                       # JetBrains Mono
│
└── docs/
    ├── SPEC.md                          # 이 문서
    ├── API.md                           # API 상세 문서
    └── ARCHITECTURE.md                  # 아키텍처 다이어그램
```

---

## 9. Key Technical Decisions & Risks

### 9.1 TransformerLens 호환성

```
리스크: TransformerLens는 GPT-2, Pythia 등 일부 모델만 공식 지원.
       Llama, Qwen 등은 커뮤니티 구현에 의존하며 버전에 따라 깨질 수 있음.

대응:
1. MVP는 GPT-2 small/medium + Pythia로 시작 (확실한 지원)
2. 새 모델 추가 시 from_pretrained() 호환성 테스트 스크립트 작성
3. TransformerLens 미지원 모델은 nnsight 백엔드로 폴백
4. 모델별 hook point 이름이 다를 수 있으므로 추상화 레이어 필요
```

### 9.2 성능

```
리스크: 3B+ 모델의 full activation cache가 수GB에 달할 수 있음.

대응:
1. 요약 우선 전략: 전체 텐서 대신 per-layer/per-head 통계만 기본 전송
2. Lazy loading: 사용자가 특정 레이어 선택 시에만 상세 데이터 전송
3. 서버사이드 캐싱: 동일 프롬프트에 대한 캐시 유지 (LRU 5개)
4. 토큰 스트리밍: 전체 시퀀스를 한번에 처리하되, 프론트엔드에는 토큰별 전송
5. GPU 메모리: 모델 + 캐시가 VRAM 초과 시 자동으로 CPU 오프로드
```

### 9.3 시각화 성능

```
리스크: 노드/엣지가 수백 개일 때 SVG 렌더링이 느려질 수 있음.

대응:
1. 집약 표현: 개별 뉴런이 아닌 "head" 또는 "layer component" 단위로 노드 표현
   (GPT-2 small: 12 layers × 3 components = ~36 nodes + embedding + output)
2. Viewport culling: 화면에 보이는 노드만 렌더링
3. 엣지 간소화: 모드에 따라 비활성 엣지를 아예 렌더링하지 않음
4. Canvas 전환: SVG 성능 한계 시 WebGL (Three.js) 또는 Canvas 2D로 전환
```

### 9.4 Perturbation 안전성

```
리스크: perturbation이 모델 weight 자체를 수정하면 복구가 어려움.

대응:
1. run_with_hooks()만 사용: 모델 weight는 절대 수정하지 않음. Hook으로 activation만 변형.
2. Reset 버튼: 모든 hook을 제거하고 원본 상태로 복귀
3. 모든 perturbation은 stateless: 각 요청마다 새로 hook을 설정
```

---

## 10. Success Metrics

### MVP (Phase 0~2 완료 기준)

```
1. GPT-2 small에 대해 5개 모드 모두 작동
2. 프롬프트 입력 → 스캔 완료까지 2초 이내 (GPU 기준)
3. 토큰 step-through가 smooth하게 작동 (프레임 드롭 없이)
4. perturbation 적용 → 결과 비교가 1초 이내
5. 모드 전환 시 토폴로지 유지하면서 0.3초 이내 전환
```

### 확장 (Phase 3 이후)

```
1. 3B 모델에서 전체 파이프라인 5초 이내
2. 최소 3개 이상의 오픈소스 모델 지원
3. Activation patching (causal tracing) 시각화가 논문 Figure 수준
```

---

## 11. References

### 핵심 라이브러리

- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- SAELens: https://github.com/jbloomAus/SAELens
- nnsight: https://github.com/ndif-team/nnsight

### 핵심 논문/자료

- Elhage et al. (2022) "Toy Models of Superposition" — 중첩(superposition) 이론
- Wang et al. (2022) "Interpretability in the Wild: IOI Circuit" — 회로 분석
- Meng et al. (2022) "ROME: Rank-One Model Editing" — 사실 저장 위치 추적
- Anthropic (2024) "Scaling Monosemanticity" — SAE feature 추출
- Neel Nanda's TransformerLens tutorials: https://neelnanda.io/

### 영감

- 3D Slicer (의료 영상 시각화): https://www.slicer.org/
- FreeSurfer (뇌 영상 분석): https://surfer.nmr.mgh.harvard.edu/
- Neuronpedia (SAE feature 탐색기): https://www.neuronpedia.org/

---

*End of Specification*
