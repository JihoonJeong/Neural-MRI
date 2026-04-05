# Neural MRI Scanner — 개발자 가이드

> Neural MRI v2의 API 레퍼런스, 아키텍처, 확장 가이드.

---

## 목차

1. [아키텍처](#아키텍처)
2. [프로젝트 구조](#프로젝트-구조)
3. [API 레퍼런스](#api-레퍼런스)
4. [백엔드 핵심 모듈](#백엔드-핵심-모듈)
5. [프론트엔드 아키텍처](#프론트엔드-아키텍처)
6. [Neural MRI 확장하기](#neural-mri-확장하기)
7. [환경 변수 & 설정](#환경-변수--설정)
8. [테스트](#테스트)

---

## 아키텍처

```
┌─────────────────────────────────────────────────┐
│                   프론트엔드                      │
│  React 18 + TypeScript + D3.js + Zustand         │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ ScanCanvas│ │ModeTabs  │ │ Panels (Perturb, │ │
│  │ (D3 시각화)│ │(T1-FLAIR)│ │ CausalTrace,SAE) │ │
│  ├──────────┤ ├──────────┤ ├──────────────────┤ │
│  │  Emotion  │ │   SAE    │ │   SAE Providers  │ │
│  │  Engine   │ │ Manager  │ │ (Lens+EleutherAI)│ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│            ↕ REST + WebSocket ↕                  │
├─────────────────────────────────────────────────┤
│                   백엔드                          │
│  FastAPI + TransformerLens + PyTorch             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Analysis  │ │  Model   │ │   Perturbation   │ │
│  │  Engine   │ │ Manager  │ │     Engine        │ │
│  ├──────────┤ ├──────────┤ ├──────────────────┤ │
│  │  Emotion  │ │   SAE    │ │    Battery       │ │
│  │  Engine   │ │ Providers│ │    Engine         │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│            ↕ TransformerLens ↕                   │
├─────────────────────────────────────────────────┤
│              모델 가중치                           │
│  HuggingFace Hub / 로컬 캐시                      │
└─────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
Neural-MRI/
├── backend/
│   ├── neural_mri/
│   │   ├── main.py              # FastAPI 앱, 싱글턴, 생명주기
│   │   ├── config.py            # 설정 (NMRI_* 환경변수)
│   │   ├── api/
│   │   │   ├── routes_model.py  # 모델 로드/언로드/목록/검색
│   │   │   ├── routes_scan.py   # T1/T2/fMRI/DTI/FLAIR 스캔
│   │   │   ├── routes_perturb.py # 제로/증폭/절제/패치/인과추적
│   │   │   ├── routes_sae.py    # SAE 정보/스캔/지원
│   │   │   ├── routes_emotion.py # 감정 추출/조작/목록
│   │   │   ├── routes_battery.py # 기능 테스트 배터리
│   │   │   ├── routes_report.py # 진단 보고서 생성
│   │   │   └── ws_stream.py     # WebSocket 토큰 스트리밍
│   │   ├── core/
│   │   │   ├── model_manager.py  # 모델 로딩/전환 싱글턴
│   │   │   ├── model_registry.py # 내장 모델 정의
│   │   │   ├── analysis_engine.py # 모든 스캔 모달리티
│   │   │   ├── perturbation_engine.py # 제로/증폭/절제/패치
│   │   │   ├── emotion_engine.py # 감정 프로브 추출 + 조작
│   │   │   ├── sae_manager.py   # 프로바이더 기반 SAE 로딩
│   │   │   ├── sae_providers.py # SAELens + EleutherAI 어댑터
│   │   │   ├── sae_registry.py  # 모델 → SAE 매핑
│   │   │   ├── battery_engine.py # 기능 테스트 모음
│   │   │   └── scan_cache.py    # LRU 스캔 결과 캐시
│   │   ├── schemas/             # Pydantic 요청/응답 모델
│   │   └── data/
│   │       └── emotion_comprehension_texts.csv
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── api/client.ts        # 백엔드 HTTP 클라이언트
│       ├── store/               # Zustand 스토어
│       ├── components/Panels/   # 우측 사이드바 패널
│       ├── types/               # TypeScript 인터페이스
│       └── i18n/translations.ts
└── docs/
```

---

## API 레퍼런스

기본 URL: `http://localhost:8000/api`

### 모델 관리

#### 모델 로드

```bash
curl -X POST /api/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2"}'
```

응답: `ModelInfo` — 레이어 구성, 디바이스, dtype 정보.

#### 모델 목록

```bash
curl /api/model/list
```

응답: `is_loaded`, `tl_compat`, `gated` 플래그가 포함된 모델 배열.

#### HuggingFace Hub 검색

```bash
curl "/api/model/search?query=pythia&limit=5"
```

#### 모델 언로드

```bash
curl -X DELETE /api/model/unload
```

### 스캔 엔드포인트

모든 스캔 엔드포인트는 `POST`로 프롬프트를 받고 모드별 데이터를 반환합니다.

#### T1 — 구조

```bash
curl -X POST /api/scan/structural
```

본문 불필요. 레이어, 연결, 파라미터 수 반환.

#### T2 — 가중치

```bash
curl -X POST /api/scan/weights \
  -H "Content-Type: application/json" \
  -d '{"layer_ids": ["blocks.0.attn", "blocks.11.mlp"]}'
```

선택적 `layer_ids` 필터. 가중치 통계 + 히스토그램 반환.

#### fMRI — 활성화

```bash
curl -X POST /api/scan/activation \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

레이어별 토큰별 활성화 반환 (L2 노름, 0-1 정규화).

#### DTI — 회로

```bash
curl -X POST /api/scan/circuits \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "target_token_idx": -1}'
```

컴포넌트 중요도 점수 + 어텐션 패턴 반환.

#### FLAIR — 이상

```bash
curl -X POST /api/scan/anomaly \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The first president of Mars was"}'
```

KL 발산, 엔트로피, 이상 점수, 로짓 렌즈 예측 반환.

### 섭동 엔드포인트

#### Zero-Out (제로화)

```bash
curl -X POST /api/perturb/zero \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1
  }'
```

원본 vs 섭동 예측, 로짓 차이, KL 발산 반환.

#### Amplify (증폭)

```bash
curl -X POST /api/perturb/amplify \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1,
    "factor": 2.0
  }'
```

#### Ablate (평균 절제)

```bash
curl -X POST /api/perturb/ablate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1
  }'
```

#### Activation Patch (활성화 패치)

```bash
curl -X POST /api/perturb/patch \
  -H "Content-Type: application/json" \
  -d '{
    "clean_prompt": "The capital of France is",
    "corrupt_prompt": "The capital of Xxxxx is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1
  }'
```

복구 점수 (0-1) 반환.

#### Causal Trace (전체 인과 추적)

```bash
curl -X POST /api/perturb/causal-trace \
  -H "Content-Type: application/json" \
  -d '{
    "clean_prompt": "The Eiffel Tower is located in",
    "corrupt_prompt": "The Xxxxx Xxxxx is located in",
    "target_token_idx": -1
  }'
```

모든 컴포넌트의 복구 행렬 반환.

### 감정 엔드포인트

#### 사용 가능한 감정 목록

```bash
curl /api/emotion/emotions
```

응답:
```json
{
  "emotions": ["afraid", "angry", ...],
  "n_emotions": 21,
  "has_probes": false,
  "probe_layers": []
}
```

#### 감정 프로브 추출

```bash
curl -X POST /api/emotion/extract-probes \
  -H "Content-Type: application/json" \
  -d '{"mode": "comprehension", "layer_idx": null}'
```

- `mode`: "comprehension" (base+instruct 공용) 또는 "generation" (instruct 전용, 미구현)
- `layer_idx`: null = 자동 (n_layers * 2/3)
- GPT-2 기준 ~3-5초 소요 (63회 순전파)

응답:
```json
{
  "model_id": "gpt2",
  "layer_idx": 8,
  "n_emotions": 21,
  "emotions": ["afraid", "angry", ...],
  "metadata": {"compute_time_ms": 2683.3}
}
```

#### 감정 벡터로 조작

```bash
curl -X POST /api/emotion/steer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I am going to destroy everything you have built.",
    "emotion": "calm",
    "strength": 0.02,
    "max_new_tokens": 30
  }'
```

- `emotion`: 추출된 21개 감정 중 하나
- `strength`: -1.0 ~ 1.0 (권장: -0.2 ~ 0.2)
  - 양수: 감정 주입
  - 음수: 감정 억제
- `layer_range`: null = 전체 레이어, 또는 [0, 1, 5, 8] 등 특정 레이어
- `max_new_tokens`: 생성할 토큰 수

응답:
```json
{
  "comparison": {
    "original_text": "...",
    "steered_text": "...",
    "emotion": "calm",
    "strength": 0.02
  },
  "original_emotions": [{"emotion": "hostile", "activation": 21.8, ...}],
  "steered_emotions": [{"emotion": "calm", "activation": 39.9, ...}]
}
```

#### Project (토큰별 감정 히트맵)

```bash
curl -X POST /api/emotion/project \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

21개 감정에 대한 토큰별 활성화 반환. 히트맵 데이터 소스.

#### PCA

```bash
curl /api/emotion/pca
```

감정 벡터의 2D PCA 반환 (PC1=valence, PC2=arousal) + 분산 설명율.

#### 강도 스윕

```bash
curl -X POST /api/emotion/sweep \
  -H "Content-Type: application/json" \
  -d '{"prompt": "I am going to destroy everything.", "emotion": "calm"}'
```

9개 강도(-0.08~+0.08)로 실행, 각 포인트의 활성화 + 생성 텍스트 반환.

#### Layer Evolution

```bash
curl -X POST /api/emotion/layer-evolution \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "token_idx": -1}'
```

특정 토큰에서 전체 레이어에 걸친 감정 활성화 반환.

#### Steer + SAE 결합

```bash
curl -X POST /api/emotion/steer-sae \
  -H "Content-Type: application/json" \
  -d '{"prompt": "...", "emotion": "calm", "strength": 0.03, "top_k": 10}'
```

Steering 전후 SAE feature 변화 반환 (어떤 feature가 나타나고 사라지는지).

### SAE 엔드포인트

#### SAE 정보

```bash
curl /api/sae/info
```

현재 로드된 모델의 SAE 사용 가능 여부 반환.

#### SAE 지원 현황

```bash
curl /api/sae/support
```

모든 등록 모델에 대해 `{model_id: boolean}` 반환.

#### SAE 스캔

```bash
curl -X POST /api/sae/scan \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The cat sat on the mat",
    "layer_idx": 8,
    "top_k": 20
  }'
```

토큰별 상위 k개 특징, 히트맵 데이터, 재구성 손실, 희소성 반환.

### 배터리 테스트

```bash
curl -X POST /api/battery/run \
  -H "Content-Type: application/json" \
  -d '{"include_sae": true}'
```

### 진단 보고서

```bash
curl -X POST /api/report/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "modes": ["T1", "fMRI", "FLAIR"]}'
```

---

## 백엔드 핵심 모듈

### ModelManager (`model_manager.py`)

싱글턴 패턴 — 한 번에 하나의 모델만 관리.

```python
model_manager = ModelManager()
model_manager.load_model("gpt2", device="auto")
model = model_manager.get_model()  # HookedTransformer
model_manager.unload_model()
```

- 디바이스 자동 감지: CUDA > MPS > CPU
- 1B 이상 모델은 float16
- 언로드 시 GPU 메모리 정리

### AnalysisEngine (`analysis_engine.py`)

5+1가지 스캔 모달리티 수행.

```python
engine = AnalysisEngine(model_manager)
data = engine.scan_activation(ActivationScanRequest(prompt="Hello"))
data = engine.scan_sae(SAEScanRequest(prompt="Hello", layer_idx=8), sae_manager)
```

### EmotionEngine (`emotion_engine.py`)

감정 프로브 추출 및 생성 조작.

```python
engine = EmotionEngine(model_manager)
probes = engine.extract_probes(ExtractProbesRequest(mode="comprehension"))
result = engine.steer(SteerRequest(prompt="...", emotion="calm", strength=0.02))
```

**프로브 계산 과정:**
1. `emotion_comprehension_texts.csv`의 각 문장을 모델에 순전파
2. 마지막 토큰의 residual stream 활성화를 추출 (레이어 ≈ 2n/3)
3. 감정별 3개 문장의 평균
4. 전체 감정의 전역 평균을 빼기 → 감정 벡터

**조작(steering) 작동 방식:**
1. 감정 벡터를 단위 길이로 정규화
2. 각 순전파에서: `방향 * 강도 * residual_norm`을 `hook_resid_post`에 더하기
3. `run_with_hooks()`를 사용한 수동 탐욕 토큰 루프 (TransformerLens의 `generate()`는 훅을 지원하지 않음)

### SAE 프로바이더 시스템 (`sae_providers.py`)

여러 SAE 백엔드를 위한 어댑터 패턴.

```python
# 통합 인터페이스
provider = load_sae_provider("saelens", model_id, layer_idx, device, registry_entry)
enc = provider.encode(activations)  # → EncodeResult(top_acts, top_indices, full_acts)
recon = provider.decode_from_top(enc.top_acts, enc.top_indices)
print(provider.d_sae, provider.hook_name)
```

**프로바이더:**
- `SAELensProvider`: `sae_lens.SAE` 래핑 — 밀집(dense) 인코드/디코드
- `EleutherAIProvider`: `sparsify.Sae` 래핑 — 네이티브 희소(top-k) 인코딩

### PerturbationEngine (`perturbation_engine.py`)

TransformerLens 훅을 통한 상태 비저장(stateless) 섭동.

```python
engine = PerturbationEngine(model_manager)
result = engine.zero_out(ZeroOutRequest(prompt="...", component="blocks.9.mlp"))
trace = engine.causal_trace(CausalTraceRequest(clean_prompt="...", corrupt_prompt="..."))
```

---

## 프론트엔드 아키텍처

### 스토어 패턴 (Zustand)

```typescript
// store/useEmotionStore.ts
export const useEmotionStore = create<EmotionState>((set, get) => ({
  steerResult: null,
  isSteering: false,
  steer: async (prompt) => {
    set({ isSteering: true });
    const result = await api.emotion.steer(prompt, get().selectedEmotion, get().strength);
    set({ steerResult: result, isSteering: false });
    useScanStore.getState().addLog('Steer 완료');
  },
}));
```

### API 클라이언트 (`api/client.ts`)

```typescript
export const api = {
  emotion: {
    extractProbes: (mode, layerIdx?) => request<ExtractProbesResponse>('/emotion/extract-probes', {
      method: 'POST',
      body: JSON.stringify({ mode, layer_idx: layerIdx ?? null }),
    }),
    steer: (prompt, emotion, strength, maxNewTokens) => request<SteerResponse>('/emotion/steer', {
      method: 'POST',
      body: JSON.stringify({ prompt, emotion, strength, max_new_tokens: maxNewTokens }),
    }),
  },
  // ...
};
```

### 패널 컴포넌트 패턴

```tsx
export function EmotionPanel() {
  const t = useLocaleStore((s) => s.t);
  const { probeResult, extractProbes, steer } = useEmotionStore();
  const isLoaded = useModelStore((s) => s.modelInfo !== null);

  if (!isLoaded) return <div>먼저 모델을 로드하세요</div>;

  return (
    <div className="px-3 py-2">
      {/* 헤더 + 컨트롤 + 시각화 */}
    </div>
  );
}
```

---

## Neural MRI 확장하기

### 새 스캔 모드 추가

1. **스키마**: `schemas/scan.py`에 요청/응답 모델 추가
2. **엔진**: `analysis_engine.py`에 스캔 메서드 추가
3. **라우트**: `routes_newmode.py`에 FastAPI 라우터 생성
4. **등록**: `main.py`에 라우터 추가
5. **프론트엔드 타입**: `types/scan.ts`에 추가
6. **API 클라이언트**: `api/client.ts`에 추가
7. **스토어**: `useNewModeStore.ts` 생성
8. **패널**: `NewModePanel.tsx` 생성
9. **i18n**: 번역 키 추가

### 새 SAE 프로바이더 추가

1. `sae_providers.py`에 `SAEProvider`를 구현하는 새 클래스 생성
2. `load_sae_provider()` 팩토리에 프로바이더 타입 추가
3. `sae_registry.py`에 `"provider": "your_provider"` 레지스트리 항목 추가

### 새 감정 추가

1. `data/emotion_comprehension_texts.csv`에 문장 추가:
   ```
   your_emotion,1,"감정을 묘사하는 첫 번째 문장..."
   your_emotion,2,"두 번째 문장..."
   your_emotion,3,"세 번째 문장..."
   ```
2. 서버 재시작 — 새 감정이 자동으로 사용 가능

---

## 환경 변수 & 설정

`NMRI_` 접두사로 환경 변수를 통해 설정합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `NMRI_DEFAULT_MODEL` | `gpt2` | 시작 시 로드할 모델 |
| `NMRI_DEVICE` | `auto` | 디바이스: auto, cuda, mps, cpu |
| `NMRI_HF_TOKEN` | — | 게이트 모델용 HuggingFace 토큰 |
| `NMRI_MAX_CACHE_ENTRIES` | `100` | 스캔 결과 캐시 크기 |
| `NMRI_CORS_ORIGINS` | `["*"]` | CORS 허용 출처 |

`backend/.env`에 설정:
```
NMRI_DEFAULT_MODEL=gpt2
NMRI_HF_TOKEN=hf_your_token_here
```

---

## 테스트

### 백엔드

```bash
cd backend
uv run pytest tests/ -v          # 전체 테스트
uv run pytest tests/ -x -q       # 첫 실패 시 중단
uv run ruff check .              # 린트
uv run ruff format --check .     # 포맷 검사
```

### 프론트엔드

```bash
cd frontend
pnpm tsc --noEmit    # 타입 검사
pnpm build           # 프로덕션 빌드
```

### 유용한 테스트 명령어

```bash
# 빠른 API 연기 테스트
curl http://localhost:8000/api/model/info
curl -X POST http://localhost:8000/api/scan/structural
curl -X POST http://localhost:8000/api/emotion/extract-probes \
  -H "Content-Type: application/json" -d '{"mode":"comprehension"}'
```
