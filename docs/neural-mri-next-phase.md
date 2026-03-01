# Neural MRI — Phase 5 방향 메모

**From:** JJ  
**To:** Cody  
**Date:** 2026-03-01  
**Re:** Phase 4 완료 리뷰 + 향후 우선순위 조정 + 신규 기능 제안

---

## 1. Phase 4 완료에 대해

Phase 0–4 전체를 깔끔하게 마무리해줘서 고맙다. 특히 SAE Feature 탐색기, 실시간 협업, 녹화/재생, 4가지 레이아웃까지 — 원래 스펙에서 "향후"로 잡았던 것들이 전부 들어간 건 기대 이상이었다. CI 파이프라인 + pytest 111개도 오픈소스 공개를 고려하면 큰 자산이다.

---

## 2. 향후 제안에 대한 우선순위 재조정

네가 제안한 Tier 분류는 **interpretability 도구**로서의 완성도 기준으로는 정확하다. 하지만 Neural MRI는 "또 하나의 interpretability 도구"가 아니라 **Model Medicine의 진단 장비**라는 더 큰 프레임 안에 있다. 이 관점에서 우선순위를 조정하고 싶다.

### 조정 1: Cross-model 비교 → Tier 1 최우선으로 승격

의학에서 진단의 핵심은 "정상 vs 비정상" 비교다. 같은 프롬프트에 대해 GPT-2 vs Pythia vs Gemma의 반응을 나란히 보는 것은 Model Medicine에서 **비교 해부학(Comparative Anatomy)**이자 **감별 진단(Differential Diagnosis)**의 기초 도구에 해당한다.

Four Shell Model 관점에서 보면, Core(모델 가중치)가 다른 모델들이 동일한 Shell(프롬프트)에 어떻게 다르게 반응하는지를 시각적으로 보여줄 수 있으면, 그 자체가 논문 figure로 쓸 수 있는 수준의 결과물이 된다. CompareView 인프라가 이미 Multi-prompt에서 만들어져 있으니 확장 난이도도 Tier 2보다 낮을 수 있다.

### 조정 2: Causal Tracing 시각화 → Tier 2로 승격

PerturbationEngine에 activation_patch가 이미 구현되어 있다. 부족한 건 **레이어 × 토큰 히트맵** 형태의 프론트엔드 시각화뿐이다. 이걸 구현하면:

- ROME/MEMIT 논문의 핵심 figure를 원클릭으로 생성 가능
- Model Medicine에서 "이 모델의 어느 레이어가 이 사실을 저장하고 있는가"라는 **진단 검사의 gold standard**가 됨
- 의학 비유로는 CT scan 또는 조영 MRI에 해당

백엔드는 거의 완성 상태이므로 D3 히트맵 시각화 + 컴포넌트별 recovery score 표시가 핵심 작업이 될 것이다.

### 조정된 우선순위 요약

| 순위 | 기능 | 이유 |
|------|------|------|
| **1** | Cross-model 비교 | 감별 진단의 기초. CompareView 확장으로 구현 가능 |
| **2** | Causal Tracing 시각화 | 백엔드 완성 상태. 프론트엔드 히트맵만 추가 |
| **3** | Attention Head Heatmap | DTI 데이터 재활용. 빠르게 구현 가능 |
| **4** | Logit Lens 대시보드 | FLAIR 내부 로직 재활용. 모델 내부 의사결정 과정 시각화 |
| **5** | 프롬프트 템플릿 라이브러리 | 사용자 온보딩. IOI/Greater-Than 등 프리셋 |
| **6** | 키보드 단축키 확장 | 파워유저 생산성. 낮은 난이도 |

Tier 3의 nnsight 폴백과 플러그인 시스템은 현재 8개 모델 지원만으로도 충분하므로 후순위 유지.

---

## 3. 신규 기능 제안: HuggingFace Hub 원격 모델 연결

현재 Neural MRI는 로컬에 모델을 다운로드해서 TransformerLens로 로드하는 방식이다. 이걸 확장해서 **HuggingFace Hub에 올라온 모델을 웹 기반으로 바로 연결하고 스캔할 수 있는 기능**을 검토해 보자.

### 구상

- 사용자가 HuggingFace 모델 ID를 입력하면 → 모델 메타데이터 조회 → TransformerLens 호환성 체크 → 로드 가능하면 원클릭 스캔
- Model Registry를 정적 목록에서 **동적 검색**으로 확장
- HuggingFace Hub API (`huggingface_hub` 라이브러리)로 모델 검색/필터링 UI 제공

### 해결해야 할 문제: 접근 토큰

현재 gated model(Gemma, Llama 등)은 `.env`에 `NMRI_HF_TOKEN`을 수동 설정하고 있다. 이 부분의 자동화가 핵심 과제인데, 몇 가지 접근법을 검토해 달라:

**접근법 A: 사용자 토큰 입력 UI**
- 프론트엔드 Settings 패널에 HF Token 입력 필드 추가
- 입력된 토큰을 세션 동안만 백엔드에 전달 (영구 저장하지 않음)
- 장점: 가장 단순하고 안전
- 단점: 사용자가 매번 토큰을 입력해야 함

**접근법 B: HuggingFace OAuth 로그인**
- HF Hub의 OAuth 플로우를 통한 인증
- `huggingface_hub` 라이브러리의 `login()` 또는 OAuth 리다이렉트 활용
- 장점: 사용자 경험이 좋음
- 단점: 구현 복잡도 높음, 서버사이드 토큰 관리 필요

**접근법 C: 로컬 HF CLI 토큰 자동 감지**
- `~/.cache/huggingface/token`에 이미 저장된 토큰을 자동으로 읽어오기
- `huggingface_hub`의 `HfApi(token=True)` 패턴 활용
- 장점: 로컬 사용자에게는 제로 설정
- 단점: Docker/원격 배포 시 적용 불가

**접근법 D: 비-gated 모델만 동적 지원**
- 동적 검색 대상을 gated가 아닌 모델로 한정
- gated 모델은 기존처럼 수동 토큰 설정 유지
- 장점: 토큰 문제를 완전히 우회
- 단점: Gemma, Llama 등 주요 모델이 제외됨

### 내 생각

현실적으로는 **C + A 하이브리드**가 가장 합리적일 것 같다. 로컬에 `huggingface-cli login`이 되어 있으면 자동 감지, 안 되어 있으면 UI에서 토큰 입력. 하지만 이게 구현 대비 가치가 있는지, 혹시 다른 방법이 있는지 네 의견을 듣고 싶다.

또한 HuggingFace 외에 다른 모델 허브(Ollama 로컬, GGUF 포맷 등)와의 연동 가능성도 같이 검토해 주면 좋겠다. TransformerLens가 지원하는 범위 내에서 어디까지 확장 가능한지 파악이 필요하다.

### 추가 고려: TransformerLens 호환성 자동 검증

동적으로 모델을 로드할 경우 호환성 문제가 생길 수 있다. 다음 로직이 필요할 것이다:

1. HF 모델 메타데이터에서 architecture 타입 확인
2. TransformerLens의 알려진 지원 아키텍처 목록과 매칭
3. 매칭되면 로드 시도, 실패 시 사용자에게 명확한 피드백
4. 장기적으로 nnsight 폴백까지 연결

---

## 4. 정리

| 구분 | 내용 |
|------|------|
| **즉시 시작** | Cross-model 비교 (최우선) + Causal Tracing 시각화 |
| **이어서** | Attention Heatmap + Logit Lens 대시보드 + 프롬프트 템플릿 |
| **조사/설계** | HuggingFace Hub 동적 모델 연결 (토큰 전략 포함) |
| **후순위** | nnsight 폴백, 플러그인 시스템, PDF 리포트 |

질문이나 다른 의견 있으면 알려줘.
