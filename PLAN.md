# VectorTrainer 프로젝트 계획서

> 제1회 OKKY 바이브코딩 해커톤 | 팀 cozymori

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **프로젝트명** | VectorTrainer |
| **한 줄 소개** | Self-Evolving AI Loop를 위한 핵심 엔진: 밀도 기반 Golden Data 선별 + 자동 프롬프트 합성 |
| **팀명** | cozymori |
| **팀원** | 김준영 (백엔드) · 조은설 (백엔드, 기획) |
| **GitHub** | https://github.com/Cozymori/VectorTrainer.git |

---

## 2. 문제 정의

LLM 기반 서비스를 운영하면 **실행 로그가 계속 쌓이지만**, 이 데이터에서 어떤 게 "좋은 데이터"인지 사람이 직접 판단하고, 프롬프트를 수동으로 개선하고, 파인튜닝도 수작업으로 돌려야 한다.

**핵심 질문:** 이 과정을 사람 없이 자동으로 돌릴 수 있을까?

---

## 3. 해결 방안

VectorTrainer는 3개의 엔진을 파이프라인으로 연결하여 **Self-Evolving AI Loop**를 완성한다.

```
[VectorWave 로그 DB]
        │
        ▼
┌─────────────────────┐
│  1. Extractor       │  ← 밀도 기반으로 Golden Data 자동 선별
│     (데이터 선별)     │
└────────┬────────────┘
         │ JSONL (ChatML)
         ▼
┌─────────────────────┐
│  2. Synthesizer     │  ← 사용자 피드백 분석 → 규칙 추출 → Hook Script 생성
│     (프롬프트 합성)   │
└────────┬────────────┘
         │ generated_prompt_hook.py
         ▼
┌─────────────────────┐
│  3. Pipeline        │  ← 선별된 데이터로 파인튜닝 실행 (OpenAI 등)
│     (파인튜닝)       │
└─────────────────────┘
```

---

## 4. 외부 의존성: VectorWave

VectorTrainer는 **VectorWave** (`vectorwave==0.2.9`)를 외부 의존성으로 사용한다.
VectorWave는 같은 팀이 개발한 별도 라이브러리로, 함수 실행 컨텍스트를 벡터 DB(Weaviate)에 자동 저장/관리한다.

**VectorTrainer가 사용하는 VectorWave 인터페이스:**

| 클래스 | 메서드 | 용도 |
|---|---|---|
| `VectorWaveDatasetManager` | `exec_col.query.fetch_objects()` | 실행 로그 전체 조회 |
| | `exec_col.query.near_vector()` | ANN 기반 인접 노드 탐색 |
| | `golden_col` | 골든 데이터 컬렉션 접근 |
| | `recommend_candidates()` | 밀도 기반 후보 추천 |
| | `register_as_golden()` | 골든 데이터 등록 |

**연동 전략 (ADR-001):** Protocol 기반 DI로 연동. VectorTrainer의 import 그래프에 vectorwave 패키지가 포함되지 않도록 하여 느슨한 결합을 유지한다.

---

## 5. 모듈별 상세 설계

### 5.1 types.py — 공유 타입

| 타입 | 종류 | 설명 |
|---|---|---|
| `DatasetManagerProtocol` | Protocol | VectorWaveDatasetManager 인터페이스 정의 |
| `ExecutionLog` | dataclass | 벡터 공간의 노드 (uuid, function_name, return_value, status, vector) |
| `GoldenCandidate` | dataclass | 선별된 골든 데이터 후보 |
| `SelectionStrategy` | Enum | STEADY / ANOMALY |
| `FeedbackPair` | dataclass | L_bad → L_fixed 피드백 쌍 |
| `Rule` | dataclass | 합성된 규칙 단위 |

### 5.2 extractor.py — 이산수학적 데이터 선별 엔진

**수학적 모델:**
- 각 실행 로그 L을 **가중치 그래프(G)의 노드**로 정의
- 노드 간 **코사인 유사도**를 에지 가중치(W)로 설정
- 임계값 ε 이내의 인접 노드 집합: N(L_i) = {L_j | sim(L_i, L_j) ≥ 1 - ε}
- **밀도 함수**: D(L_i) = Σ_{j ∈ N(L_i)} W_ij

**선별 전략:**

| 전략 | 조건 | 목적 |
|---|---|---|
| **Steady State** | 밀도 상위 N개 노드의 Centroid | 모델의 '기본기' 강화 |
| **Anomaly Detection** | 밀도 하위 + SUCCESS 상태 | 새로운 패턴 학습 |

**출력:** ChatML 포맷 JSONL (`{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`)

**핵심 메서드:**

```
DensityBasedExtractor
├── __init__(dataset_manager, epsilon=0.3, top_k=50)
├── _build_weighted_graph(logs) → 가중치 그래프 구성
├── _compute_density(node_id, graph) → 노드 밀도 계산
├── select_steady_state(function_name) → 고밀도 데이터 선별
├── select_anomalies(function_name) → 저밀도+성공 데이터 선별
└── extract_golden_data(function_name) → 최종 JSONL 출력
```

### 5.3 synthesizer.py — 자동 프롬프트 생성 파이프라인

**3단계 프로세스:**

```
[FeedbackPair]  →  [FeedbackDiffAnalyzer]  →  [RuleSetSynthesizer]  →  [HookScriptGenerator]
(L_bad→L_fixed)    (Edit Distance + 의미적 차이)    (규칙 세트 변환/병합)     (물리적 .py 파일 생성)
```

| 클래스 | 역할 |
|---|---|
| `FeedbackDiffAnalyzer` | 문자열 편집 거리(Levenshtein) + 벡터 거리로 변경점 추출 |
| `RuleSetSynthesizer` | 차이점 → 규칙 세트(R) 변환, 모순 규칙 감지 및 병합 (LLM 활용) |
| `HookScriptGenerator` | 규칙 → `generated_prompt_hook.py` 파일 생성 + AST 구문 검증 |

**Hook Script 구조 (생성될 파일):**
```python
# generated_prompt_hook.py — 자동 생성됨, 수동 편집 금지
def intercept_prompt(user_input: str, context: dict) -> str:
    """사용자 입력과 모델 출력 사이에 개입하는 인터셉터."""
    # Rule 1: ...
    # Rule 2: ...
    return modified_prompt

def intercept_output(model_output: str, context: dict) -> str:
    """모델 출력을 후처리하는 인터셉터."""
    return modified_output
```

### 5.4 pipeline.py — 파인튜닝 파이프라인

**Strategy Pattern (ADR-004):**

```
BaseTrainer (ABC)
├── prepare_data(jsonl_path) → file_id
├── start_training(file_id) → job_id
├── get_status(job_id) → Dict
└── get_result(job_id) → Dict
        │
        ├── OpenAITrainer          ← 현재 구현
        └── (LocalLLMTrainer)      ← 추후 확장
```

**오케스트레이터: TrainingPipeline**
```
TrainingPipeline
├── __init__(extractor, synthesizer, trainer)
└── run(function_name, feedback_pairs, output_dir)
    ├── 1. extractor.extract_golden_data() → JSONL
    ├── 2. synthesizer로 Hook Script 생성
    └── 3. trainer로 파인튜닝 실행
```

### 5.5 dashboard.py — 파이프라인 모니터링 대시보드

파인튜닝 과정을 실시간으로 시각화하는 **이중 인터페이스**를 제공한다.

**CLI 대시보드 (rich 기반):**
- 파이프라인 단계별 진행률 바 (Extractor → Synthesizer → Trainer)
- 실시간 로그 스트리밍 (선별된 데이터 수, 생성된 규칙 수, 학습 상태)
- 최종 결과 요약 테이블 (Golden Data 통계, 학습 메트릭)

```
┌─ VectorTrainer Pipeline ──────────────────────────────┐
│                                                        │
│  [1/3] Extracting Golden Data    ████████████░░  80%  │
│        Steady: 32  |  Anomaly: 8  |  Total: 40        │
│                                                        │
│  [2/3] Synthesizing Rules        ░░░░░░░░░░░░░░  --   │
│  [3/3] Fine-tuning               ░░░░░░░░░░░░░░  --   │
│                                                        │
│  ⏱  Elapsed: 00:02:15                                 │
└────────────────────────────────────────────────────────┘
```

**웹 대시보드 (Streamlit 기반):**
- 파이프라인 상태 실시간 모니터링
- Golden Data 벡터 분포 2D 시각화 (Steady vs Anomaly)
- 생성된 규칙 세트 미리보기
- 파인튜닝 학습 곡선 (loss, accuracy)

**핵심 메서드:**
```
PipelineMonitor
├── __init__(pipeline)
├── on_stage_start(stage_name)    ← 콜백: 단계 시작
├── on_progress(stage, current, total)  ← 콜백: 진행률 갱신
├── on_stage_complete(stage, result)    ← 콜백: 단계 완료
└── summary() → Dict              ← 최종 결과 요약

CLIDashboard(PipelineMonitor)     ← rich 기반 터미널 출력
WebDashboard(PipelineMonitor)     ← Streamlit 웹 앱
```

---

## 6. 테스트 전략

| 대상 | 테스트 항목 | Mocking |
|---|---|---|
| Extractor | 코사인 유사도, 밀도 함수, Steady/Anomaly 선별, JSONL 출력 | VectorWaveDatasetManager Mock |
| Synthesizer | 편집 거리, 규칙 합성, 모순 병합, Hook Script 생성 + AST 검증 | OpenAI API Mock |
| Pipeline | ABC 인스턴스화 불가, 파일 업로드, 학습 시작, 통합 테스트 | OpenAI API + DatasetManager Mock |

**도구:** pytest + pytest-mock + unittest.mock.patch

---

## 7. 기술 스택

| 구분 | 기술 |
|---|---|
| 언어 | Python ≥ 3.10 |
| 빌드 | setuptools (pyproject.toml) |
| 외부 의존성 | VectorWave (vectorwave), OpenAI API (openai), NumPy |
| 벡터 DB | Weaviate (VectorWave를 통해 간접 사용) |
| CLI 대시보드 | rich (진행률 바, 테이블, 실시간 로그) |
| 웹 대시보드 | Streamlit (파이프라인 모니터링, 벡터 시각화) |
| 테스트 | pytest, pytest-mock |
| 버전관리 | Git, GitHub |

---

## 8. 구현 순서 및 커밋 전략

공정성 가이드에 따라 **1커밋 = 1의도 = 1변화** 원칙을 준수한다.

| 순서 | 작업 | 커밋 메시지 | 상태 |
|---|---|---|---|
| 1 | 프로젝트 스캐폴딩 + Git 초기화 | `chore: initialize project scaffold` | ✅ 완료 |
| 2 | SPEC.md + ADR.md 작성 | `docs: add SPEC.md and ADR.md` | ✅ 완료 |
| 3 | 공유 타입 정의 (types.py) | `feat(types): define shared protocols and data classes` | ⏳ 대기 |
| 4 | 데이터 선별 엔진 (extractor.py) | `feat(extractor): implement density-based selection` | ⏳ 대기 |
| 5 | 프롬프트 합성기 (synthesizer.py) | `feat(synthesizer): implement criteria synthesis` | ⏳ 대기 |
| 6 | 파인튜닝 파이프라인 (pipeline.py) | `feat(pipeline): implement BaseTrainer and OpenAITrainer` | ⏳ 대기 |
| 7 | 모니터링 대시보드 (dashboard.py) | `feat(dashboard): add CLI and web pipeline monitoring` | ⏳ 대기 |
| 8 | 단위 테스트 | `test: add comprehensive unit tests` | ⏳ 대기 |
| 9 | README.md 완성 | `docs(readme): add build, run, test instructions` | ⏳ 대기 |

---

## 9. 디렉토리 구조 (최종)

```
vectortrainer/
├── pyproject.toml
├── README.md
├── SPEC.md              ← 기능 요구사항
├── ADR.md               ← 아키텍처 결정 기록
├── PLAN.md              ← 이 문서
├── .gitignore
├── src/
│   └── vector_trainer/
│       ├── __init__.py
│       ├── types.py      ← Protocol, dataclass 등 공유 타입
│       ├── extractor.py  ← 밀도 기반 Golden Data 선별 엔진
│       ├── synthesizer.py ← 피드백 분석 → 규칙 합성 → Hook Script 생성
│       ├── pipeline.py   ← BaseTrainer ABC + OpenAITrainer + 오케스트레이터
│       └── dashboard.py  ← CLI(rich) + 웹(Streamlit) 파이프라인 모니터링
└── tests/
    ├── __init__.py
    └── test_vector_trainer.py
```

---

## 10. AI 도구 사용 기록

- **사용 도구:** Claude Code (Claude Opus 4.6)
- **위임 범위:** 코드 생성, 문서 작성, 아키텍처 설계
- **검증:** 사용자(팀원)가 모든 결과물을 리뷰 후 승인
- **기록:** 모든 커밋 메시지에 AI 사용 여부 명시, Co-Authored-By 태그 포함
