# VectorTrainer 사용 가이드

## 개요

VectorTrainer는 LLM 서비스의 운영(Serving)과 개선(Fine-tuning)을 자동화하는 **Self-Evolving AI Loop** 엔진입니다.

```
운영 중 LLM 호출 로그 수집 (VectorWave)
        ↓
밀도 기반 골든 데이터 선별 (Extractor)
        ↓
피드백 분석 → 규칙 합성 → 훅 스크립트 생성 (Synthesizer)
        ↓
골든 데이터로 파인튜닝 (Trainer)
        ↓
개선된 모델로 다시 서비스 → 반복
```

---

## 설치

```bash
# 패키지 설치
pip install -e ".[dev]"

# VectorWave 의존성 (Weaviate 실행 필요)
docker compose up -d
```

---

## 전체 파이프라인 실행

가장 일반적인 사용 방법입니다. 데이터 선별 → 규칙 합성 → 파인튜닝을 한 번에 수행합니다.

```python
from vectorwave import VectorWaveDatasetManager
from vector_trainer.extractor import DensityBasedExtractor
from vector_trainer.synthesizer import run_synthesis_pipeline
from vector_trainer.pipeline import OpenAITrainer, TrainingPipeline
from vector_trainer.dashboard import CLIDashboard
from vector_trainer.types import FeedbackPair

# ── 1. VectorWave 연결 ──────────────────────────────────
dataset_manager = VectorWaveDatasetManager()

# ── 2. 골든 데이터 선별기 생성 ──────────────────────────
extractor = DensityBasedExtractor(
    dataset_manager=dataset_manager,
    epsilon=0.3,   # 유사도 임계값 (낮을수록 엄격)
    top_k=50,      # 전략별 최대 선별 수
)

# ── 3. 피드백 쌍 정의 ───────────────────────────────────
# "모델이 이렇게 답했는데(bad), 이렇게 답했어야 했다(fixed)"
feedback_pairs = [
    FeedbackPair(
        input_prompt="이 제품의 리뷰를 요약해주세요",
        bad_output="좋은 제품입니다",
        fixed_output="이 제품은 내구성과 디자인이 우수하며, 배송도 빠릅니다.",
        context={"function": "generate_review_summary"},
    ),
    FeedbackPair(
        input_prompt="사용자 리뷰를 분석해주세요",
        bad_output="별로입니다",
        fixed_output=(
            "사용자 리뷰 분석 결과: 긍정 요소(디자인, 가격)와 "
            "개선 필요 요소(AS, 설명서)가 혼재합니다. 만족도: 3.8/5."
        ),
        context={"function": "generate_review_summary"},
    ),
]

# ── 4. 트레이너 & 대시보드 ──────────────────────────────
trainer = OpenAITrainer(api_key="sk-...")
dashboard = CLIDashboard()

# ── 5. 파이프라인 조립 & 실행 ───────────────────────────
pipeline = TrainingPipeline(
    extractor=extractor,
    synthesizer_output_dir="./output/hooks",
    trainer=trainer,
    monitor=dashboard,
)

result = pipeline.run(
    function_name="generate_review_summary",
    feedback_pairs=feedback_pairs,
    output_dir="./output",
)

# ── 6. 결과 확인 ────────────────────────────────────────
dashboard.summary()
print(f"파인튜닝 모델: {result['train']['fine_tuned_model']}")
print(f"골든 데이터: {result['golden_data_path']}")
print(f"훅 스크립트: {result['hook_script_path']}")
```

---

## 모듈별 개별 사용

각 모듈은 독립적으로도 사용할 수 있습니다.

### 1. Extractor — 골든 데이터 선별

실행 로그에서 벡터 그래프를 구축하고, 밀도 분석으로 학습 가치가 높은 데이터를 자동 선별합니다.

```python
from vector_trainer.extractor import DensityBasedExtractor

extractor = DensityBasedExtractor(
    dataset_manager=dataset_manager,
    epsilon=0.3,
    top_k=50,
)

# 정상 상태 (Steady State): 고밀도 클러스터의 중심점
# → 모델의 '기본기' 강화용 데이터
steady = extractor.select_steady_state("generate_review_summary")
for c in steady[:5]:
    print(f"  [{c.uuid}] 밀도={c.density:.4f} | {c.return_value[:60]}")

# 이상치 (Anomaly): 저밀도이지만 SUCCESS인 노드
# → 새로운 패턴 학습을 위한 경계 데이터
anomalies = extractor.select_anomalies("generate_review_summary")
for c in anomalies[:5]:
    print(f"  [{c.uuid}] 밀도={c.density:.4f} | {c.return_value[:60]}")

# ChatML JSONL로 내보내기 (OpenAI 파인튜닝 포맷)
extractor.extract_golden_data("generate_review_summary", "./golden_data.jsonl")
```

**출력 예시 (golden_data.jsonl):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a function executor for 'generate_review_summary'. ..."},
    {"role": "user", "content": "function_name: generate_review_summary\nstrategy: steady\ndensity: 23.05"},
    {"role": "assistant", "content": "이 제품은 품질이 우수하고 배송이 빠릅니다. 평점: 5/5"}
  ]
}
```

### 2. Synthesizer — 피드백 분석 & 훅 스크립트 생성

피드백 쌍(bad → fixed)을 분석하여 규칙을 추출하고, 런타임에 적용 가능한 Python 스크립트를 생성합니다.

```python
from vector_trainer.synthesizer import (
    FeedbackDiffAnalyzer,
    RuleSetSynthesizer,
    HookScriptGenerator,
    run_synthesis_pipeline,
)
from vector_trainer.types import FeedbackPair

# ── 방법 1: 한 줄로 실행 ────────────────────────────────
hook_path = run_synthesis_pipeline(
    pairs=feedback_pairs,
    output_dir="./hooks",
)
print(f"훅 스크립트 생성 완료: {hook_path}")

# ── 방법 2: 단계별 실행 ─────────────────────────────────

# 2-1. 피드백 차이 분석
analyzer = FeedbackDiffAnalyzer()
for pair in feedback_pairs:
    diff = analyzer.analyze(pair)
    print(f"  편집거리={diff['edit_distance']}, 유사도={diff['similarity_score']:.3f}")

# 2-2. 규칙 합성
diffs = [analyzer.analyze(p) for p in feedback_pairs]
synthesizer = RuleSetSynthesizer()
rules = synthesizer.synthesize(diffs)
for rule in rules:
    print(f"  [{rule.rule_id}] {rule.description[:60]}")

# 2-3. 훅 스크립트 생성
generator = HookScriptGenerator()
hook_path = generator.generate(rules, "./hooks")
```

**생성되는 훅 스크립트 사용법:**

```python
# 생성된 훅 스크립트를 임포트하여 즉시 적용
from generated_prompt_hook import intercept_prompt, intercept_output

# 사용자 입력 전처리
modified_prompt = intercept_prompt(user_input, context={})

# 모델 출력 후처리
corrected_output = intercept_output(model_output, context={})
```

### 3. Trainer — 파인튜닝 실행

OpenAI 파인튜닝 API를 래핑한 트레이너입니다. Strategy Pattern으로 다른 벤더로 교체할 수 있습니다.

```python
from vector_trainer.pipeline import OpenAITrainer

trainer = OpenAITrainer(api_key="sk-...")

# 데이터 업로드
file_id = trainer.prepare_data("./golden_data.jsonl")

# 파인튜닝 시작
job_id = trainer.start_training(file_id, model="gpt-4o-mini-2024-07-18")

# 상태 확인
status = trainer.get_status(job_id)
print(f"상태: {status['status']}")

# 결과 조회
result = trainer.get_result(job_id)
print(f"파인튜닝 모델: {result['fine_tuned_model']}")
```

---

## 대시보드

### CLI 대시보드

터미널에서 파이프라인 진행 과정을 실시간으로 확인합니다.

```bash
python demo.py
```

`TrainingPipeline`에 `CLIDashboard`를 연결하면 각 단계의 시작/완료/에러를 자동으로 표시합니다.

### 웹 대시보드 (Streamlit)

브라우저에서 파이프라인 결과를 시각적으로 탐색합니다.

```bash
streamlit run streamlit_app.py
```

http://localhost:8501 에서 확인 가능하며, 4개 탭을 제공합니다:

| 탭 | 내용 |
|----|------|
| **파이프라인** | 단계별 진행 상태, 소요 시간, 출력 산출물 |
| **골든 데이터** | 선별된 데이터 목록, 밀도 분포 차트, JSONL 미리보기 |
| **규칙** | 합성된 규칙 테이블, 상세 조건/동작, 피드백 차이 분석 |
| **훅 스크립트** | 생성된 Python 스크립트 전문, 동작 방식 설명 |

---

## 핵심 파라미터

### DensityBasedExtractor

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `epsilon` | 0.3 | 코사인 유사도 임계값. 낮을수록 에지가 적어지고 그래프가 희소해짐 |
| `top_k` | 50 | 전략별 최대 선별 후보 수 |

### FeedbackPair

| 필드 | 설명 |
|------|------|
| `input_prompt` | 원래 사용자가 입력한 프롬프트 |
| `bad_output` | 모델이 실제로 생성한 (품질이 낮은) 출력 |
| `fixed_output` | 사람이 수정한 (바람직한) 출력 |
| `context` | 추가 메타데이터 (함수명, 태그 등) |

---

## 데모 실행

```bash
# Mock 모드 (VectorWave/OpenAI 없이 즉시 실행)
python demo.py

# 실제 VectorWave 연동 (Weaviate + OpenAI 키 필요)
export OPENAI_API_KEY='sk-...'
docker compose -f /path/to/vectorwave/docker-compose.yml up -d
python demo.py
```

---

## 테스트

```bash
# 전체 테스트 (42개)
pytest tests/ -v

# 모듈별 테스트
pytest tests/ -v -k "extractor"
pytest tests/ -v -k "synthesizer"
pytest tests/ -v -k "pipeline"
pytest tests/ -v -k "dashboard"
```

외부 의존성(VectorWave, OpenAI API)은 모두 Mocking 처리되어 있어, API 키 없이도 테스트가 가능합니다.
