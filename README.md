# VectorTrainer

![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Self-Evolving AI Loop를 위한 핵심 엔진**

---

## 프로젝트 개요

### 문제 정의

LLM 기반 서비스를 운영하면 실행 로그가 계속 쌓이지만, 이 데이터에서 Golden Data를 선별하고, 프롬프트를 개선하고, 파인튜닝을 실행하는 과정이 모두 수작업으로 이루어진다. 이로 인해 모델 개선 주기가 길어지고, 운영 비용이 증가한다.

### 해결

VectorTrainer는 3개의 엔진(**Extractor**, **Synthesizer**, **Pipeline**)을 파이프라인으로 연결하여 **Self-Evolving AI Loop**를 구축한다. 사람의 개입 없이 데이터 선별부터 프롬프트 합성, 파인튜닝까지 자동으로 수행한다.

[VectorWave](https://github.com/Cozymori/VectorTrainer.git) 라이브러리를 외부 의존성으로 활용하며, 함수 실행 컨텍스트를 벡터 DB(Weaviate)에 자동 저장/관리하는 기능을 제공받는다.

---

## 아키텍처

```
[VectorWave 로그 DB]
        |
        v
+-----------------------+
|  1. Extractor         |  -- 밀도 기반으로 Golden Data 자동 선별
|     (데이터 선별)      |
+-----------+-----------+
            | JSONL (ChatML)
            v
+-----------------------+
|  2. Synthesizer       |  -- 피드백 분석 -> 규칙 추출 -> Hook Script 생성
|     (프롬프트 합성)    |
+-----------+-----------+
            | generated_prompt_hook.py
            v
+-----------------------+
|  3. Pipeline          |  -- 선별된 데이터로 파인튜닝 실행 (OpenAI 등)
|     (파인튜닝)         |
+-----------+-----------+
            |
            v
+-----------------------+
|  4. Dashboard         |  -- 파이프라인 실행 과정 실시간 모니터링
|     (CLI + Web)       |
+-----------------------+
```

---

## 핵심 기능

### 밀도 기반 Golden Data 선별

각 실행 로그를 가중치 그래프의 노드로 정의하고, 코사인 유사도를 에지 가중치로 설정한다. 밀도 함수 `D(L_i) = sum of W_ij`를 기반으로 **Steady State**(고밀도 중심점)와 **Anomaly Detection**(저밀도 + SUCCESS) 전략을 적용하여 학습 효율이 높은 데이터를 자동 선별한다.

### 자동 프롬프트 합성

사용자 피드백(L_bad -> L_fixed)의 편집 거리(Edit Distance)와 의미적 차이를 분석하여 규칙 세트(Rule Set)를 추출하고, 런타임에 강제 주입 가능한 Hook Script(`generated_prompt_hook.py`)를 자동 생성한다.

### 벤더 독립적 파인튜닝

`BaseTrainer` 추상 클래스를 기반으로 Strategy Pattern을 적용하여 벤더 종속 없이 파인튜닝을 실행한다. 현재 `OpenAITrainer`가 구현되어 있으며, 로컬 모델 등으로 확장 가능하다.

### 실시간 모니터링 대시보드

- **CLI 대시보드** (rich 기반): 진행률 바, 단계별 상태, 실시간 로그를 터미널에 출력
- **Web 대시보드** (Streamlit 기반): 파이프라인 상태 모니터링, 벡터 분포 시각화, 학습 곡선 표시

---

## 설치

```bash
pip install -e ".[dev]"
```

---

## 사용법

```python
from vector_trainer.extractor import DensityBasedExtractor
from vector_trainer.synthesizer import FeedbackDiffAnalyzer, RuleSetSynthesizer, HookScriptGenerator
from vector_trainer.pipeline import OpenAITrainer, TrainingPipeline

# 1. 데이터 선별 엔진 초기화
extractor = DensityBasedExtractor(
    dataset_manager=your_vectorwave_dataset_manager,
    epsilon=0.3,
    top_k=50,
)

# 2. Golden Data 추출 (ChatML JSONL)
golden_data_path = extractor.extract_golden_data(function_name="your_function")

# 3. 프롬프트 합성기 구성
analyzer = FeedbackDiffAnalyzer()
synthesizer = RuleSetSynthesizer()
hook_generator = HookScriptGenerator()

# 4. 파인튜닝 파이프라인 실행
trainer = OpenAITrainer(api_key="your-api-key")
pipeline = TrainingPipeline(
    extractor=extractor,
    synthesizer=synthesizer,
    trainer=trainer,
)
pipeline.run(
    function_name="your_function",
    feedback_pairs=feedback_pairs,
    output_dir="./output",
)
```

---

## 테스트

```bash
pytest tests/ -v
```

외부 의존성(VectorWave DB, OpenAI API)은 Mocking 처리되어 있어, API 키 없이도 테스트를 실행할 수 있다.

---

## 기술 스택

| 구분 | 기술 |
|---|---|
| 언어 | Python >= 3.10 |
| 빌드 | setuptools (pyproject.toml) |
| 외부 의존성 | VectorWave, OpenAI API, NumPy |
| 벡터 DB | Weaviate (VectorWave를 통해 간접 사용) |
| CLI 대시보드 | rich |
| 웹 대시보드 | Streamlit |
| 테스트 | pytest, pytest-mock |
| 버전 관리 | Git, GitHub |

---

## 팀

| 팀명 | cozymori |
|---|---|
| 김준영 | 백엔드 |
| 조은설 | 백엔드, 기획 |

---

## 라이선스

MIT License
