# VectorTrainer 기능 요구사항 명세 (SPEC)

> Self-Evolving AI Loop를 위한 VectorTrainer 핵심 엔진

## 개요

VectorTrainer는 VectorWave 라이브러리를 외부 의존성으로 활용하여, LLM 서비스의 운영과 개선을 자동화하는 독립 패키지이다. 인간의 개입 없이 모델 성능을 고도화하는 "Self-Evolving AI Loop"를 완성한다.

---

## FR-1: 이산수학적 데이터 선별 엔진 (Discrete Density-based Selection)

### 목적
VectorWave에서 추출한 로그 데이터셋을 가중치 그래프(Weighted Graph)로 간주하고, 학습 효율이 높은 'Golden Data'를 자동 선별한다.

### 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-1.1 | 각 실행 로그를 벡터 공간의 노드로 정의하고, 코사인 유사도를 에지 가중치로 설정한다 | 필수 |
| FR-1.2 | 밀도 함수 D(L_i) = Σ_{j ∈ N(L_i)} W_ij를 구현한다 (ε 임계값 기반 인접 노드 집합) | 필수 |
| FR-1.3 | Steady State: 밀도가 높은 클러스터의 중심점을 추출하여 파인튜닝 데이터로 활용한다 | 필수 |
| FR-1.4 | Anomaly Detection: 밀도가 낮지만 SUCCESS인 노드를 '새로운 패턴'으로 선별한다 | 필수 |
| FR-1.5 | VectorWaveDatasetManager를 DI로 주입받아 ANN Search를 수행한다 | 필수 |
| FR-1.6 | ChatML 포맷({"messages": [...]})의 JSONL로 데이터를 출력한다 | 필수 |

---

## FR-2: 자동 프롬프트 생성 파이프라인 (Automatic Criteria Synthesis)

### 목적
선별된 데이터로부터 '지침'을 추출하고, 런타임에 강제 주입 가능한 물리적 스크립트 파일로 출력한다.

### 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-2.1 | 사용자 피드백(L_bad → L_fixed) 쌍을 입력으로 받는다 | 필수 |
| FR-2.2 | 문자열 차이(Edit Distance)와 의미적 차이를 분석하여 변경점을 추출한다 | 필수 |
| FR-2.3 | 추출된 차이점을 이산적 규칙 세트(Rule Set, R)로 변환한다 | 필수 |
| FR-2.4 | 모순되는 규칙을 감지하고 병합한다 | 필수 |
| FR-2.5 | 규칙 세트를 물리적 Python 스크립트 파일(Hook Script)로 생성하여 지정 경로에 Export한다 | 필수 |
| FR-2.6 | 생성된 스크립트는 사용자 Input과 모델 Output 사이에 개입하는 인터셉터 역할을 한다 | 필수 |

---

## FR-3: 추상화된 파인튜닝 파이프라인 (Training Pipeline)

### 목적
특정 벤더에 종속되지 않는 추상화 인터페이스를 정의하고, Strategy Pattern으로 Trainer를 교체 가능하게 설계한다.

### 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-3.1 | BaseTrainer 추상 기반 클래스를 정의한다 (prepare_data, start_training, get_status, get_result) | 필수 |
| FR-3.2 | OpenAITrainer를 BaseTrainer의 구체 구현으로 작성한다 | 필수 |
| FR-3.3 | TrainingPipeline 오케스트레이터가 Extractor, Synthesizer, Trainer를 조합하여 전체 플로우를 실행한다 | 필수 |
| FR-3.4 | 추후 로컬 모델(Local LLM) 파인튜닝 확장을 위해 추상화를 유지한다 | 권장 |

---

## FR-4: 테스트 및 유지보수 (Testability)

### 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-4.1 | pytest를 사용한 단위 테스트 코드를 작성한다 | 필수 |
| FR-4.2 | 외부 의존성(DB, OpenAI API)은 Mocking 처리한다 | 필수 |
| FR-4.3 | 이산수학적 밀도 계산 로직을 검증한다 | 필수 |
| FR-4.4 | 동적 생성 스크립트 파일의 구문 정확성을 AST로 검증한다 | 필수 |
