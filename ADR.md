# VectorTrainer 아키텍처 결정 기록 (ADR)

> Architecture Decision Records for VectorTrainer

---

## ADR-001: VectorWave를 Protocol 기반 DI로 연동

### 상태
채택 (Accepted)

### 컨텍스트
VectorTrainer는 VectorWave의 VectorWaveDatasetManager를 사용해야 하지만, 독립 패키지로서 VectorWave에 직접 의존하면 결합도가 높아진다.

### 결정
Python의 `typing.Protocol`을 사용하여 `DatasetManagerProtocol`을 정의하고, 런타임에 VectorWaveDatasetManager 인스턴스를 주입(Dependency Injection)받는다. VectorTrainer의 import 그래프에 vectorwave가 포함되지 않는다.

### 근거
- 단위 테스트에서 Mock 객체로 쉽게 대체 가능
- VectorWave 버전 변경에 영향받지 않음
- 추후 다른 데이터 소스로 교체 가능

---

## ADR-002: 밀도 함수 D(L_i) = Σ W_ij (코사인 유사도 기반 가중치 그래프)

### 상태
채택 (Accepted)

### 컨텍스트
Golden Data 선별을 위해 로그 데이터의 분포를 분석해야 한다. 단순 거리 기반 방식은 고차원 벡터 공간에서 의미적 군집을 정확히 포착하기 어렵다.

### 결정
- 각 실행 로그를 가중치 그래프의 노드로 모델링
- 노드 간 코사인 유사도를 에지 가중치(W)로 설정
- 임계값 ε 이내의 인접 노드 집합 N(L_i)에 대해 밀도 D(L_i) = Σ_{j ∈ N(L_i)} W_ij로 정의
- Steady State: 고밀도 클러스터 중심점 → 기본기 강화 데이터
- Anomaly Detection: 저밀도 + SUCCESS 노드 → 새로운 패턴

### 근거
- 이산수학의 그래프 이론을 활용한 엄밀한 모델링
- 코사인 유사도는 고차원 벡터 공간에서 방향 유사성을 잘 포착
- 밀도 기반 접근은 DBSCAN 계열 알고리즘과 이론적 기반을 공유

---

## ADR-003: Hook Script를 물리적 .py 파일로 생성 (런타임 동적 로드)

### 상태
채택 (Accepted)

### 컨텍스트
합성된 규칙을 런타임 서비스에 적용해야 한다. 메모리 내 규칙만으로는 서비스 재시작 시 소실되고, 버전 관리가 어렵다.

### 결정
규칙 세트를 물리적 Python 스크립트 파일(`generated_prompt_hook.py`)로 생성하여 지정 디렉토리에 Export한다. 이 스크립트는 사용자 Input과 모델 Output 사이에 개입하는 미들웨어(Interceptor) 역할을 한다.

### 근거
- 파일 기반이므로 Git 버전 관리 가능
- 서비스 재시작 후에도 규칙 유지
- AST 파싱으로 구문 정확성을 자동 검증 가능
- `importlib`으로 동적 로드 가능

---

## ADR-004: Strategy Pattern으로 Trainer 벤더 추상화

### 상태
채택 (Accepted)

### 컨텍스트
현재는 OpenAI Fine-tuning API를 사용하지만, 추후 로컬 모델(Local LLM) 파인튜닝으로의 확장이 예정되어 있다.

### 결정
`BaseTrainer` 추상 기반 클래스(ABC)를 정의하고, `OpenAITrainer`를 첫 번째 구체 구현으로 작성한다. `TrainingPipeline` 오케스트레이터는 Trainer 인터페이스에만 의존하여, 런타임에 다른 Trainer로 교체 가능하다.

### 근거
- OCP(Open-Closed Principle): 새 벤더 추가 시 기존 코드 수정 불필요
- 테스트에서 Mock Trainer 주입 가능
- 파인튜닝 벤더 전환 시 Pipeline 코드 변경 없음

---

## ADR-005: 콜백 기반 모니터링 (Observer Pattern)

### 상태
채택 (Accepted)

### 컨텍스트
파인튜닝 파이프라인의 진행 과정을 실시간으로 시각화해야 하며, CLI와 웹 두 가지 인터페이스를 모두 지원해야 한다.

### 결정
PipelineMonitor 기반 클래스에 콜백 메서드(on_stage_start, on_progress, on_stage_complete)를 정의하고, CLIDashboard(rich)와 WebDashboard(Streamlit)가 이를 상속받아 구현한다. TrainingPipeline은 monitor 인스턴스를 선택적으로 주입받아 호출한다.

### 근거
- Observer Pattern으로 Pipeline 코드를 수정하지 않고 모니터링 계층 추가 가능
- CLI/Web 대시보드를 독립적으로 개발 및 교체 가능
- 모니터가 없어도 파이프라인은 정상 동작 (선택적 의존)

---

## ADR-006: tiktoken 기반 사전 비용 추정 (Cost Guard)

### 상태
채택 (Accepted)

### 컨텍스트
OpenAI Fine-tuning API는 학습 토큰 수에 비례하여 과금한다. 대규모 데이터셋을 실수로 업로드하면 예상치 못한 비용이 발생할 수 있다.

### 결정
파일 업로드(prepare_data) 전 tiktoken으로 JSONL의 토큰 수를 사전 계산하고, 모델별 단가를 적용하여 예상 비용을 산출한다. 사용자가 설정한 `max_budget_usd`를 초과하면 `BudgetExceededError`를 발생시켜 API 호출을 차단한다.

### 근거
- 비용 발생 전 차단(Shift-Left): 비싼 API 호출을 사전에 방지
- tiktoken은 OpenAI 공식 토크나이저로 정확한 토큰 수 제공
- 예산 파라미터가 None이면 가드레일 비활성화하여 기존 동작 유지 (하위 호환)
- CostEstimate 데이터클래스로 투명한 비용 정보 반환

---

## ADR-007: 파일 기반 훅 스크립트 버전 관리

### 상태
채택 (Accepted)

### 컨텍스트
기존 HookScriptGenerator는 매 실행마다 `generated_prompt_hook.py`를 덮어쓰기한다. 잘못된 규칙이 생성되더라도 이전 버전으로 즉시 복구할 방법이 없다.

### 결정
`.hook_versions/` 디렉토리에 타임스탬프 기반 버전 파일(v001_YYYYMMDDTHHMMSSZ.py)을 저장하고, `manifest.json`으로 메타데이터를 관리한다. `HookVersionManager.rollback()`으로 특정 또는 직전 버전으로 즉시 복원 가능하다.

### 근거
- 외부 도구(Git, DB) 없이 자체적으로 버전 관리 가능
- manifest.json으로 버전 이력을 투명하게 추적
- 롤백 시 기존 활성 파일을 안전하게 교체 (shutil.copy2)
- enable_versioning 플래그로 기존 코드와 하위 호환 유지
