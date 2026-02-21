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
