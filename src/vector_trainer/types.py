"""Shared types, protocols, and data classes for VectorTrainer."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol: VectorWave DatasetManager 인터페이스 (ADR-001)
# ---------------------------------------------------------------------------

@runtime_checkable
class CollectionQueryProtocol(Protocol):
    """Weaviate Collection의 쿼리 인터페이스 부분 집합."""

    def fetch_objects(self, *, filters: Any = None, include_vector: bool = False,
                      limit: int = 100, **kwargs: Any) -> Any: ...

    def near_vector(self, *, near_vector: List[float], limit: int = 10,
                    filters: Any = None, return_metadata: Any = None,
                    include_vector: bool = False, **kwargs: Any) -> Any: ...


@runtime_checkable
class CollectionProtocol(Protocol):
    """Weaviate Collection 프로토콜."""

    @property
    def query(self) -> CollectionQueryProtocol: ...


@runtime_checkable
class DatasetManagerProtocol(Protocol):
    """VectorWaveDatasetManager가 만족해야 하는 인터페이스.

    VectorTrainer는 이 Protocol에만 의존하며, vectorwave 패키지를
    직접 import하지 않는다.
    """

    @property
    def exec_col(self) -> CollectionProtocol: ...

    @property
    def golden_col(self) -> CollectionProtocol: ...

    def recommend_candidates(self, function_name: str, limit: int = 5) -> List[Dict[str, Any]]: ...

    def register_as_golden(self, log_uuid: str, note: str = "",
                           tags: Optional[List[str]] = None) -> bool: ...


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionLog:
    """벡터 공간의 노드(Node)로 사용되는 실행 로그."""

    uuid: str
    function_name: str
    return_value: str
    status: str  # "SUCCESS" | "FAILURE"
    vector: List[float]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenCandidate:
    """선별된 골든 데이터 후보."""

    uuid: str
    strategy: SelectionStrategy
    density: float
    distance_to_centroid: float
    return_value: str
    vector: List[float]


class SelectionStrategy(enum.Enum):
    """데이터 선별 전략."""

    STEADY = "steady"    # 고밀도 클러스터 중심점
    ANOMALY = "anomaly"  # 저밀도 + SUCCESS 노드


@dataclass(frozen=True)
class FeedbackPair:
    """사용자 피드백 쌍: L_bad → L_fixed."""

    bad_output: str
    fixed_output: str
    input_prompt: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """합성된 규칙 단위."""

    rule_id: str
    description: str
    condition: str   # 적용 조건
    action: str      # 수행할 변환
    priority: int = 0
    source_pair_index: int = -1  # 규칙의 출처가 된 FeedbackPair 인덱스


# ---------------------------------------------------------------------------
# Cost Estimation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostEstimate:
    """파인튜닝 비용 추정 결과."""

    token_count: int
    estimated_cost_usd: float
    budget_usd: float
    model: str
    n_epochs: int
    approved: bool


# ---------------------------------------------------------------------------
# Forward reference 해결을 위한 GoldenCandidate 재정의 방지 (SelectionStrategy 이후)
# GoldenCandidate는 이미 위에서 SelectionStrategy를 문자열 어노테이션으로 참조하므로
# __future__.annotations 덕분에 정상 동작한다.
# ---------------------------------------------------------------------------
