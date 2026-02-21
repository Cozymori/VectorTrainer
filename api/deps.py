"""Shared state, mock data factories, and constants for the API layer."""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock

import numpy as np
from dotenv import load_dotenv

from vector_trainer.types import FeedbackPair

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level state (replaces st.session_state)
# ---------------------------------------------------------------------------

pipeline_state: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_FUNCTION_NAMES: List[str] = [
    "generate_review_summary",
    "extract_sentiment",
    "generate_product_recommendation",
    "translate_review",
]

feedback_pairs_store: List[Dict[str, Any]] = [
    {
        "id": "fp-001",
        "input_prompt": "이 제품의 리뷰를 요약해주세요",
        "bad_output": "좋은 제품입니다",
        "fixed_output": (
            "이 제품은 품질이 우수하며, 특히 내구성과 디자인 면에서 높은 평가를 받고 있습니다. "
            "배송 속도도 빠릅니다."
        ),
        "context": {"function": "generate_review_summary"},
    },
    {
        "id": "fp-002",
        "input_prompt": "사용자 리뷰를 분석해주세요",
        "bad_output": "별로입니다",
        "fixed_output": (
            "사용자 리뷰 분석 결과: 긍정 요소(디자인, 가격)와 개선 필요 요소(AS, 설명서)가 "
            "혼재합니다. 전반적 만족도: 3.8/5."
        ),
        "context": {"function": "generate_review_summary"},
    },
    {
        "id": "fp-003",
        "input_prompt": "이 리뷰의 핵심 포인트를 정리해주세요",
        "bad_output": "핵심 포인트: 좋음",
        "fixed_output": (
            "핵심 포인트: 1) 가격 대비 성능이 우수 2) 배터리 수명이 경쟁 제품 대비 "
            "30% 향상 3) A/S 대응 속도 개선 필요"
        ),
        "context": {"function": "generate_review_summary"},
    },
]
_next_fp_id: int = 4


def generate_fp_id() -> str:
    """Generate the next feedback-pair ID."""
    global _next_fp_id
    fp_id = f"fp-{_next_fp_id:03d}"
    _next_fp_id += 1
    return fp_id


def get_feedback_pairs_as_dataclass() -> List[FeedbackPair]:
    """Return FeedbackPair dataclass instances for pipeline execution."""
    return [
        FeedbackPair(
            input_prompt=item["input_prompt"],
            bad_output=item["bad_output"],
            fixed_output=item["fixed_output"],
            context=item.get("context", {}),
        )
        for item in feedback_pairs_store
    ]

# ---------------------------------------------------------------------------
# Mock DatasetManager factory
# ---------------------------------------------------------------------------


def get_mock_dataset_manager() -> Any:
    """Create a mock DatasetManager that simulates VectorWave execution logs."""
    np.random.seed(42)

    cluster_a = [
        {
            "uuid": f"log-a{i}",
            "function_name": "generate_review_summary",
            "return_value": (
                f"이 제품은 품질이 우수하고 배송이 빠릅니다. "
                f"평점: {4 + (i % 2)}/5"
            ),
            "status": "SUCCESS",
            "vector": list(
                np.random.normal(loc=[0.8, 0.6, 0.7, 0.5], scale=0.05, size=4)
            ),
        }
        for i in range(15)
    ]

    cluster_b = [
        {
            "uuid": f"log-b{i}",
            "function_name": "generate_review_summary",
            "return_value": (
                f"사용 경험이 보통입니다. 개선이 필요한 부분이 있습니다. "
                f"평점: {2 + (i % 2)}/5"
            ),
            "status": "SUCCESS",
            "vector": list(
                np.random.normal(loc=[0.3, 0.8, 0.2, 0.6], scale=0.05, size=4)
            ),
        }
        for i in range(8)
    ]

    anomalies = [
        {
            "uuid": "log-anomaly-1",
            "function_name": "generate_review_summary",
            "return_value": (
                "이 제품은 예술 작품입니다. 기능을 넘어선 감성적 가치가 있습니다."
            ),
            "status": "SUCCESS",
            "vector": [0.1, 0.1, 0.9, 0.1],
        },
        {
            "uuid": "log-anomaly-2",
            "function_name": "generate_review_summary",
            "return_value": (
                "기술적 스펙은 경쟁사 대비 열위이나, UX 설계가 이를 완전히 상쇄합니다."
            ),
            "status": "SUCCESS",
            "vector": [0.9, 0.1, 0.1, 0.9],
        },
    ]

    failures = [
        {
            "uuid": "log-fail-1",
            "function_name": "generate_review_summary",
            "return_value": "Error: context too long",
            "status": "FAILURE",
            "vector": [0.5, 0.5, 0.5, 0.5],
        },
    ]

    all_logs = cluster_a + cluster_b + anomalies + failures

    class MockWeaviateObject:
        def __init__(self, data: Dict[str, Any]) -> None:
            self.uuid = data["uuid"]
            self.properties = {
                "function_name": data["function_name"],
                "return_value": data["return_value"],
                "status": data["status"],
            }
            self.vector = {"default": data["vector"]}

    mock_objects = [MockWeaviateObject(log) for log in all_logs]

    class MockFetchResponse:
        def __init__(self, objects: list) -> None:
            self.objects = objects

    manager = MagicMock()

    exec_query = MagicMock()
    exec_query.fetch_objects.return_value = MockFetchResponse(mock_objects)
    exec_col = MagicMock()
    type(exec_col).query = PropertyMock(return_value=exec_query)
    type(manager).exec_col = PropertyMock(return_value=exec_col)

    golden_query = MagicMock()
    golden_query.fetch_objects.return_value = MockFetchResponse([])
    golden_col = MagicMock()
    type(golden_col).query = PropertyMock(return_value=golden_query)
    type(manager).golden_col = PropertyMock(return_value=golden_col)

    return manager
