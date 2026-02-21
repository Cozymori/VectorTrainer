"""Pydantic request/response models for the VectorTrainer API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class PipelineRunRequest(BaseModel):
    epsilon: float = 0.3
    top_k: int = 50
    use_real: bool = False
    function_name: str = "generate_review_summary"


class TrainRequest(BaseModel):
    api_key: str
    model: str = "gpt-4o-mini-2024-07-18"
    n_epochs: int = 3
    max_budget_usd: float = 10.0


class CostEstimateRequest(BaseModel):
    model: str = "gpt-4o-mini-2024-07-18"
    n_epochs: int = 3


class RollbackRequest(BaseModel):
    version_id: str


class FeedbackPairCreate(BaseModel):
    input_prompt: str
    bad_output: str
    fixed_output: str
    context: Dict[str, Any] = {}


class FeedbackPairUpdate(BaseModel):
    input_prompt: Optional[str] = None
    bad_output: Optional[str] = None
    fixed_output: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FeedbackPairResponse(BaseModel):
    id: str
    input_prompt: str
    bad_output: str
    fixed_output: str
    context: Dict[str, Any]


class GoldenCandidateResponse(BaseModel):
    uuid: str
    strategy: str
    density: float
    distance_to_centroid: float
    return_value: str


class RuleResponse(BaseModel):
    rule_id: str
    description: str
    condition: str
    action: str
    priority: int
    source_pair_index: int


class DiffAnalysisResponse(BaseModel):
    edit_distance: int
    similarity_score: float
    diff_summary: str
    diff_segments: List[Dict[str, Any]]
    input_prompt: str
    context: Dict[str, Any]
    bad_output: str
    fixed_output: str


class StatsResponse(BaseModel):
    has_results: bool
    total_time: Optional[float] = None
    stages_completed: int = 0
    golden_candidates_count: int = 0
    rules_count: int = 0
    token_count: int = 0
    cost_1_epoch: float = 0.0
    cost_3_epoch: float = 0.0
    stage_times: Dict[str, float] = {}
    per_function_stats: List[Dict[str, Any]] = []
    output_dir: Optional[str] = None
    jsonl_path: Optional[str] = None
    hook_script_path: Optional[str] = None
    mock_model: Optional[str] = None
    mock_job_id: Optional[str] = None
    function_name: Optional[str] = None
    epsilon: Optional[float] = None
    top_k: Optional[int] = None


class HookVersionResponse(BaseModel):
    version_id: str
    timestamp: str
    rules_count: int
    active: bool
    filename: Optional[str] = None


class CostEstimateResponse(BaseModel):
    token_count: int
    cost_estimate: float
    model: str
    n_epochs: int


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    model: Optional[str] = None
    trained_tokens: Optional[int] = None
    created_at: Optional[Any] = None
    finished_at: Optional[Any] = None
    error: Optional[Any] = None
    fine_tuned_model: Optional[str] = None
