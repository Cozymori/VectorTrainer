"""API route definitions — wraps vector_trainer modules directly."""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from vector_trainer.cost_guard import (
    BudgetExceededError,
    count_tokens_in_jsonl,
    estimate_training_cost,
)
from vector_trainer.extractor import DensityBasedExtractor
from vector_trainer.pipeline import OpenAITrainer
from vector_trainer.synthesizer import (
    FeedbackDiffAnalyzer,
    HookScriptGenerator,
    HookVersionManager,
    RuleSetSynthesizer,
)
from vector_trainer.types import GoldenCandidate

from .deps import (
    ALL_FUNCTION_NAMES,
    FEEDBACK_PAIRS,
    get_mock_dataset_manager,
    pipeline_state,
)
from .schemas import (
    CostEstimateRequest,
    CostEstimateResponse,
    DiffAnalysisResponse,
    GoldenCandidateResponse,
    HookVersionResponse,
    PipelineRunRequest,
    RollbackRequest,
    RuleResponse,
    StatsResponse,
    TrainRequest,
    TrainStatusResponse,
)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Pipeline execution (mirrors streamlit_app.py _run_pipeline)
# ---------------------------------------------------------------------------


def _run_pipeline(
    epsilon: float,
    top_k: int,
    use_real: bool = False,
    function_name: str = "generate_review_summary",
) -> Dict[str, Any]:
    """Execute the extraction and synthesis pipeline."""
    results: Dict[str, Any] = {}
    stage_times: Dict[str, float] = {}

    output_dir = tempfile.mkdtemp(prefix="vectortrainer_api_")
    results["output_dir"] = output_dir

    is_all = function_name.startswith("전체")
    target_functions = ALL_FUNCTION_NAMES if is_all else [function_name]

    # Stage 1: Extract
    t0 = time.time()
    if use_real:
        from vectorwave import VectorWaveDatasetManager
        dataset_manager = VectorWaveDatasetManager()
    else:
        dataset_manager = get_mock_dataset_manager()

    all_steady: List[GoldenCandidate] = []
    all_anomaly: List[GoldenCandidate] = []
    per_function_stats: List[Dict[str, Any]] = []

    for func in target_functions:
        extractor = DensityBasedExtractor(
            dataset_manager=dataset_manager,
            epsilon=epsilon,
            top_k=top_k,
        )
        steady = extractor.select_steady_state(func)
        anomaly = extractor.select_anomalies(func)

        func_jsonl = os.path.join(output_dir, f"{func}_golden.jsonl")
        extractor.extract_golden_data(func, func_jsonl)

        all_steady.extend(steady)
        all_anomaly.extend(anomaly)
        per_function_stats.append({
            "function": func,
            "steady": len(steady),
            "anomaly": len(anomaly),
            "jsonl_path": func_jsonl,
        })

    jsonl_path = os.path.join(
        output_dir, "merged_golden.jsonl" if is_all else "golden_data.jsonl"
    )
    if is_all:
        with open(jsonl_path, "w", encoding="utf-8") as out_f:
            for stat in per_function_stats:
                try:
                    with open(stat["jsonl_path"], "r", encoding="utf-8") as in_f:
                        out_f.write(in_f.read())
                except FileNotFoundError:
                    pass
    else:
        jsonl_path = per_function_stats[0]["jsonl_path"]

    stage_times["추출"] = time.time() - t0
    results["steady_candidates"] = all_steady
    results["anomaly_candidates"] = all_anomaly
    results["jsonl_path"] = jsonl_path
    results["per_function_stats"] = per_function_stats

    # Stage 2: Synthesize
    t0 = time.time()
    analyzer = FeedbackDiffAnalyzer()
    diffs = [analyzer.analyze(pair) for pair in FEEDBACK_PAIRS]

    synthesizer = RuleSetSynthesizer()
    rules = synthesizer.synthesize(diffs)

    stage_times["합성"] = time.time() - t0
    results["diffs"] = diffs
    results["rules"] = rules

    # Stage 3: Prepare (hook script)
    t0 = time.time()
    generator = HookScriptGenerator()
    hook_script_path = generator.generate(rules, output_dir)

    with open(hook_script_path, "r", encoding="utf-8") as fh:
        hook_source = fh.read()

    stage_times["준비"] = time.time() - t0
    results["hook_script_path"] = hook_script_path
    results["hook_source"] = hook_source

    # Cost estimation
    try:
        token_count = count_tokens_in_jsonl(jsonl_path, "gpt-4o-mini-2024-07-18")
        cost_1_epoch = estimate_training_cost(
            token_count, "gpt-4o-mini-2024-07-18", n_epochs=1
        )
        cost_3_epoch = estimate_training_cost(
            token_count, "gpt-4o-mini-2024-07-18", n_epochs=3
        )
    except Exception:
        token_count = 0
        cost_1_epoch = 0.0
        cost_3_epoch = 0.0

    results["token_count"] = token_count
    results["cost_1_epoch"] = cost_1_epoch
    results["cost_3_epoch"] = cost_3_epoch

    # Stage 4: Training (mock)
    t0 = time.time()
    time.sleep(0.05)
    stage_times["학습"] = time.time() - t0
    results["mock_model"] = "ft:gpt-4o-mini-2024-07-18:org::demo"
    results["mock_job_id"] = "ftjob-demo456"
    results["mock_file_id"] = "file-demo123"

    results["stage_times"] = stage_times
    results["total_time"] = sum(stage_times.values())

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/pipeline/run")
async def run_pipeline(req: PipelineRunRequest) -> Dict[str, Any]:
    """Run the full extraction + synthesis pipeline."""
    results = _run_pipeline(
        epsilon=req.epsilon,
        top_k=req.top_k,
        use_real=req.use_real,
        function_name=req.function_name,
    )

    # Store in module-level state
    pipeline_state["results"] = results
    pipeline_state["epsilon"] = req.epsilon
    pipeline_state["top_k"] = req.top_k
    pipeline_state["function_name"] = req.function_name

    # Serialize candidates for JSON response
    steady = [
        {
            "uuid": c.uuid,
            "strategy": c.strategy.value,
            "density": c.density,
            "distance_to_centroid": c.distance_to_centroid,
            "return_value": c.return_value,
        }
        for c in results["steady_candidates"]
    ]
    anomaly = [
        {
            "uuid": c.uuid,
            "strategy": c.strategy.value,
            "density": c.density,
            "distance_to_centroid": c.distance_to_centroid,
            "return_value": c.return_value,
        }
        for c in results["anomaly_candidates"]
    ]

    return {
        "status": "ok",
        "total_time": results["total_time"],
        "stage_times": results["stage_times"],
        "steady_count": len(steady),
        "anomaly_count": len(anomaly),
        "rules_count": len(results["rules"]),
        "token_count": results["token_count"],
        "cost_1_epoch": results["cost_1_epoch"],
        "cost_3_epoch": results["cost_3_epoch"],
        "per_function_stats": results["per_function_stats"],
    }


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Return current pipeline state."""
    if "results" not in pipeline_state:
        return StatsResponse(has_results=False)

    res = pipeline_state["results"]
    return StatsResponse(
        has_results=True,
        total_time=res["total_time"],
        stages_completed=4,
        golden_candidates_count=(
            len(res["steady_candidates"]) + len(res["anomaly_candidates"])
        ),
        rules_count=len(res["rules"]),
        token_count=res.get("token_count", 0),
        cost_1_epoch=res.get("cost_1_epoch", 0.0),
        cost_3_epoch=res.get("cost_3_epoch", 0.0),
        stage_times=res["stage_times"],
        per_function_stats=[
            {k: v for k, v in s.items() if k != "jsonl_path"}
            for s in res.get("per_function_stats", [])
        ],
        output_dir=res.get("output_dir"),
        jsonl_path=res.get("jsonl_path"),
        hook_script_path=res.get("hook_script_path"),
        mock_model=res.get("mock_model"),
        mock_job_id=res.get("mock_job_id"),
        function_name=pipeline_state.get("function_name"),
        epsilon=pipeline_state.get("epsilon"),
        top_k=pipeline_state.get("top_k"),
    )


@router.get("/golden-data", response_model=List[GoldenCandidateResponse])
async def get_golden_data() -> List[GoldenCandidateResponse]:
    """Return the list of golden data candidates."""
    if "results" not in pipeline_state:
        return []

    res = pipeline_state["results"]
    candidates = res["steady_candidates"] + res["anomaly_candidates"]
    return [
        GoldenCandidateResponse(
            uuid=c.uuid,
            strategy=c.strategy.value,
            density=c.density,
            distance_to_centroid=c.distance_to_centroid,
            return_value=c.return_value,
        )
        for c in candidates
    ]


@router.get("/golden-data/download")
async def download_golden_data() -> StreamingResponse:
    """Stream the JSONL golden data file."""
    if "results" not in pipeline_state:
        raise HTTPException(status_code=404, detail="No pipeline results available")

    jsonl_path = pipeline_state["results"]["jsonl_path"]
    if not os.path.isfile(jsonl_path):
        raise HTTPException(status_code=404, detail="JSONL file not found")

    def stream():
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            yield from fh

    return StreamingResponse(
        stream(),
        media_type="application/jsonl",
        headers={"Content-Disposition": "attachment; filename=golden_data.jsonl"},
    )


@router.get("/rules")
async def get_rules() -> Dict[str, Any]:
    """Return synthesized rules and diff analysis."""
    if "results" not in pipeline_state:
        return {"rules": [], "diffs": []}

    res = pipeline_state["results"]
    rules = [
        {
            "rule_id": r.rule_id,
            "description": r.description,
            "condition": r.condition,
            "action": r.action,
            "priority": r.priority,
            "source_pair_index": r.source_pair_index,
        }
        for r in res["rules"]
    ]
    diffs = res.get("diffs", [])

    return {"rules": rules, "diffs": diffs}


@router.get("/hook-script")
async def get_hook_script() -> Dict[str, Any]:
    """Return the generated hook script source."""
    if "results" not in pipeline_state:
        raise HTTPException(status_code=404, detail="No pipeline results available")

    res = pipeline_state["results"]
    return {
        "source": res.get("hook_source", ""),
        "path": res.get("hook_script_path", ""),
    }


@router.get("/hook-versions", response_model=List[HookVersionResponse])
async def get_hook_versions() -> List[HookVersionResponse]:
    """List all hook script versions."""
    if "results" not in pipeline_state:
        return []

    output_dir = pipeline_state["results"].get("output_dir", "")
    if not output_dir:
        return []

    vm = HookVersionManager(output_dir)
    versions = vm.list_versions()
    return [
        HookVersionResponse(
            version_id=v["version_id"],
            timestamp=v["timestamp"],
            rules_count=v["rules_count"],
            active=v["active"],
            filename=v.get("filename"),
        )
        for v in versions
    ]


@router.post("/hook-rollback")
async def rollback_hook(req: RollbackRequest) -> Dict[str, str]:
    """Rollback hook script to a specific version."""
    if "results" not in pipeline_state:
        raise HTTPException(status_code=404, detail="No pipeline results available")

    output_dir = pipeline_state["results"].get("output_dir", "")
    if not output_dir:
        raise HTTPException(status_code=400, detail="No output directory")

    vm = HookVersionManager(output_dir)
    try:
        rolled = vm.rollback(req.version_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Reload hook source
    hook_path = pipeline_state["results"].get("hook_script_path", "")
    if hook_path and os.path.isfile(hook_path):
        with open(hook_path, "r", encoding="utf-8") as fh:
            pipeline_state["results"]["hook_source"] = fh.read()

    return {"status": "ok", "active_version": rolled}


@router.post("/cost-estimate", response_model=CostEstimateResponse)
async def cost_estimate(req: CostEstimateRequest) -> CostEstimateResponse:
    """Estimate training cost for the current JSONL data."""
    if "results" not in pipeline_state:
        raise HTTPException(status_code=404, detail="No pipeline results available")

    jsonl_path = pipeline_state["results"]["jsonl_path"]
    if not os.path.isfile(jsonl_path):
        raise HTTPException(status_code=404, detail="JSONL file not found")

    token_count = count_tokens_in_jsonl(jsonl_path, req.model)
    cost = estimate_training_cost(token_count, req.model, req.n_epochs)

    return CostEstimateResponse(
        token_count=token_count,
        cost_estimate=cost,
        model=req.model,
        n_epochs=req.n_epochs,
    )


@router.post("/pipeline/train")
async def start_training(req: TrainRequest) -> Dict[str, str]:
    """Start an OpenAI fine-tuning job."""
    if "results" not in pipeline_state:
        raise HTTPException(status_code=404, detail="No pipeline results available")

    jsonl_path = pipeline_state["results"]["jsonl_path"]
    if not os.path.isfile(jsonl_path):
        raise HTTPException(status_code=404, detail="JSONL file not found")

    try:
        trainer = OpenAITrainer(
            api_key=req.api_key,
            model=req.model,
            max_budget_usd=req.max_budget_usd,
        )
        file_id = trainer.prepare_data(jsonl_path)
        job_id = trainer.start_training(
            file_id, hyperparameters={"n_epochs": req.n_epochs}
        )

        pipeline_state["trainer"] = trainer
        pipeline_state["ft_file_id"] = file_id
        pipeline_state["ft_job_id"] = job_id

        return {"status": "ok", "file_id": file_id, "job_id": job_id}
    except BudgetExceededError as e:
        raise HTTPException(status_code=400, detail=f"Budget exceeded: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/status/{job_id}")
async def get_training_status(job_id: str) -> Dict[str, Any]:
    """Check fine-tuning job status."""
    trainer = pipeline_state.get("trainer")
    if trainer is None:
        raise HTTPException(
            status_code=400,
            detail="No trainer instance. Start training first or provide API key.",
        )

    try:
        status = trainer.get_status(job_id)
        result: Dict[str, Any] = dict(status)

        if status.get("status") == "succeeded":
            full_result = trainer.get_result(job_id)
            result["fine_tuned_model"] = full_result.get("fine_tuned_model")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
