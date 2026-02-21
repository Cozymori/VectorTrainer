"""VectorTrainer Streamlit Dashboard.

A standalone Streamlit web application that demonstrates the VectorTrainer
pipeline visually.  Uses mock data (no real VectorWave connection needed)
and caches results in ``st.session_state`` across reruns.

Run with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VectorTrainer",
    page_icon="\u2699\ufe0f",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Imports from the VectorTrainer package
# ---------------------------------------------------------------------------

from vector_trainer.extractor import DensityBasedExtractor
from vector_trainer.synthesizer import (
    FeedbackDiffAnalyzer,
    HookScriptGenerator,
    RuleSetSynthesizer,
    run_synthesis_pipeline,
)
from vector_trainer.types import FeedbackPair, GoldenCandidate, SelectionStrategy


# ---------------------------------------------------------------------------
# Mock DatasetManager (same approach as demo.py)
# ---------------------------------------------------------------------------


def _create_mock_dataset_manager() -> Any:
    """Build a mock DatasetManager that simulates VectorWave execution logs."""

    np.random.seed(42)

    # Cluster A: high-density (similar responses) -- review summarisation
    cluster_a = [
        {
            "uuid": f"log-a{i}",
            "function_name": "generate_review_summary",
            "return_value": (
                f"This product has excellent quality and fast delivery. "
                f"Rating: {4 + (i % 2)}/5"
            ),
            "status": "SUCCESS",
            "vector": list(
                np.random.normal(loc=[0.8, 0.6, 0.7, 0.5], scale=0.05, size=4)
            ),
        }
        for i in range(15)
    ]

    # Cluster B: medium-density -- different review type
    cluster_b = [
        {
            "uuid": f"log-b{i}",
            "function_name": "generate_review_summary",
            "return_value": (
                f"The user experience is average. Some areas need improvement. "
                f"Rating: {2 + (i % 2)}/5"
            ),
            "status": "SUCCESS",
            "vector": list(
                np.random.normal(loc=[0.3, 0.8, 0.2, 0.6], scale=0.05, size=4)
            ),
        }
        for i in range(8)
    ]

    # Anomalies: low-density SUCCESS nodes -- unusual success cases
    anomalies = [
        {
            "uuid": "log-anomaly-1",
            "function_name": "generate_review_summary",
            "return_value": (
                "This product is a work of art. "
                "It has emotional value beyond its functionality."
            ),
            "status": "SUCCESS",
            "vector": [0.1, 0.1, 0.9, 0.1],
        },
        {
            "uuid": "log-anomaly-2",
            "function_name": "generate_review_summary",
            "return_value": (
                "Technical specs are inferior to competitors, "
                "but the UX design fully compensates for it."
            ),
            "status": "SUCCESS",
            "vector": [0.9, 0.1, 0.1, 0.9],
        },
    ]

    # Failure logs (should be excluded from selection)
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

    # Simulate Weaviate response objects
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


# ---------------------------------------------------------------------------
# Sample feedback pairs
# ---------------------------------------------------------------------------

_FEEDBACK_PAIRS: List[FeedbackPair] = [
    FeedbackPair(
        input_prompt="Summarise this product review",
        bad_output="Good product",
        fixed_output=(
            "This product has outstanding quality, particularly praised "
            "for its durability and design. Delivery is also fast."
        ),
        context={"function": "generate_review_summary"},
    ),
    FeedbackPair(
        input_prompt="Analyse user reviews",
        bad_output="Not great",
        fixed_output=(
            "User review analysis: positive aspects (design, price) and "
            "areas for improvement (after-sales service, manual) coexist. "
            "Overall satisfaction is 3.8/5."
        ),
        context={"function": "generate_review_summary"},
    ),
    FeedbackPair(
        input_prompt="List the key points of this review",
        bad_output="Key points: good",
        fixed_output=(
            "Key points: 1) Excellent performance-to-price ratio "
            "2) Battery life improved 30% over competitors "
            "3) After-sales response speed needs improvement"
        ),
        context={"function": "generate_review_summary"},
    ),
]


# ---------------------------------------------------------------------------
# Pipeline execution (cached in session_state)
# ---------------------------------------------------------------------------


def _run_pipeline(epsilon: float, top_k: int) -> Dict[str, Any]:
    """Execute the extraction and synthesis stages, returning all results.

    This function is called once when the user clicks *Run Pipeline* and
    the results are stored in ``st.session_state`` so that subsequent
    reruns simply re-render the cached data.
    """
    results: Dict[str, Any] = {}
    stage_times: Dict[str, float] = {}

    output_dir = tempfile.mkdtemp(prefix="vectortrainer_st_")
    results["output_dir"] = output_dir

    # -- Stage 1: Extract ---------------------------------------------------
    t0 = time.time()
    dataset_manager = _create_mock_dataset_manager()
    extractor = DensityBasedExtractor(
        dataset_manager=dataset_manager,
        epsilon=epsilon,
        top_k=top_k,
    )

    steady_candidates = extractor.select_steady_state("generate_review_summary")
    anomaly_candidates = extractor.select_anomalies("generate_review_summary")

    jsonl_path = os.path.join(output_dir, "golden_data.jsonl")
    extractor.extract_golden_data("generate_review_summary", jsonl_path)

    stage_times["Extract"] = time.time() - t0
    results["steady_candidates"] = steady_candidates
    results["anomaly_candidates"] = anomaly_candidates
    results["jsonl_path"] = jsonl_path

    # -- Stage 2: Synthesize ------------------------------------------------
    t0 = time.time()
    analyzer = FeedbackDiffAnalyzer()
    diffs = [analyzer.analyze(pair) for pair in _FEEDBACK_PAIRS]

    synthesizer = RuleSetSynthesizer()
    rules = synthesizer.synthesize(diffs)

    stage_times["Synthesize"] = time.time() - t0
    results["diffs"] = diffs
    results["rules"] = rules

    # -- Stage 3: Prepare (generate hook script) ----------------------------
    t0 = time.time()
    generator = HookScriptGenerator()
    hook_script_path = generator.generate(rules, output_dir)

    with open(hook_script_path, "r", encoding="utf-8") as fh:
        hook_source = fh.read()

    stage_times["Prepare"] = time.time() - t0
    results["hook_script_path"] = hook_script_path
    results["hook_source"] = hook_source

    # -- Stage 4: Train (mock) ----------------------------------------------
    t0 = time.time()
    # Simulate a short training delay
    time.sleep(0.15)
    stage_times["Train"] = time.time() - t0
    results["mock_model"] = "ft:gpt-4o-mini-2024-07-18:org::demo"
    results["mock_job_id"] = "ftjob-demo456"
    results["mock_file_id"] = "file-demo123"

    results["stage_times"] = stage_times
    results["total_time"] = sum(stage_times.values())

    return results


# ---------------------------------------------------------------------------
# UI: Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")
st.sidebar.markdown("---")

epsilon = st.sidebar.slider(
    "Epsilon (neighbour threshold)",
    min_value=0.1,
    max_value=0.5,
    value=0.3,
    step=0.01,
    help="Controls sparsity of the similarity graph. Lower values create a stricter graph.",
)

top_k = st.sidebar.slider(
    "Top K (candidates per strategy)",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    help="Maximum number of golden data candidates to return per selection strategy.",
)

st.sidebar.markdown("---")

run_clicked = st.sidebar.button(
    "Run Pipeline",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    with st.spinner("Running VectorTrainer pipeline..."):
        st.session_state["pipeline_results"] = _run_pipeline(epsilon, top_k)
        st.session_state["epsilon"] = epsilon
        st.session_state["top_k"] = top_k

# Show current parameters in sidebar
if "pipeline_results" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Last run parameters:**")
    st.sidebar.write(f"- epsilon = {st.session_state.get('epsilon', 'N/A')}")
    st.sidebar.write(f"- top_k = {st.session_state.get('top_k', 'N/A')}")


# ---------------------------------------------------------------------------
# UI: Title
# ---------------------------------------------------------------------------

st.title("VectorTrainer Dashboard")
st.caption(
    "Self-Evolving AI Loop -- density-based golden data extraction, "
    "feedback synthesis, and fine-tuning pipeline monitor"
)

st.markdown("---")


# ---------------------------------------------------------------------------
# UI: Tabs
# ---------------------------------------------------------------------------

tab_pipeline, tab_golden, tab_rules, tab_hook = st.tabs(
    ["Pipeline", "Golden Data", "Rules", "Hook Script"]
)

# ============================== TAB 1: Pipeline ==============================
with tab_pipeline:
    if "pipeline_results" not in st.session_state:
        st.info(
            "No pipeline results yet. Configure parameters in the sidebar "
            "and click **Run Pipeline** to start."
        )
    else:
        res = st.session_state["pipeline_results"]
        stage_times: Dict[str, float] = res["stage_times"]

        # -- Overall metrics -------------------------------------------------
        st.subheader("Overall Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Time", f"{res['total_time']:.3f}s")
        m2.metric("Stages Completed", "4 / 4")
        m3.metric(
            "Golden Candidates",
            len(res["steady_candidates"]) + len(res["anomaly_candidates"]),
        )
        m4.metric("Rules Synthesized", len(res["rules"]))

        st.markdown("---")

        # -- Stage progress --------------------------------------------------
        st.subheader("Pipeline Stages")

        stages_info = [
            ("Extract", "Density-based golden data selection from vector graph"),
            ("Synthesize", "Feedback diff analysis and rule synthesis"),
            ("Prepare", "Hook script generation and data upload"),
            ("Train", "Fine-tuning job execution"),
        ]

        # Visual stage flow
        cols = st.columns(len(stages_info))
        for idx, (stage_name, stage_desc) in enumerate(stages_info):
            with cols[idx]:
                elapsed = stage_times.get(stage_name, 0.0)
                st.success(f"**{stage_name}**")
                st.metric(f"Elapsed", f"{elapsed:.3f}s")
                st.caption(stage_desc)

        st.markdown("---")

        # -- Stage timeline bar chart ----------------------------------------
        st.subheader("Stage Timing Breakdown")
        timing_df = pd.DataFrame(
            {
                "Stage": list(stage_times.keys()),
                "Time (s)": list(stage_times.values()),
            }
        )
        st.bar_chart(timing_df, x="Stage", y="Time (s)", horizontal=False)

        st.markdown("---")

        # -- Output artifacts ------------------------------------------------
        st.subheader("Output Artifacts")
        st.write(f"**Output directory:** `{res['output_dir']}`")
        st.write(f"**Golden data JSONL:** `{res['jsonl_path']}`")
        st.write(f"**Hook script:** `{res['hook_script_path']}`")
        st.write(f"**Mock fine-tuned model:** `{res['mock_model']}`")
        st.write(f"**Mock job ID:** `{res['mock_job_id']}`")


# ============================== TAB 2: Golden Data ===========================
with tab_golden:
    if "pipeline_results" not in st.session_state:
        st.info("Run the pipeline first to see golden data results.")
    else:
        res = st.session_state["pipeline_results"]
        steady: List[GoldenCandidate] = res["steady_candidates"]
        anomalies: List[GoldenCandidate] = res["anomaly_candidates"]

        # -- Counts ----------------------------------------------------------
        st.subheader("Selection Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Steady State", len(steady))
        c2.metric("Anomaly", len(anomalies))
        c3.metric("Total (deduplicated)", len(steady) + len(anomalies))

        st.markdown("---")

        # -- Combined dataframe ----------------------------------------------
        st.subheader("Extracted Golden Data")

        all_candidates = steady + anomalies
        if all_candidates:
            df_data = []
            for c in all_candidates:
                df_data.append(
                    {
                        "UUID": c.uuid,
                        "Strategy": c.strategy.value,
                        "Density": round(c.density, 4),
                        "Dist. to Centroid": round(c.distance_to_centroid, 4),
                        "Return Value": c.return_value[:80] + (
                            "..." if len(c.return_value) > 80 else ""
                        ),
                    }
                )
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No golden candidates were selected.")

        st.markdown("---")

        # -- Density distribution chart --------------------------------------
        st.subheader("Density Distribution")

        if all_candidates:
            density_data = []
            for c in all_candidates:
                density_data.append(
                    {"Density": c.density, "Strategy": c.strategy.value}
                )
            density_df = pd.DataFrame(density_data)

            # Show as a bar chart grouped by strategy
            steady_densities = [c.density for c in steady]
            anomaly_densities = [c.density for c in anomalies]

            # Create histogram-like bins
            all_densities = [c.density for c in all_candidates]
            if all_densities:
                bin_min = min(all_densities)
                bin_max = max(all_densities)
                n_bins = 15
                bins = np.linspace(bin_min, bin_max, n_bins + 1)
                steady_counts, _ = np.histogram(steady_densities, bins=bins)
                anomaly_counts, _ = np.histogram(anomaly_densities, bins=bins)
                bin_labels = [f"{b:.2f}" for b in bins[:-1]]

                hist_df = pd.DataFrame(
                    {
                        "Density Bin": bin_labels,
                        "Steady": steady_counts,
                        "Anomaly": anomaly_counts,
                    }
                )
                st.bar_chart(hist_df, x="Density Bin", y=["Steady", "Anomaly"])

        st.markdown("---")

        # -- JSONL preview ---------------------------------------------------
        st.subheader("ChatML JSONL Preview")

        jsonl_path = res["jsonl_path"]
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            if lines:
                preview_count = min(3, len(lines))
                st.caption(
                    f"Showing first {preview_count} of {len(lines)} records"
                )
                for i, line in enumerate(lines[:preview_count]):
                    with st.expander(f"Record {i + 1}", expanded=(i == 0)):
                        record = json.loads(line)
                        st.json(record)
            else:
                st.warning("JSONL file is empty.")
        except FileNotFoundError:
            st.error(f"JSONL file not found: {jsonl_path}")


# ============================== TAB 3: Rules =================================
with tab_rules:
    if "pipeline_results" not in st.session_state:
        st.info("Run the pipeline first to see synthesized rules.")
    else:
        res = st.session_state["pipeline_results"]
        rules = res["rules"]
        diffs = res["diffs"]

        st.subheader("Synthesized Rules")

        if rules:
            # -- Rules table -------------------------------------------------
            rules_data = []
            for r in rules:
                rules_data.append(
                    {
                        "Rule ID": r.rule_id,
                        "Priority": r.priority,
                        "Description": r.description[:100] + (
                            "..." if len(r.description) > 100 else ""
                        ),
                        "Source Pair": r.source_pair_index,
                    }
                )
            rules_df = pd.DataFrame(rules_data)
            st.dataframe(rules_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # -- Rule details in expandable sections -------------------------
            st.subheader("Rule Details")

            for r in rules:
                with st.expander(f"{r.rule_id} -- Priority {r.priority}"):
                    st.markdown(f"**Description:** {r.description}")
                    st.markdown("**Condition:**")
                    st.code(r.condition, language="text")
                    st.markdown("**Action:**")
                    st.code(r.action, language="text")

                    if 0 <= r.source_pair_index < len(_FEEDBACK_PAIRS):
                        pair = _FEEDBACK_PAIRS[r.source_pair_index]
                        st.markdown("---")
                        st.markdown("**Source Feedback Pair:**")
                        st.markdown(f"- **Prompt:** {pair.input_prompt}")
                        st.markdown(f"- **Bad output:** {pair.bad_output}")
                        st.markdown(f"- **Fixed output:** {pair.fixed_output}")
        else:
            st.warning("No rules were synthesized.")

        st.markdown("---")

        # -- Diff analysis ---------------------------------------------------
        st.subheader("Feedback Diff Analysis")

        if diffs:
            for i, diff in enumerate(diffs):
                with st.expander(
                    f"Diff #{i + 1} -- Edit distance: {diff['edit_distance']}, "
                    f"Similarity: {diff['similarity_score']:.3f}",
                    expanded=False,
                ):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**Bad output:**")
                        st.code(diff["bad_output"], language="text")
                    with d2:
                        st.markdown("**Fixed output:**")
                        st.code(diff["fixed_output"], language="text")

                    st.markdown(f"**Summary:** {diff['diff_summary']}")

                    if diff["diff_segments"]:
                        st.markdown("**Diff segments:**")
                        for seg in diff["diff_segments"]:
                            seg_type = seg["type"]
                            if seg_type == "added":
                                st.markdown(
                                    f"- :green[**+** Added:] `{seg['fixed'][:80]}`"
                                )
                            elif seg_type == "removed":
                                st.markdown(
                                    f"- :red[**-** Removed:] `{seg['bad'][:80]}`"
                                )
                            elif seg_type == "changed":
                                st.markdown(
                                    f"- :orange[**~** Changed:] "
                                    f"`{seg['bad'][:40]}` -> `{seg['fixed'][:40]}`"
                                )


# ============================== TAB 4: Hook Script ===========================
with tab_hook:
    if "pipeline_results" not in st.session_state:
        st.info("Run the pipeline first to see the generated hook script.")
    else:
        res = st.session_state["pipeline_results"]
        hook_source = res["hook_source"]

        st.subheader("Generated Hook Script")
        st.caption(
            f"File: `{res['hook_script_path']}`"
        )

        st.code(hook_source, language="python", line_numbers=True)

        st.markdown("---")

        # -- Quick explanation -----------------------------------------------
        st.subheader("How the Hook Script Works")

        st.markdown(
            """
The generated hook script exposes two public functions:

1. **`intercept_prompt(user_input, context)`** -- Pre-processes the user
   prompt before it reaches the model. Applies all matching rules (sorted
   by priority, descending) to the input text.

2. **`intercept_output(model_output, context)`** -- Post-processes the model
   output before returning it to the user. Applies the same rule-matching
   logic to the generated text.

**Rule matching** uses simple substring detection: quoted strings inside
each rule's `condition` are checked against the text. The `action` field
supports three operations:

- `Replace "X" with "Y"` -- substitution
- `Remove: "X"` -- deletion
- `Add: "X"` -- appending text
"""
        )
