"""VectorTrainer Streamlit 대시보드.

VectorTrainer 파이프라인을 시각적으로 보여주는 Streamlit 웹 애플리케이션.
Mock 데이터를 사용하며 (실제 VectorWave 연결 불필요),
``st.session_state`` 에 결과를 캐싱합니다.

실행::

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
# 페이지 설정 (Streamlit 첫 번째 명령어)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VectorTrainer 대시보드",
    page_icon="\u2699\ufe0f",
    layout="wide",
)

# ---------------------------------------------------------------------------
# VectorTrainer 패키지 임포트
# ---------------------------------------------------------------------------

from vector_trainer.cost_guard import (
    BudgetExceededError,
    check_budget,
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
    run_synthesis_pipeline,
)
from vector_trainer.types import FeedbackPair, GoldenCandidate, SelectionStrategy


# ---------------------------------------------------------------------------
# Mock DatasetManager (demo.py와 동일한 방식)
# ---------------------------------------------------------------------------


def _create_mock_dataset_manager() -> Any:
    """VectorWave 실행 로그를 시뮬레이션하는 Mock DatasetManager 생성."""

    np.random.seed(42)

    # 클러스터 A: 고밀도 (유사한 응답) -- 리뷰 요약
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

    # 클러스터 B: 중간 밀도 -- 다른 유형의 리뷰
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

    # 이상치: 저밀도 SUCCESS 노드 -- 특이한 성공 사례
    anomalies = [
        {
            "uuid": "log-anomaly-1",
            "function_name": "generate_review_summary",
            "return_value": (
                "이 제품은 예술 작품입니다. "
                "기능을 넘어선 감성적 가치가 있습니다."
            ),
            "status": "SUCCESS",
            "vector": [0.1, 0.1, 0.9, 0.1],
        },
        {
            "uuid": "log-anomaly-2",
            "function_name": "generate_review_summary",
            "return_value": (
                "기술적 스펙은 경쟁사 대비 열위이나, "
                "UX 설계가 이를 완전히 상쇄합니다."
            ),
            "status": "SUCCESS",
            "vector": [0.9, 0.1, 0.1, 0.9],
        },
    ]

    # 실패 로그 (선별에서 제외되어야 함)
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

    # Weaviate 응답 객체 시뮬레이션
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
# 샘플 피드백 쌍
# ---------------------------------------------------------------------------

_FEEDBACK_PAIRS: List[FeedbackPair] = [
    FeedbackPair(
        input_prompt="이 제품의 리뷰를 요약해주세요",
        bad_output="좋은 제품입니다",
        fixed_output=(
            "이 제품은 품질이 우수하며, 특히 내구성과 디자인 면에서 높은 평가를 받고 있습니다. "
            "배송 속도도 빠릅니다."
        ),
        context={"function": "generate_review_summary"},
    ),
    FeedbackPair(
        input_prompt="사용자 리뷰를 분석해주세요",
        bad_output="별로입니다",
        fixed_output=(
            "사용자 리뷰 분석 결과: 긍정 요소(디자인, 가격)와 개선 필요 요소(AS, 설명서)가 "
            "혼재합니다. 전반적 만족도: 3.8/5."
        ),
        context={"function": "generate_review_summary"},
    ),
    FeedbackPair(
        input_prompt="이 리뷰의 핵심 포인트를 정리해주세요",
        bad_output="핵심 포인트: 좋음",
        fixed_output=(
            "핵심 포인트: 1) 가격 대비 성능이 우수 2) 배터리 수명이 경쟁 제품 대비 "
            "30% 향상 3) A/S 대응 속도 개선 필요"
        ),
        context={"function": "generate_review_summary"},
    ),
]


# ---------------------------------------------------------------------------
# 파이프라인 실행 (session_state에 캐싱)
# ---------------------------------------------------------------------------


_ALL_FUNCTION_NAMES = [
    "generate_review_summary",
    "extract_sentiment",
    "generate_product_recommendation",
    "translate_review",
]


def _run_pipeline(
    epsilon: float,
    top_k: int,
    use_real: bool = False,
    function_name: str = "generate_review_summary",
) -> Dict[str, Any]:
    """추출 및 합성 단계를 실행하고 모든 결과를 반환합니다."""
    results: Dict[str, Any] = {}
    stage_times: Dict[str, float] = {}

    output_dir = tempfile.mkdtemp(prefix="vectortrainer_st_")
    results["output_dir"] = output_dir

    # 전체 모드 판별
    is_all = function_name.startswith("전체")
    target_functions = _ALL_FUNCTION_NAMES if is_all else [function_name]

    # -- 단계 1: 추출 ----------------------------------------------------------
    t0 = time.time()
    if use_real:
        from vectorwave import VectorWaveDatasetManager
        dataset_manager = VectorWaveDatasetManager()
    else:
        dataset_manager = _create_mock_dataset_manager()

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

    # 통합 JSONL 생성
    jsonl_path = os.path.join(output_dir, "merged_golden.jsonl" if is_all else "golden_data.jsonl")
    if is_all:
        with open(jsonl_path, "w", encoding="utf-8") as out_f:
            for stat in per_function_stats:
                try:
                    with open(stat["jsonl_path"], "r", encoding="utf-8") as in_f:
                        out_f.write(in_f.read())
                except FileNotFoundError:
                    pass
    else:
        # 단일 함수: 이미 생성된 파일 사용
        jsonl_path = per_function_stats[0]["jsonl_path"]

    stage_times["추출"] = time.time() - t0
    results["steady_candidates"] = all_steady
    results["anomaly_candidates"] = all_anomaly
    results["jsonl_path"] = jsonl_path
    results["per_function_stats"] = per_function_stats

    # -- 단계 2: 합성 ----------------------------------------------------------
    t0 = time.time()
    analyzer = FeedbackDiffAnalyzer()
    diffs = [analyzer.analyze(pair) for pair in _FEEDBACK_PAIRS]

    synthesizer = RuleSetSynthesizer()
    rules = synthesizer.synthesize(diffs)

    stage_times["합성"] = time.time() - t0
    results["diffs"] = diffs
    results["rules"] = rules

    # -- 단계 3: 준비 (훅 스크립트 생성) ----------------------------------------
    t0 = time.time()
    generator = HookScriptGenerator()
    hook_script_path = generator.generate(rules, output_dir)

    with open(hook_script_path, "r", encoding="utf-8") as fh:
        hook_source = fh.read()

    stage_times["준비"] = time.time() - t0
    results["hook_script_path"] = hook_script_path
    results["hook_source"] = hook_source

    # -- 비용 추정 (Cost Guard) ------------------------------------------------
    try:
        token_count = count_tokens_in_jsonl(jsonl_path, "gpt-4o-mini-2024-07-18")
        cost_1_epoch = estimate_training_cost(token_count, "gpt-4o-mini-2024-07-18", n_epochs=1)
        cost_3_epoch = estimate_training_cost(token_count, "gpt-4o-mini-2024-07-18", n_epochs=3)
    except Exception:
        token_count = 0
        cost_1_epoch = 0.0
        cost_3_epoch = 0.0

    results["token_count"] = token_count
    results["cost_1_epoch"] = cost_1_epoch
    results["cost_3_epoch"] = cost_3_epoch

    # -- 단계 4: 학습 (모의) ---------------------------------------------------
    t0 = time.time()
    time.sleep(0.15)
    stage_times["학습"] = time.time() - t0
    results["mock_model"] = "ft:gpt-4o-mini-2024-07-18:org::demo"
    results["mock_job_id"] = "ftjob-demo456"
    results["mock_file_id"] = "file-demo123"

    results["stage_times"] = stage_times
    results["total_time"] = sum(stage_times.values())

    return results


# ---------------------------------------------------------------------------
# UI: 사이드바
# ---------------------------------------------------------------------------

st.sidebar.title("설정")
st.sidebar.markdown("---")

# -- 데이터 소스 모드 --------------------------------------------------------
st.sidebar.subheader("데이터 소스")
data_source = st.sidebar.radio(
    "데이터 소스 선택",
    ["Mock (시뮬레이션)", "Real VectorWave (Weaviate)"],
    index=0,
    help="Real 모드는 로컬 Weaviate (localhost:8080)에 연결합니다.",
)
use_real_vectorwave = data_source.startswith("Real")

ALL_FUNCTIONS = [
    "generate_review_summary",
    "extract_sentiment",
    "generate_product_recommendation",
    "translate_review",
]

if use_real_vectorwave:
    function_name = st.sidebar.selectbox(
        "분석 대상 함수",
        ["전체 (All)"] + ALL_FUNCTIONS,
        index=0,
        help="VectorWave에 기록된 실행 로그에서 선택합니다. demo_real.py를 먼저 실행하세요.",
    )
else:
    function_name = "generate_review_summary"

st.sidebar.markdown("---")

# -- OpenAI API 키 ----------------------------------------------------------
st.sidebar.subheader("OpenAI API")
api_key_input = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="sk-...",
    help="실제 파인튜닝 실행에 필요합니다. 입력하지 않으면 모의(Mock) 모드로 동작합니다.",
)
max_budget = st.sidebar.number_input(
    "최대 예산 (USD)",
    min_value=0.0,
    max_value=1000.0,
    value=10.0,
    step=1.0,
    help="파인튜닝 예상 비용이 이 금액을 초과하면 자동 차단됩니다.",
)

st.sidebar.markdown("---")

epsilon = st.sidebar.slider(
    "Epsilon (이웃 임계값)",
    min_value=0.1,
    max_value=0.5,
    value=0.3,
    step=0.01,
    help="유사도 그래프의 희소성을 제어합니다. 값이 낮을수록 더 엄격한 그래프가 생성됩니다.",
)

top_k = st.sidebar.slider(
    "Top K (전략별 후보 수)",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    help="각 선별 전략에서 반환할 최대 골든 데이터 후보 수입니다.",
)

st.sidebar.markdown("---")

run_clicked = st.sidebar.button(
    "파이프라인 실행",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    spinner_msg = (
        f"VectorTrainer 파이프라인 실행 중 ({'Real VectorWave' if use_real_vectorwave else 'Mock'})..."
    )
    with st.spinner(spinner_msg):
        st.session_state["pipeline_results"] = _run_pipeline(
            epsilon,
            top_k,
            use_real=use_real_vectorwave,
            function_name=function_name,
        )
        st.session_state["epsilon"] = epsilon
        st.session_state["top_k"] = top_k
        st.session_state["function_name"] = function_name
        st.session_state["use_real"] = use_real_vectorwave

# 사이드바에 현재 파라미터 표시
if "pipeline_results" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**마지막 실행 파라미터:**")
    st.sidebar.write(f"- epsilon = {st.session_state.get('epsilon', 'N/A')}")
    st.sidebar.write(f"- top_k = {st.session_state.get('top_k', 'N/A')}")


# ---------------------------------------------------------------------------
# UI: 제목
# ---------------------------------------------------------------------------

st.title("VectorTrainer 대시보드")
st.caption(
    "Self-Evolving AI Loop -- 밀도 기반 골든 데이터 추출, "
    "피드백 합성, 파인튜닝 파이프라인 모니터"
)

st.markdown("---")


# ---------------------------------------------------------------------------
# UI: 탭
# ---------------------------------------------------------------------------

tab_pipeline, tab_golden, tab_rules, tab_hook, tab_finetune = st.tabs(
    ["파이프라인", "골든 데이터", "규칙", "훅 스크립트", "파인튜닝"]
)

# ============================== 탭 1: 파이프라인 ==============================
with tab_pipeline:
    if "pipeline_results" not in st.session_state:
        st.info(
            "아직 파이프라인 결과가 없습니다. 사이드바에서 파라미터를 설정하고 "
            "**파이프라인 실행** 버튼을 클릭하세요."
        )
    else:
        res = st.session_state["pipeline_results"]
        stage_times: Dict[str, float] = res["stage_times"]

        # -- 전체 지표 ---------------------------------------------------------
        st.subheader("전체 지표")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 소요 시간", f"{res['total_time']:.3f}초")
        m2.metric("완료된 단계", "4 / 4")
        m3.metric(
            "골든 후보 수",
            len(res["steady_candidates"]) + len(res["anomaly_candidates"]),
        )
        m4.metric("합성된 규칙 수", len(res["rules"]))

        # -- 비용 추정 지표 ----------------------------------------------------
        st.subheader("비용 추정 (Cost Guard)")
        c1, c2, c3 = st.columns(3)
        c1.metric("학습 토큰 수", f"{res.get('token_count', 0):,}")
        c2.metric("예상 비용 (1 epoch)", f"${res.get('cost_1_epoch', 0):.6f}")
        c3.metric("예상 비용 (3 epochs)", f"${res.get('cost_3_epoch', 0):.6f}")

        st.markdown("---")

        # -- 단계별 진행 상황 ---------------------------------------------------
        st.subheader("파이프라인 단계")

        stages_info = [
            ("추출", "벡터 그래프 기반 밀도 분석으로 골든 데이터 선별"),
            ("합성", "피드백 차이 분석 및 규칙 합성"),
            ("준비", "훅 스크립트 생성 및 데이터 업로드"),
            ("학습", "파인튜닝 작업 실행"),
        ]

        # 시각적 단계 흐름
        cols = st.columns(len(stages_info))
        for idx, (stage_name, stage_desc) in enumerate(stages_info):
            with cols[idx]:
                elapsed = stage_times.get(stage_name, 0.0)
                st.success(f"**{stage_name}**")
                st.metric("소요 시간", f"{elapsed:.3f}초")
                st.caption(stage_desc)

        st.markdown("---")

        # -- 단계별 소요 시간 차트 ---------------------------------------------
        st.subheader("단계별 소요 시간")
        timing_df = pd.DataFrame(
            {
                "단계": list(stage_times.keys()),
                "시간 (초)": list(stage_times.values()),
            }
        )
        st.bar_chart(timing_df, x="단계", y="시간 (초)", horizontal=False)

        st.markdown("---")

        # -- 함수별 추출 통계 (전체 모드) -----------------------------------------
        per_func = res.get("per_function_stats", [])
        if len(per_func) > 1:
            st.subheader("함수별 추출 통계")
            func_df = pd.DataFrame([
                {
                    "함수": s["function"],
                    "Steady": s["steady"],
                    "Anomaly": s["anomaly"],
                    "합계": s["steady"] + s["anomaly"],
                }
                for s in per_func
            ])
            st.dataframe(func_df, use_container_width=True, hide_index=True)
            st.bar_chart(func_df, x="함수", y=["Steady", "Anomaly"])
            st.markdown("---")

        # -- 출력 산출물 -------------------------------------------------------
        st.subheader("출력 산출물")
        mode_label = "Real VectorWave" if st.session_state.get("use_real") else "Mock"
        st.write(f"**데이터 소스:** `{mode_label}`")
        st.write(f"**분석 함수:** `{st.session_state.get('function_name', 'N/A')}`")
        st.write(f"**출력 디렉토리:** `{res['output_dir']}`")
        st.write(f"**골든 데이터 JSONL:** `{res['jsonl_path']}`")
        st.write(f"**훅 스크립트:** `{res['hook_script_path']}`")
        st.write(f"**파인튜닝 모델 (모의):** `{res['mock_model']}`")
        st.write(f"**작업 ID (모의):** `{res['mock_job_id']}`")


# ============================== 탭 2: 골든 데이터 =============================
with tab_golden:
    if "pipeline_results" not in st.session_state:
        st.info("골든 데이터 결과를 보려면 먼저 파이프라인을 실행하세요.")
    else:
        res = st.session_state["pipeline_results"]
        steady: List[GoldenCandidate] = res["steady_candidates"]
        anomalies: List[GoldenCandidate] = res["anomaly_candidates"]

        # -- 개수 요약 ---------------------------------------------------------
        st.subheader("선별 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("정상 상태 (Steady)", len(steady))
        c2.metric("이상치 (Anomaly)", len(anomalies))
        c3.metric("전체 (중복 제거)", len(steady) + len(anomalies))

        st.markdown("---")

        # -- 통합 데이터프레임 --------------------------------------------------
        st.subheader("추출된 골든 데이터")

        all_candidates = steady + anomalies
        if all_candidates:
            df_data = []
            for c in all_candidates:
                df_data.append(
                    {
                        "UUID": c.uuid,
                        "전략": c.strategy.value,
                        "밀도": round(c.density, 4),
                        "중심까지 거리": round(c.distance_to_centroid, 4),
                        "반환값": c.return_value[:80] + (
                            "..." if len(c.return_value) > 80 else ""
                        ),
                    }
                )
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("선별된 골든 후보가 없습니다.")

        st.markdown("---")

        # -- 밀도 분포 차트 ----------------------------------------------------
        st.subheader("밀도 분포")

        if all_candidates:
            density_data = []
            for c in all_candidates:
                density_data.append(
                    {"밀도": c.density, "전략": c.strategy.value}
                )
            density_df = pd.DataFrame(density_data)

            steady_densities = [c.density for c in steady]
            anomaly_densities = [c.density for c in anomalies]

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
                        "밀도 구간": bin_labels,
                        "정상 상태": steady_counts,
                        "이상치": anomaly_counts,
                    }
                )
                st.bar_chart(hist_df, x="밀도 구간", y=["정상 상태", "이상치"])

        st.markdown("---")

        # -- JSONL 다운로드 & 미리보기 ------------------------------------------
        st.subheader("ChatML JSONL 다운로드 & 미리보기")

        jsonl_path = res["jsonl_path"]
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                jsonl_content = fh.read()
            lines = jsonl_content.strip().split("\n") if jsonl_content.strip() else []

            if lines:
                # -- 다운로드 버튼 -----------------------------------------------
                dl_col1, dl_col2 = st.columns([1, 2])
                with dl_col1:
                    fname = st.session_state.get("function_name", "golden")
                    st.download_button(
                        label=f"JSONL 다운로드 ({len(lines)}건)",
                        data=jsonl_content,
                        file_name=f"{fname}_golden_data.jsonl",
                        mime="application/jsonl",
                        type="primary",
                        use_container_width=True,
                    )
                with dl_col2:
                    st.caption(
                        f"전체 {len(lines)}개 레코드 | "
                        f"파일: `{os.path.basename(jsonl_path)}`"
                    )

                st.markdown("---")

                # -- 미리보기 ---------------------------------------------------
                preview_count = min(5, len(lines))
                st.caption(
                    f"처음 {preview_count}개 레코드 미리보기"
                )
                for i, line in enumerate(lines[:preview_count]):
                    with st.expander(f"레코드 {i + 1}", expanded=(i == 0)):
                        record = json.loads(line)
                        st.json(record)
            else:
                st.warning("JSONL 파일이 비어 있습니다.")
        except FileNotFoundError:
            st.error(f"JSONL 파일을 찾을 수 없습니다: {jsonl_path}")


# ============================== 탭 3: 규칙 ====================================
with tab_rules:
    if "pipeline_results" not in st.session_state:
        st.info("합성된 규칙을 보려면 먼저 파이프라인을 실행하세요.")
    else:
        res = st.session_state["pipeline_results"]
        rules = res["rules"]
        diffs = res["diffs"]

        st.subheader("합성된 규칙")

        if rules:
            # -- 규칙 테이블 ---------------------------------------------------
            rules_data = []
            for r in rules:
                rules_data.append(
                    {
                        "규칙 ID": r.rule_id,
                        "우선순위": r.priority,
                        "설명": r.description[:100] + (
                            "..." if len(r.description) > 100 else ""
                        ),
                        "출처 쌍": r.source_pair_index,
                    }
                )
            rules_df = pd.DataFrame(rules_data)
            st.dataframe(rules_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # -- 규칙 상세 정보 (확장 가능 섹션) --------------------------------
            st.subheader("규칙 상세 정보")

            for r in rules:
                with st.expander(f"{r.rule_id} -- 우선순위 {r.priority}"):
                    st.markdown(f"**설명:** {r.description}")
                    st.markdown("**조건:**")
                    st.code(r.condition, language="text")
                    st.markdown("**동작:**")
                    st.code(r.action, language="text")

                    if 0 <= r.source_pair_index < len(_FEEDBACK_PAIRS):
                        pair = _FEEDBACK_PAIRS[r.source_pair_index]
                        st.markdown("---")
                        st.markdown("**출처 피드백 쌍:**")
                        st.markdown(f"- **프롬프트:** {pair.input_prompt}")
                        st.markdown(f"- **잘못된 출력:** {pair.bad_output}")
                        st.markdown(f"- **수정된 출력:** {pair.fixed_output}")
        else:
            st.warning("합성된 규칙이 없습니다.")

        st.markdown("---")

        # -- 차이 분석 ---------------------------------------------------------
        st.subheader("피드백 차이 분석")

        if diffs:
            for i, diff in enumerate(diffs):
                with st.expander(
                    f"차이 #{i + 1} -- 편집 거리: {diff['edit_distance']}, "
                    f"유사도: {diff['similarity_score']:.3f}",
                    expanded=False,
                ):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**잘못된 출력:**")
                        st.code(diff["bad_output"], language="text")
                    with d2:
                        st.markdown("**수정된 출력:**")
                        st.code(diff["fixed_output"], language="text")

                    st.markdown(f"**요약:** {diff['diff_summary']}")

                    if diff["diff_segments"]:
                        st.markdown("**변경 세그먼트:**")
                        for seg in diff["diff_segments"]:
                            seg_type = seg["type"]
                            if seg_type == "added":
                                st.markdown(
                                    f"- :green[**+** 추가:] `{seg['fixed'][:80]}`"
                                )
                            elif seg_type == "removed":
                                st.markdown(
                                    f"- :red[**-** 삭제:] `{seg['bad'][:80]}`"
                                )
                            elif seg_type == "changed":
                                st.markdown(
                                    f"- :orange[**~** 변경:] "
                                    f"`{seg['bad'][:40]}` -> `{seg['fixed'][:40]}`"
                                )


# ============================== 탭 4: 훅 스크립트 =============================
with tab_hook:
    if "pipeline_results" not in st.session_state:
        st.info("생성된 훅 스크립트를 보려면 먼저 파이프라인을 실행하세요.")
    else:
        res = st.session_state["pipeline_results"]
        hook_source = res["hook_source"]

        st.subheader("생성된 훅 스크립트")
        st.caption(
            f"파일: `{res['hook_script_path']}`"
        )

        st.code(hook_source, language="python", line_numbers=True)

        st.markdown("---")

        # -- 버전 관리 ---------------------------------------------------------
        st.subheader("훅 스크립트 버전 관리")

        output_dir = res.get("output_dir", "")
        if output_dir:
            vm = HookVersionManager(output_dir)
            versions = vm.list_versions()

            if versions:
                ver_data = []
                for v in versions:
                    ver_data.append({
                        "버전": v["version_id"],
                        "타임스탬프": v["timestamp"],
                        "규칙 수": v["rules_count"],
                        "활성": "현재" if v["active"] else "",
                    })
                ver_df = pd.DataFrame(ver_data)
                st.dataframe(ver_df, use_container_width=True, hide_index=True)

                # Rollback UI
                version_ids = [v["version_id"] for v in versions]
                active_version = vm.get_active_version()
                active_id = active_version["version_id"] if active_version else None

                rollback_target = st.selectbox(
                    "롤백 대상 버전 선택",
                    version_ids,
                    index=0,
                    help="선택한 버전으로 활성 훅 스크립트를 복원합니다.",
                )

                if st.button("롤백 실행", type="secondary"):
                    try:
                        rolled = vm.rollback(rollback_target)
                        st.success(f"버전 {rolled}으로 롤백 완료!")
                        # Reload the hook source
                        hook_path = res["hook_script_path"]
                        with open(hook_path, "r", encoding="utf-8") as fh:
                            st.session_state["pipeline_results"]["hook_source"] = fh.read()
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            else:
                st.info("저장된 버전이 없습니다.")

        st.markdown("---")

        # -- 동작 설명 ---------------------------------------------------------
        st.subheader("훅 스크립트 동작 방식")

        st.markdown(
            """
생성된 훅 스크립트는 두 개의 공개 함수를 제공합니다:

1. **`intercept_prompt(user_input, context)`** -- 사용자 프롬프트가 모델에
   전달되기 전에 전처리합니다. 모든 매칭되는 규칙을 우선순위 내림차순으로
   정렬하여 입력 텍스트에 적용합니다.

2. **`intercept_output(model_output, context)`** -- 모델 출력이 사용자에게
   반환되기 전에 후처리합니다. 동일한 규칙 매칭 로직을 생성된 텍스트에
   적용합니다.

**규칙 매칭**은 단순 부분 문자열 감지를 사용합니다: 각 규칙의 `condition`
내부에 있는 인용된 문자열을 텍스트와 대조합니다. `action` 필드는 세 가지
연산을 지원합니다:

- `Replace "X" with "Y"` -- 치환
- `Remove: "X"` -- 삭제
- `Add: "X"` -- 텍스트 추가
"""
        )


# ============================== 탭 5: 파인튜닝 ================================
with tab_finetune:
    st.subheader("실제 OpenAI 파인튜닝")

    if not api_key_input:
        st.warning(
            "사이드바에서 **OpenAI API Key**를 입력해야 실제 파인튜닝을 실행할 수 있습니다."
        )
    elif "pipeline_results" not in st.session_state:
        st.info(
            "먼저 **파이프라인 실행** 버튼을 클릭하여 골든 데이터 JSONL을 생성하세요."
        )
    else:
        res = st.session_state["pipeline_results"]
        jsonl_path = res["jsonl_path"]

        st.markdown(f"**학습 데이터:** `{jsonl_path}`")

        # -- 비용 사전 검증 ----------------------------------------------------
        st.markdown("---")
        st.subheader("비용 사전 검증 (Cost Guard)")

        ft_model = st.selectbox(
            "파인튜닝 모델",
            ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-3.5-turbo-0125"],
            index=0,
        )
        n_epochs = st.slider("Epochs", min_value=1, max_value=10, value=3)

        try:
            token_count = count_tokens_in_jsonl(jsonl_path, ft_model)
            cost_est = estimate_training_cost(token_count, ft_model, n_epochs)

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("토큰 수", f"{token_count:,}")
            cc2.metric("예상 비용", f"${cost_est:.6f}")
            cc3.metric("예산", f"${max_budget:.2f}")

            if cost_est > max_budget:
                st.error(
                    f"예상 비용 ${cost_est:.6f}이 예산 ${max_budget:.2f}을 "
                    f"초과합니다. 예산을 늘리거나 데이터를 줄이세요."
                )
                budget_ok = False
            else:
                st.success(
                    f"예산 내 (${cost_est:.6f} / ${max_budget:.2f})"
                )
                budget_ok = True
        except Exception as e:
            st.error(f"비용 계산 오류: {e}")
            budget_ok = False

        st.markdown("---")

        # -- 파인튜닝 실행 / 상태 조회 -----------------------------------------
        col_upload, col_status = st.columns(2)

        with col_upload:
            st.subheader("파인튜닝 실행")

            start_disabled = not budget_ok
            if st.button(
                "파일 업로드 & 학습 시작",
                type="primary",
                disabled=start_disabled,
                use_container_width=True,
            ):
                try:
                    trainer = OpenAITrainer(
                        api_key=api_key_input,
                        model=ft_model,
                        max_budget_usd=max_budget,
                    )

                    with st.spinner("JSONL 파일 업로드 중..."):
                        file_id = trainer.prepare_data(jsonl_path)
                    st.success(f"업로드 완료: `{file_id}`")
                    st.session_state["ft_file_id"] = file_id

                    with st.spinner("파인튜닝 작업 생성 중..."):
                        job_id = trainer.start_training(
                            file_id,
                            hyperparameters={"n_epochs": n_epochs},
                        )
                    st.success(f"학습 시작: `{job_id}`")
                    st.session_state["ft_job_id"] = job_id
                    st.session_state["ft_trainer"] = trainer

                except BudgetExceededError as e:
                    st.error(f"예산 초과로 차단: {e}")
                except Exception as e:
                    st.error(f"오류 발생: {e}")

            # 이전 실행 결과 표시
            if "ft_file_id" in st.session_state:
                st.info(f"마지막 file_id: `{st.session_state['ft_file_id']}`")
            if "ft_job_id" in st.session_state:
                st.info(f"마지막 job_id: `{st.session_state['ft_job_id']}`")

        with col_status:
            st.subheader("학습 상태 조회")

            job_id_input = st.text_input(
                "Job ID",
                value=st.session_state.get("ft_job_id", ""),
                placeholder="ftjob-...",
            )

            if st.button("상태 확인", use_container_width=True) and job_id_input:
                try:
                    trainer = st.session_state.get("ft_trainer")
                    if trainer is None:
                        trainer = OpenAITrainer(api_key=api_key_input, model=ft_model)

                    status = trainer.get_status(job_id_input)
                    st.json(status)

                    if status.get("status") == "succeeded":
                        st.balloons()
                        result = trainer.get_result(job_id_input)
                        st.success(
                            f"학습 완료! 모델: `{result.get('fine_tuned_model')}`"
                        )
                except Exception as e:
                    st.error(f"상태 조회 오류: {e}")
