"""
VectorTrainer 전체 파이프라인 데모.

실제 사용 패턴을 재현한다:
1. Mock DatasetManager로 VectorWave 로그 시뮬레이션
2. DensityBasedExtractor로 Golden Data 선별
3. FeedbackPair 기반 규칙 합성 + Hook Script 생성
4. OpenAITrainer(Mock)로 파인튜닝
5. CLIDashboard로 전체 과정 모니터링
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
from unittest.mock import MagicMock, PropertyMock

# --- 1. Mock VectorWave DatasetManager ---

def create_mock_dataset_manager():
    """VectorWave가 없어도 동작하도록 Mock DatasetManager를 구성."""

    # 실제 실행 로그를 시뮬레이션하는 벡터 데이터
    # 클러스터 A: 고밀도 (유사한 응답들) — 리뷰 요약 태스크
    cluster_a = [
        {"uuid": f"log-a{i}", "function_name": "generate_review_summary",
         "return_value": f"이 제품은 품질이 우수하고 배송이 빠릅니다. 평점: {4 + (i % 2)}/5",
         "status": "SUCCESS",
         "vector": list(np.random.normal(loc=[0.8, 0.6, 0.7, 0.5], scale=0.05, size=4))}
        for i in range(15)
    ]

    # 클러스터 B: 중밀도 — 다른 유형의 리뷰
    cluster_b = [
        {"uuid": f"log-b{i}", "function_name": "generate_review_summary",
         "return_value": f"사용 경험이 보통입니다. 개선이 필요한 부분이 있습니다. 평점: {2 + (i % 2)}/5",
         "status": "SUCCESS",
         "vector": list(np.random.normal(loc=[0.3, 0.8, 0.2, 0.6], scale=0.05, size=4))}
        for i in range(8)
    ]

    # Anomaly: 저밀도지만 SUCCESS — 특이한 성공 케이스
    anomalies = [
        {"uuid": "log-anomaly-1", "function_name": "generate_review_summary",
         "return_value": "이 제품은 예술 작품입니다. 기능을 넘어선 감성적 가치가 있습니다.",
         "status": "SUCCESS",
         "vector": [0.1, 0.1, 0.9, 0.1]},
        {"uuid": "log-anomaly-2", "function_name": "generate_review_summary",
         "return_value": "기술적 스펙은 경쟁사 대비 열위이나, UX 설계가 이를 완전히 상쇄합니다.",
         "status": "SUCCESS",
         "vector": [0.9, 0.1, 0.1, 0.9]},
    ]

    # FAILURE 로그 (선별에서 제외되어야 함)
    failures = [
        {"uuid": "log-fail-1", "function_name": "generate_review_summary",
         "return_value": "Error: context too long",
         "status": "FAILURE",
         "vector": [0.5, 0.5, 0.5, 0.5]},
    ]

    all_logs = cluster_a + cluster_b + anomalies + failures

    # Weaviate 응답 객체 시뮬레이션
    class MockWeaviateObject:
        def __init__(self, data):
            self.uuid = data["uuid"]
            self.properties = {
                "function_name": data["function_name"],
                "return_value": data["return_value"],
                "status": data["status"],
            }
            self.vector = {"default": data["vector"]}

    mock_objects = [MockWeaviateObject(log) for log in all_logs]

    class MockFetchResponse:
        def __init__(self, objects):
            self.objects = objects

    # Mock DatasetManager 구성
    manager = MagicMock()

    # exec_col.query.fetch_objects 설정
    exec_query = MagicMock()
    exec_query.fetch_objects.return_value = MockFetchResponse(mock_objects)
    exec_col = MagicMock()
    type(exec_col).query = PropertyMock(return_value=exec_query)
    type(manager).exec_col = PropertyMock(return_value=exec_col)

    # golden_col 설정
    golden_query = MagicMock()
    golden_query.fetch_objects.return_value = MockFetchResponse([])
    golden_col = MagicMock()
    type(golden_col).query = PropertyMock(return_value=golden_query)
    type(manager).golden_col = PropertyMock(return_value=golden_col)

    return manager


# --- 2. 실제 사용 패턴 ---

def main():
    from vector_trainer.extractor import DensityBasedExtractor
    from vector_trainer.synthesizer import (
        FeedbackDiffAnalyzer,
        RuleSetSynthesizer,
        HookScriptGenerator,
        run_synthesis_pipeline,
    )
    from vector_trainer.pipeline import OpenAITrainer, TrainingPipeline
    from vector_trainer.dashboard import CLIDashboard
    from vector_trainer.types import FeedbackPair

    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(
        "[bold]VectorTrainer Demo[/bold]\n"
        "Self-Evolving AI Loop 전체 파이프라인 데모\n\n"
        "Mock DatasetManager로 VectorWave 로그를 시뮬레이션합니다.",
        title="VectorTrainer v0.1.0",
        border_style="blue",
    ))

    # 출력 디렉토리 설정
    output_dir = tempfile.mkdtemp(prefix="vectortrainer_demo_")
    console.print(f"\n[dim]Output directory: {output_dir}[/dim]\n")

    # --- Step 1: Golden Data 선별 ---
    console.print("[bold cyan]== Step 1: Golden Data 선별 ==[/bold cyan]\n")

    dataset_manager = create_mock_dataset_manager()
    extractor = DensityBasedExtractor(
        dataset_manager=dataset_manager,
        epsilon=0.3,
        top_k=10,
    )

    # Steady State 선별
    steady = extractor.select_steady_state("generate_review_summary")
    console.print(f"  Steady State 후보: [green]{len(steady)}개[/green]")
    for c in steady[:3]:
        console.print(f"    - {c.uuid} | density={c.density:.4f} | {c.return_value[:50]}...")

    # Anomaly 선별
    anomalies = extractor.select_anomalies("generate_review_summary")
    console.print(f"\n  Anomaly 후보: [yellow]{len(anomalies)}개[/yellow]")
    for c in anomalies[:3]:
        console.print(f"    - {c.uuid} | density={c.density:.4f} | {c.return_value[:50]}...")

    # JSONL 출력
    jsonl_path = os.path.join(output_dir, "golden_data.jsonl")
    extractor.extract_golden_data("generate_review_summary", jsonl_path)
    with open(jsonl_path) as f:
        line_count = sum(1 for _ in f)
    console.print(f"\n  [green]Golden Data JSONL: {line_count}개 레코드 → {jsonl_path}[/green]")

    # ChatML 포맷 샘플 출력
    with open(jsonl_path) as f:
        sample = json.loads(f.readline())
    console.print(f"\n  [dim]ChatML 샘플:[/dim]")
    console.print_json(json.dumps(sample, ensure_ascii=False, indent=2))

    # --- Step 2: 프롬프트 합성 ---
    console.print("\n[bold magenta]== Step 2: 프롬프트 합성 ==[/bold magenta]\n")

    feedback_pairs = [
        FeedbackPair(
            input_prompt="이 제품의 리뷰를 요약해주세요",
            bad_output="좋은 제품입니다",
            fixed_output="이 제품은 품질이 우수하며, 특히 내구성과 디자인 면에서 높은 평가를 받고 있습니다. 배송 속도도 빠릅니다.",
            context={"function": "generate_review_summary"},
        ),
        FeedbackPair(
            input_prompt="사용자 리뷰를 분석해주세요",
            bad_output="별로입니다",
            fixed_output="사용자 리뷰 분석 결과: 긍정 요소(디자인, 가격)와 개선 필요 요소(AS, 설명서)가 혼재합니다. 전반적 만족도는 3.8/5입니다.",
            context={"function": "generate_review_summary"},
        ),
        FeedbackPair(
            input_prompt="이 리뷰의 핵심 포인트를 정리해주세요",
            bad_output="핵심 포인트: 좋음",
            fixed_output="핵심 포인트: 1) 가격 대비 성능이 우수 2) 배터리 수명이 경쟁 제품 대비 30% 향상 3) A/S 대응 속도 개선 필요",
            context={"function": "generate_review_summary"},
        ),
    ]

    # 피드백 분석
    analyzer = FeedbackDiffAnalyzer()
    for i, pair in enumerate(feedback_pairs):
        diff = analyzer.analyze(pair)
        console.print(f"  Pair {i+1}: edit_distance={diff['edit_distance']}, similarity={diff['similarity_score']:.3f}")
        console.print(f"    Bad:   \"{pair.bad_output}\"")
        console.print(f"    Fixed: \"{pair.fixed_output[:60]}...\"")

    # Hook Script 생성
    hook_path = run_synthesis_pipeline(
        pairs=feedback_pairs,
        output_dir=output_dir,
    )
    console.print(f"\n  [green]Hook Script 생성됨: {hook_path}[/green]")

    # 생성된 Hook Script 내용 미리보기
    with open(hook_path) as f:
        hook_content = f.read()
    console.print(f"\n  [dim]Hook Script 미리보기 (첫 30줄):[/dim]")
    for line in hook_content.split("\n")[:30]:
        console.print(f"  [dim]{line}[/dim]")

    # --- Step 3: 파이프라인 통합 실행 (with Dashboard) ---
    console.print("\n[bold green]== Step 3: 전체 파이프라인 + 대시보드 ==[/bold green]\n")

    # Mock OpenAI Trainer
    trainer = OpenAITrainer(api_key="demo-key")
    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.id = "file-demo123"
    mock_client.files.create.return_value = mock_file
    mock_job = MagicMock()
    mock_job.id = "ftjob-demo456"
    mock_client.fine_tuning.jobs.create.return_value = mock_job
    mock_status = MagicMock()
    mock_status.status = "succeeded"
    mock_status.trained_tokens = 15420
    mock_status.created_at = 1700000000
    mock_status.finished_at = 1700003600
    mock_status.error = None
    mock_status.fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:cozymori::demo"
    mock_status.result_files = ["file-result-789"]
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_status
    trainer._client = mock_client

    # Dashboard 연결
    dashboard = CLIDashboard(console=console)

    # 파이프라인 실행
    pipeline = TrainingPipeline(
        extractor=extractor,
        synthesizer_output_dir=os.path.join(output_dir, "hooks"),
        trainer=trainer,
        monitor=dashboard,
    )

    result = pipeline.run(
        function_name="generate_review_summary",
        feedback_pairs=feedback_pairs,
        output_dir=output_dir,
    )

    # 최종 요약
    dashboard.summary()

    # 결과 출력
    console.print("\n[bold]== 최종 결과 ==[/bold]\n")
    console.print(f"  Golden Data JSONL: {jsonl_path}")
    console.print(f"  Hook Script:       {hook_path}")
    console.print(f"  Fine-tuned Model:  {result.get('train', {}).get('fine_tuned_model', 'N/A')}")
    console.print(f"\n  [dim]Output directory: {output_dir}[/dim]")


if __name__ == "__main__":
    main()
