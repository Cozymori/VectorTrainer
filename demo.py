"""
VectorTrainer Full Pipeline Demo (Real VectorWave Integration).

Uses the REAL VectorWave library connected to Weaviate (localhost:8080)
instead of mocks:

1. Initialize VectorWave DB and populate execution logs via @vectorize
2. DensityBasedExtractor with real VectorWaveDatasetManager for Golden Data
3. FeedbackPair-based rule synthesis + Hook Script generation
4. OpenAITrainer (MOCKED -- no actual OpenAI API calls) for fine-tuning
5. CLIDashboard for real-time pipeline monitoring

Falls back to mock mode if Weaviate is unreachable.
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
from unittest.mock import MagicMock, PropertyMock

from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

VECTORWAVE_AVAILABLE = False
USE_MOCK_FALLBACK = False


def _check_openai_key() -> bool:
    """Check if OPENAI_API_KEY is set (needed by text2vec-openai vectorizer)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        console.print(
            "[bold yellow]WARNING:[/bold yellow] OPENAI_API_KEY environment variable is not set.\n"
            "  The @vectorize decorator relies on text2vec-openai for embedding.\n"
            "  Set it with:\n"
            "    export OPENAI_API_KEY='sk-...'\n"
            "  Falling back to mock mode.\n",
        )
        return False
    return True


def _try_initialize_vectorwave():
    """Attempt to connect to Weaviate via VectorWave. Returns the client or None."""
    global VECTORWAVE_AVAILABLE
    try:
        from vectorwave import initialize_database

        client = initialize_database()
        if client:
            VECTORWAVE_AVAILABLE = True
            return client
        else:
            console.print(
                "[bold yellow]WARNING:[/bold yellow] VectorWave initialize_database() "
                "returned None. Is Weaviate running at localhost:8080?\n"
                "  Start it with: docker compose up -d\n"
                "  Falling back to mock mode.\n",
            )
            return None
    except Exception as exc:
        console.print(
            f"[bold yellow]WARNING:[/bold yellow] VectorWave/Weaviate connection failed: {exc}\n"
            "  Make sure Weaviate is running at localhost:8080.\n"
            "  Falling back to mock mode.\n",
        )
        return None


# ---------------------------------------------------------------------------
# Mock DatasetManager (fallback when VectorWave is unavailable)
# ---------------------------------------------------------------------------

def create_mock_dataset_manager():
    """Build a mock DatasetManager that simulates VectorWave execution logs."""

    # Cluster A: high density -- review summaries (similar responses)
    cluster_a = [
        {
            "uuid": f"log-a{i}",
            "function_name": "generate_review_summary",
            "return_value": f"This product has excellent quality and fast delivery. Rating: {4 + (i % 2)}/5",
            "status": "SUCCESS",
            "vector": list(np.random.normal(loc=[0.8, 0.6, 0.7, 0.5], scale=0.05, size=4)),
        }
        for i in range(15)
    ]

    # Cluster B: medium density -- different review type
    cluster_b = [
        {
            "uuid": f"log-b{i}",
            "function_name": "generate_review_summary",
            "return_value": f"Usage experience is average. There are areas that need improvement. Rating: {2 + (i % 2)}/5",
            "status": "SUCCESS",
            "vector": list(np.random.normal(loc=[0.3, 0.8, 0.2, 0.6], scale=0.05, size=4)),
        }
        for i in range(8)
    ]

    # Anomalies: low density but SUCCESS -- unusual success cases
    anomalies = [
        {
            "uuid": "log-anomaly-1",
            "function_name": "generate_review_summary",
            "return_value": "This product is a work of art. It has emotional value beyond functionality.",
            "status": "SUCCESS",
            "vector": [0.1, 0.1, 0.9, 0.1],
        },
        {
            "uuid": "log-anomaly-2",
            "function_name": "generate_review_summary",
            "return_value": "Technical specs are inferior to competitors, but UX design completely compensates for it.",
            "status": "SUCCESS",
            "vector": [0.9, 0.1, 0.1, 0.9],
        },
    ]

    # FAILURE logs (should be excluded from selection)
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
# Sample @vectorize function
# ---------------------------------------------------------------------------

def _define_vectorized_function():
    """Define and return a @vectorize-decorated function.

    Must be called AFTER initialize_database() succeeds so that the
    decorator can register the function schema in Weaviate.
    """
    from vectorwave import vectorize

    @vectorize(capture_return_value=True, capture_inputs=True)
    def generate_review_summary(review_text: str) -> str:
        """Summarize a product review into a concise assessment."""
        # Simple heuristic summarizer for demo purposes
        positive_words = ["excellent", "great", "amazing", "fast", "good", "love", "best", "quality"]
        negative_words = ["bad", "slow", "terrible", "worst", "poor", "broken", "disappointing"]

        text_lower = review_text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            sentiment = "positive"
            rating = min(5, 3 + pos_count)
        elif neg_count > pos_count:
            sentiment = "negative"
            rating = max(1, 3 - neg_count)
        else:
            sentiment = "neutral"
            rating = 3

        summary = (
            f"Review analysis: {sentiment} sentiment. "
            f"Key points extracted from {len(review_text.split())} words. "
            f"Overall rating: {rating}/5."
        )
        return summary

    return generate_review_summary


# ---------------------------------------------------------------------------
# Sample reviews for populating execution logs
# ---------------------------------------------------------------------------

SAMPLE_REVIEWS = [
    "This product is absolutely excellent! The quality is amazing and delivery was super fast. Best purchase I've made this year.",
    "Great value for money. The build quality is good and it works as advertised. Would recommend to friends.",
    "Love this product! Amazing design and the performance is excellent. Fast shipping too.",
    "The quality is great and the customer service was helpful. Good experience overall.",
    "Excellent product with fast delivery. The quality exceeded my expectations.",
    "This is the best gadget I've ever owned. Amazing battery life and great screen quality.",
    "Good product but delivery was slow. The quality is decent for the price though.",
    "The product broke after two weeks. Terrible quality and poor customer service. Very disappointing.",
    "Average product. Nothing special but it works. Delivery was on time.",
    "Worst purchase ever. The product is bad and the return process is terrible. Do not buy.",
    "Innovative design that pushes boundaries. Not the most practical, but truly a conversation piece.",
    "The technical specifications are underwhelming compared to competitors, but the user experience design is exceptional.",
]

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main():
    from vector_trainer.extractor import DensityBasedExtractor
    from vector_trainer.synthesizer import (
        FeedbackDiffAnalyzer,
        run_synthesis_pipeline,
    )
    from vector_trainer.pipeline import OpenAITrainer, TrainingPipeline
    from vector_trainer.dashboard import CLIDashboard
    from vector_trainer.types import FeedbackPair

    # ==================================================================
    # Banner
    # ==================================================================
    has_openai_key = _check_openai_key()
    client = _try_initialize_vectorwave() if has_openai_key else None
    use_real = VECTORWAVE_AVAILABLE and client is not None

    mode_label = "[green]REAL VectorWave[/green]" if use_real else "[yellow]MOCK mode (fallback)[/yellow]"

    console.print(Panel(
        "[bold]VectorTrainer Demo[/bold]\n"
        "Self-Evolving AI Loop -- Full Pipeline Demo\n\n"
        f"Mode: {mode_label}",
        title="VectorTrainer v0.1.0",
        border_style="blue",
    ))

    # Output directory
    output_dir = tempfile.mkdtemp(prefix="vectortrainer_demo_")
    console.print(f"\n[dim]Output directory: {output_dir}[/dim]\n")

    # ==================================================================
    # Phase 1: VectorWave Setup -- populate execution logs
    # ==================================================================
    if use_real:
        console.print("[bold blue]== Phase 1: VectorWave Setup ==[/bold blue]\n")
        console.print("  Populating execution logs via @vectorize decorator...\n")

        generate_review_summary = _define_vectorized_function()

        for i, review in enumerate(SAMPLE_REVIEWS):
            try:
                result = generate_review_summary(review_text=review)
                console.print(f"  [{i+1:>2}/{len(SAMPLE_REVIEWS)}] {result[:80]}...")
            except Exception as exc:
                console.print(f"  [{i+1:>2}/{len(SAMPLE_REVIEWS)}] [red]Error: {exc}[/red]")

        wait_seconds = 4
        console.print(f"\n  Waiting {wait_seconds}s for async indexing to complete...")
        time.sleep(wait_seconds)

        # Verify logs exist
        try:
            from vectorwave import search_executions

            logs = search_executions(
                limit=5,
                filters={"function_name": "generate_review_summary"},
            )
            console.print(f"  [green]Verified: {len(logs)} execution logs found in Weaviate.[/green]")
            if logs:
                console.print(f"  [dim]Latest log UUID: {logs[0].get('uuid', 'N/A')}[/dim]")
        except Exception as exc:
            console.print(f"  [yellow]Could not verify logs: {exc}[/yellow]")

        console.print()

    # ==================================================================
    # Phase 2: Golden Data Selection
    # ==================================================================
    console.print("[bold cyan]== Phase 2: Golden Data Selection ==[/bold cyan]\n")

    if use_real:
        # Real VectorWave DatasetManager
        from vectorwave import VectorWaveDatasetManager

        try:
            dataset_manager = VectorWaveDatasetManager()
            console.print("  [green]Using real VectorWaveDatasetManager[/green]\n")
        except Exception as exc:
            console.print(
                f"  [yellow]Failed to create VectorWaveDatasetManager: {exc}[/yellow]\n"
                "  Falling back to mock DatasetManager.\n"
            )
            dataset_manager = create_mock_dataset_manager()
    else:
        dataset_manager = create_mock_dataset_manager()
        console.print("  [dim]Using mock DatasetManager (VectorWave unavailable)[/dim]\n")

    extractor = DensityBasedExtractor(
        dataset_manager=dataset_manager,
        epsilon=0.3,
        top_k=10,
    )

    # Steady State selection
    func_name = "generate_review_summary"
    steady = extractor.select_steady_state(func_name)
    console.print(f"  Steady State candidates: [green]{len(steady)}[/green]")
    for c in steady[:3]:
        rv_preview = c.return_value[:60] if c.return_value else "N/A"
        console.print(f"    - {c.uuid[:20]:20s} | density={c.density:.4f} | {rv_preview}...")

    # Anomaly selection
    anomalies = extractor.select_anomalies(func_name)
    console.print(f"\n  Anomaly candidates: [yellow]{len(anomalies)}[/yellow]")
    for c in anomalies[:3]:
        rv_preview = c.return_value[:60] if c.return_value else "N/A"
        console.print(f"    - {c.uuid[:20]:20s} | density={c.density:.4f} | {rv_preview}...")

    if not steady and not anomalies:
        console.print(
            "\n  [yellow]No candidates found. This may be the first run and "
            "execution logs have not been indexed yet.[/yellow]\n"
            "  [dim]Try running the demo again after Weaviate finishes indexing.[/dim]"
        )

    # JSONL export
    jsonl_path = os.path.join(output_dir, "golden_data.jsonl")
    extractor.extract_golden_data(func_name, jsonl_path)

    with open(jsonl_path) as f:
        line_count = sum(1 for _ in f)
    console.print(f"\n  [green]Golden Data JSONL: {line_count} records -> {jsonl_path}[/green]")

    # ChatML format sample
    if line_count > 0:
        with open(jsonl_path) as f:
            sample = json.loads(f.readline())
        console.print(f"\n  [dim]ChatML sample:[/dim]")
        console.print_json(json.dumps(sample, ensure_ascii=False, indent=2))

    # ==================================================================
    # Phase 3: Prompt Synthesis
    # ==================================================================
    console.print("\n[bold magenta]== Phase 3: Prompt Synthesis ==[/bold magenta]\n")

    feedback_pairs = [
        FeedbackPair(
            input_prompt="Please summarize this product review",
            bad_output="Good product",
            fixed_output=(
                "This product has excellent quality, especially in durability and design. "
                "It receives high ratings. Delivery speed is also fast."
            ),
            context={"function": "generate_review_summary"},
        ),
        FeedbackPair(
            input_prompt="Analyze the user reviews",
            bad_output="Not great",
            fixed_output=(
                "User review analysis: Positive factors (design, price) and areas needing "
                "improvement (after-sales service, documentation) are mixed. Overall satisfaction: 3.8/5."
            ),
            context={"function": "generate_review_summary"},
        ),
        FeedbackPair(
            input_prompt="Summarize the key points of this review",
            bad_output="Key points: good",
            fixed_output=(
                "Key points: 1) Excellent price-to-performance ratio 2) Battery life improved "
                "by 30% compared to competitors 3) After-sales response speed needs improvement"
            ),
            context={"function": "generate_review_summary"},
        ),
    ]

    # Feedback analysis
    analyzer = FeedbackDiffAnalyzer()
    for i, pair in enumerate(feedback_pairs):
        diff = analyzer.analyze(pair)
        console.print(
            f"  Pair {i+1}: edit_distance={diff['edit_distance']}, "
            f"similarity={diff['similarity_score']:.3f}"
        )
        console.print(f'    Bad:   "{pair.bad_output}"')
        fixed_preview = pair.fixed_output[:60]
        console.print(f'    Fixed: "{fixed_preview}..."')

    # Hook Script generation
    hook_path = run_synthesis_pipeline(
        pairs=feedback_pairs,
        output_dir=output_dir,
    )
    console.print(f"\n  [green]Hook Script generated: {hook_path}[/green]")

    # Preview the generated Hook Script
    with open(hook_path) as f:
        hook_content = f.read()
    console.print(f"\n  [dim]Hook Script preview (first 30 lines):[/dim]")
    for line in hook_content.split("\n")[:30]:
        console.print(f"  [dim]{line}[/dim]")

    # ==================================================================
    # Phase 4: Full Pipeline with CLIDashboard
    # ==================================================================
    console.print("\n[bold green]== Phase 4: Full Pipeline + Dashboard ==[/bold green]\n")

    # Mock OpenAI Trainer (we do NOT want to call the real OpenAI API)
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
    mock_status.fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:vectortrainer::demo"
    mock_status.result_files = ["file-result-789"]
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_status
    trainer._client = mock_client

    # Dashboard
    dashboard = CLIDashboard(console=console)

    # Pipeline
    pipeline = TrainingPipeline(
        extractor=extractor,
        synthesizer_output_dir=os.path.join(output_dir, "hooks"),
        trainer=trainer,
        monitor=dashboard,
    )

    result = pipeline.run(
        function_name=func_name,
        feedback_pairs=feedback_pairs,
        output_dir=output_dir,
    )

    # Summary
    dashboard.summary()

    # ==================================================================
    # Final results
    # ==================================================================
    console.print("\n[bold]== Final Results ==[/bold]\n")
    console.print(f"  Mode:              {mode_label}")
    console.print(f"  Golden Data JSONL: {jsonl_path}")
    console.print(f"  Hook Script:       {hook_path}")
    console.print(f"  Fine-tuned Model:  {result.get('train', {}).get('fine_tuned_model', 'N/A')}")
    console.print(f"  Job ID:            {result.get('job_id', 'N/A')}")
    console.print(f"  File ID:           {result.get('file_id', 'N/A')}")
    console.print(f"\n  [dim]Output directory: {output_dir}[/dim]")
    console.print()


if __name__ == "__main__":
    main()
