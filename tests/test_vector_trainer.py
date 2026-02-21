"""Comprehensive pytest unit tests for the VectorTrainer project.

Covers:
  1. Extractor  -- cosine similarity, graph construction, density, selection strategies, JSONL output
  2. Synthesizer -- Levenshtein, diff analysis, rule synthesis, contradictions, hook script generation
  3. Pipeline   -- abstract trainer guard, OpenAI trainer mocking, orchestration
  4. Dashboard  -- PipelineMonitor callbacks, CLIDashboard instantiation, summary structure
  5. Cost Guard -- token counting, cost estimation, budget validation
  6. Hook Versioning -- version save, rollback, list, integration
"""

from __future__ import annotations

import ast
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch, mock_open

import pytest
import numpy as np

from vector_trainer.types import (
    CostEstimate,
    ExecutionLog,
    FeedbackPair,
    GoldenCandidate,
    Rule,
    SelectionStrategy,
)
from vector_trainer.cost_guard import (
    BudgetExceededError,
    check_budget,
    count_tokens_in_jsonl,
    estimate_training_cost,
)
from vector_trainer.extractor import DensityBasedExtractor
from vector_trainer.synthesizer import (
    FeedbackDiffAnalyzer,
    HookScriptGenerator,
    HookVersionManager,
    RuleSetSynthesizer,
    run_synthesis_pipeline,
)
from vector_trainer.pipeline import BaseTrainer, OpenAITrainer, TrainingPipeline
from vector_trainer.dashboard import CLIDashboard, PipelineMonitor


# ============================================================================
# Helper fixtures and factories
# ============================================================================


def _make_weaviate_object(
    uuid: str,
    function_name: str,
    return_value: str,
    status: str,
    vector: List[float],
) -> MagicMock:
    """Create a mock Weaviate object with the expected attributes."""
    obj = MagicMock()
    obj.uuid = uuid
    obj.properties = {
        "function_name": function_name,
        "return_value": return_value,
        "status": status,
    }
    obj.vector = {"default": vector}
    return obj


def _make_fetch_response(objects: List[MagicMock]) -> MagicMock:
    """Create a mock Weaviate fetch_objects response."""
    response = MagicMock()
    response.objects = objects
    return response


@pytest.fixture
def mock_dataset_manager():
    """Create a mock DatasetManager with exec_col and golden_col."""
    dm = MagicMock()

    # exec_col.query.fetch_objects returns a mock response
    exec_query = MagicMock()
    exec_col = MagicMock()
    type(exec_col).query = PropertyMock(return_value=exec_query)
    type(dm).exec_col = PropertyMock(return_value=exec_col)

    # golden_col.query.fetch_objects
    golden_query = MagicMock()
    golden_col = MagicMock()
    type(golden_col).query = PropertyMock(return_value=golden_query)
    type(dm).golden_col = PropertyMock(return_value=golden_col)

    return dm


@pytest.fixture
def sample_logs() -> List[ExecutionLog]:
    """Provide a small set of ExecutionLog instances with known vectors."""
    return [
        ExecutionLog(
            uuid="log-1",
            function_name="my_func",
            return_value="result_1",
            status="SUCCESS",
            vector=[1.0, 0.0, 0.0],
        ),
        ExecutionLog(
            uuid="log-2",
            function_name="my_func",
            return_value="result_2",
            status="SUCCESS",
            vector=[0.9, 0.1, 0.0],  # very similar to log-1
        ),
        ExecutionLog(
            uuid="log-3",
            function_name="my_func",
            return_value="result_3",
            status="SUCCESS",
            vector=[0.0, 0.0, 1.0],  # orthogonal to log-1 and log-2
        ),
        ExecutionLog(
            uuid="log-4",
            function_name="my_func",
            return_value="result_4",
            status="FAILURE",
            vector=[0.95, 0.05, 0.0],  # similar to log-1/log-2, but FAILURE
        ),
    ]


@pytest.fixture
def sample_feedback_pairs() -> List[FeedbackPair]:
    """Provide a small set of FeedbackPair instances for synthesizer tests."""
    return [
        FeedbackPair(
            bad_output="The color is red.",
            fixed_output="The colour is blue.",
            input_prompt="What colour is the sky?",
            context={"source": "user"},
        ),
        FeedbackPair(
            bad_output="Hello World",
            fixed_output="Hello, World!",
            input_prompt="Greet the user",
            context={"source": "review"},
        ),
    ]


# ============================================================================
# 1. Extractor Tests
# ============================================================================


class TestCosineSimlarity:
    """Tests for DensityBasedExtractor._cosine_similarity."""

    def test_cosine_similarity_calculation(self):
        """Verify cosine similarity between known vectors produces correct value."""
        # Two identical unit vectors -> similarity = 1.0
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert DensityBasedExtractor._cosine_similarity(v1, v2) == pytest.approx(1.0)

        # Orthogonal vectors -> similarity = 0.0
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert DensityBasedExtractor._cosine_similarity(v1, v2) == pytest.approx(0.0)

        # Opposite vectors -> similarity = -1.0
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert DensityBasedExtractor._cosine_similarity(v1, v2) == pytest.approx(-1.0)

        # Known angle: 45 degrees between (1,0) and (1,1) -> cos(45) ~ 0.7071
        v1 = [1.0, 0.0]
        v2 = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert DensityBasedExtractor._cosine_similarity(v1, v2) == pytest.approx(
            expected, abs=1e-6
        )

    def test_cosine_similarity_zero_vector(self):
        """Zero vector should return 0.0 regardless of the other vector."""
        v_zero = [0.0, 0.0, 0.0]
        v_normal = [1.0, 2.0, 3.0]

        assert DensityBasedExtractor._cosine_similarity(v_zero, v_normal) == 0.0
        assert DensityBasedExtractor._cosine_similarity(v_normal, v_zero) == 0.0
        assert DensityBasedExtractor._cosine_similarity(v_zero, v_zero) == 0.0


class TestBuildWeightedGraph:
    """Tests for DensityBasedExtractor._build_weighted_graph."""

    def test_build_weighted_graph(self, mock_dataset_manager, sample_logs):
        """Verify graph construction: nodes present, edges respect epsilon threshold."""
        extractor = DensityBasedExtractor(mock_dataset_manager, epsilon=0.1, top_k=10)
        graph = extractor._build_weighted_graph(sample_logs)

        # All nodes should be in the adjacency dict
        assert set(graph.keys()) == {"log-1", "log-2", "log-3", "log-4"}

        # log-1 and log-2 have high cosine similarity (~0.9945) -> above 0.9 threshold
        cos_12 = DensityBasedExtractor._cosine_similarity(
            sample_logs[0].vector, sample_logs[1].vector
        )
        assert cos_12 >= 0.9  # sanity check that they pass the threshold

        # log-1 -> log-2 edge should exist
        assert "log-2" in graph["log-1"]
        assert "log-1" in graph["log-2"]

        # log-1 and log-3 are orthogonal (sim=0) -> no edge
        assert "log-3" not in graph["log-1"]
        assert "log-1" not in graph["log-3"]

        # Symmetry: if edge (u,v) exists, weight should be equal both ways
        for node_id, neighbours in graph.items():
            for neighbour_id, weight in neighbours.items():
                assert graph[neighbour_id][node_id] == pytest.approx(weight)

    def test_build_graph_empty_logs(self, mock_dataset_manager):
        """Empty log list should produce an empty graph."""
        extractor = DensityBasedExtractor(mock_dataset_manager)
        graph = extractor._build_weighted_graph([])
        assert graph == {}


class TestDensityComputation:
    """Tests for DensityBasedExtractor._compute_density."""

    def test_density_computation(self):
        """D(L_i) = sum of weights of all edges incident to L_i."""
        graph = {
            "A": {"B": 0.5, "C": 0.3},
            "B": {"A": 0.5},
            "C": {"A": 0.3},
            "D": {},
        }

        assert DensityBasedExtractor._compute_density("A", graph) == pytest.approx(0.8)
        assert DensityBasedExtractor._compute_density("B", graph) == pytest.approx(0.5)
        assert DensityBasedExtractor._compute_density("C", graph) == pytest.approx(0.3)
        assert DensityBasedExtractor._compute_density("D", graph) == pytest.approx(0.0)

    def test_density_missing_node(self):
        """Non-existent node returns 0.0."""
        graph = {"A": {"B": 0.9}}
        assert DensityBasedExtractor._compute_density("Z", graph) == 0.0


class TestSteadyStateSelection:
    """Tests for DensityBasedExtractor.select_steady_state."""

    def test_steady_state_selection(self, mock_dataset_manager, sample_logs):
        """High density nodes should be selected as steady-state candidates."""
        # Set up mock: fetch_objects returns sample_logs as Weaviate objects
        weaviate_objects = [
            _make_weaviate_object(
                uuid=log.uuid,
                function_name=log.function_name,
                return_value=log.return_value,
                status=log.status,
                vector=log.vector,
            )
            for log in sample_logs
        ]
        fetch_response = _make_fetch_response(weaviate_objects)
        mock_dataset_manager.exec_col.query.fetch_objects.return_value = fetch_response

        extractor = DensityBasedExtractor(
            mock_dataset_manager, epsilon=0.1, top_k=10
        )
        candidates = extractor.select_steady_state("my_func")

        # Should get candidates (the cluster of similar vectors: log-1, log-2, log-4)
        assert len(candidates) > 0

        # All candidates should have STEADY strategy
        for c in candidates:
            assert c.strategy == SelectionStrategy.STEADY

        # Candidates should be ordered by density descending (highest first)
        densities = [c.density for c in candidates]
        assert densities == sorted(densities, reverse=True)

        # Candidates with zero density should be excluded
        for c in candidates:
            assert c.density > 0.0


class TestAnomalyDetection:
    """Tests for DensityBasedExtractor.select_anomalies."""

    def test_anomaly_detection(self, mock_dataset_manager, sample_logs):
        """Low density SUCCESS nodes should be selected as anomalies."""
        weaviate_objects = [
            _make_weaviate_object(
                uuid=log.uuid,
                function_name=log.function_name,
                return_value=log.return_value,
                status=log.status,
                vector=log.vector,
            )
            for log in sample_logs
        ]
        fetch_response = _make_fetch_response(weaviate_objects)
        mock_dataset_manager.exec_col.query.fetch_objects.return_value = fetch_response

        extractor = DensityBasedExtractor(
            mock_dataset_manager, epsilon=0.1, top_k=10
        )
        candidates = extractor.select_anomalies("my_func")

        # Should return candidates
        assert len(candidates) > 0

        # All candidates should have ANOMALY strategy
        for c in candidates:
            assert c.strategy == SelectionStrategy.ANOMALY

        # Anomalies are sorted by density ascending (lowest first)
        densities = [c.density for c in candidates]
        assert densities == sorted(densities)

        # FAILURE nodes (log-4) should NOT appear in anomaly candidates
        anomaly_uuids = {c.uuid for c in candidates}
        assert "log-4" not in anomaly_uuids

    def test_anomaly_detection_empty_logs(self, mock_dataset_manager):
        """When no logs are found, select_anomalies returns an empty list."""
        fetch_response = _make_fetch_response([])
        mock_dataset_manager.exec_col.query.fetch_objects.return_value = fetch_response

        extractor = DensityBasedExtractor(mock_dataset_manager)
        candidates = extractor.select_anomalies("nonexistent_func")
        assert candidates == []


class TestChatMLJSONLOutput:
    """Tests for DensityBasedExtractor.extract_golden_data JSONL output."""

    def test_chatml_jsonl_output(self, tmp_path, mock_dataset_manager, sample_logs):
        """Verify output is valid JSONL with correct ChatML structure."""
        weaviate_objects = [
            _make_weaviate_object(
                uuid=log.uuid,
                function_name=log.function_name,
                return_value=log.return_value,
                status=log.status,
                vector=log.vector,
            )
            for log in sample_logs
        ]
        fetch_response = _make_fetch_response(weaviate_objects)
        mock_dataset_manager.exec_col.query.fetch_objects.return_value = fetch_response

        output_path = str(tmp_path / "golden_data.jsonl")
        extractor = DensityBasedExtractor(
            mock_dataset_manager, epsilon=0.3, top_k=50
        )
        returned_path = extractor.extract_golden_data("my_func", output_path)

        # Returned path matches requested path
        assert returned_path == output_path

        # File should exist
        assert Path(output_path).is_file()

        # Each line must be valid JSON with ChatML structure
        with open(output_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        assert len(lines) > 0

        for line in lines:
            record = json.loads(line.strip())
            assert "messages" in record
            messages = record["messages"]
            assert len(messages) == 3

            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"

            # System message should mention the function name
            assert "my_func" in messages[0]["content"]

            # User message should contain metadata
            assert "function_name:" in messages[1]["content"]
            assert "density:" in messages[1]["content"]

            # Assistant message should be a return value from our logs
            assert messages[2]["content"] in {
                "result_1", "result_2", "result_3", "result_4"
            }

    def test_chatml_jsonl_creates_parent_directories(
        self, tmp_path, mock_dataset_manager
    ):
        """extract_golden_data should create parent directories if they do not exist."""
        fetch_response = _make_fetch_response([])
        mock_dataset_manager.exec_col.query.fetch_objects.return_value = fetch_response

        nested_path = str(tmp_path / "a" / "b" / "c" / "output.jsonl")
        extractor = DensityBasedExtractor(mock_dataset_manager)
        extractor.extract_golden_data("some_func", nested_path)

        # The file should exist (even if empty because no candidates)
        assert Path(nested_path).is_file()


# ============================================================================
# 2. Synthesizer Tests
# ============================================================================


class TestLevenshteinDistance:
    """Tests for FeedbackDiffAnalyzer._levenshtein_distance."""

    def test_levenshtein_distance(self):
        """Verify Levenshtein distance for known cases."""
        lev = FeedbackDiffAnalyzer._levenshtein_distance

        # Identical strings -> 0
        assert lev("kitten", "kitten") == 0

        # Classic example: kitten -> sitting = 3
        assert lev("kitten", "sitting") == 3

        # Single character difference
        assert lev("cat", "bat") == 1

        # Insertion
        assert lev("abc", "abcd") == 1

        # Deletion
        assert lev("abcd", "abc") == 1

        # Complete replacement
        assert lev("abc", "xyz") == 3

    def test_edit_distance_empty_strings(self):
        """Edge cases with empty strings."""
        lev = FeedbackDiffAnalyzer._levenshtein_distance

        # Both empty
        assert lev("", "") == 0

        # One empty
        assert lev("", "hello") == 5
        assert lev("hello", "") == 5

        # Single character vs empty
        assert lev("a", "") == 1
        assert lev("", "a") == 1


class TestFeedbackDiffAnalysis:
    """Tests for FeedbackDiffAnalyzer.analyze."""

    def test_feedback_diff_analysis(self):
        """Analyze a FeedbackPair and verify all expected keys are present."""
        analyzer = FeedbackDiffAnalyzer()
        pair = FeedbackPair(
            bad_output="The color is red.",
            fixed_output="The colour is blue.",
            input_prompt="What colour?",
            context={"source": "test"},
        )
        result = analyzer.analyze(pair)

        # Check required keys
        assert "edit_distance" in result
        assert "similarity_score" in result
        assert "diff_summary" in result
        assert "diff_segments" in result
        assert "input_prompt" in result
        assert "context" in result
        assert "bad_output" in result
        assert "fixed_output" in result

        # edit_distance should be a positive integer (strings differ)
        assert isinstance(result["edit_distance"], int)
        assert result["edit_distance"] > 0

        # similarity_score should be in [0, 1]
        assert 0.0 <= result["similarity_score"] <= 1.0

        # diff_segments should be a non-empty list
        assert len(result["diff_segments"]) > 0

        # Each segment should have type, bad, fixed
        for seg in result["diff_segments"]:
            assert "type" in seg
            assert seg["type"] in ("added", "removed", "changed")
            assert "bad" in seg
            assert "fixed" in seg

        # Pass-through fields
        assert result["input_prompt"] == "What colour?"
        assert result["context"] == {"source": "test"}

        # diff_summary should contain the edit distance
        assert str(result["edit_distance"]) in result["diff_summary"]

    def test_feedback_diff_identical_outputs(self):
        """When bad and fixed outputs are identical, edit distance is 0."""
        analyzer = FeedbackDiffAnalyzer()
        pair = FeedbackPair(
            bad_output="same text",
            fixed_output="same text",
            input_prompt="prompt",
        )
        result = analyzer.analyze(pair)
        assert result["edit_distance"] == 0
        assert result["similarity_score"] == pytest.approx(1.0)
        assert len(result["diff_segments"]) == 0


class TestRuleSynthesis:
    """Tests for RuleSetSynthesizer."""

    def test_rule_synthesis_from_diffs(self):
        """Diffs should produce Rule objects with correct fields."""
        analyzer = FeedbackDiffAnalyzer()
        pair = FeedbackPair(
            bad_output="Error: undefined",
            fixed_output="Result: 42",
            input_prompt="Calculate the answer",
        )
        diffs = [analyzer.analyze(pair)]

        synthesizer = RuleSetSynthesizer()
        rules = synthesizer.synthesize(diffs)

        assert len(rules) == 1
        rule = rules[0]

        assert isinstance(rule, Rule)
        assert rule.rule_id == "R-000"
        assert len(rule.description) > 0
        assert len(rule.condition) > 0
        assert len(rule.action) > 0
        assert rule.source_pair_index == 0
        assert isinstance(rule.priority, int)

    def test_rule_synthesis_multiple_diffs(self, sample_feedback_pairs):
        """Multiple feedback pairs produce multiple rules."""
        analyzer = FeedbackDiffAnalyzer()
        diffs = [analyzer.analyze(pair) for pair in sample_feedback_pairs]

        synthesizer = RuleSetSynthesizer()
        rules = synthesizer.synthesize(diffs)

        # Should have at least one rule per diff (could be fewer if contradictions merge)
        assert len(rules) >= 1
        assert len(rules) <= len(diffs)

        # Each rule should have a unique rule_id
        rule_ids = [r.rule_id for r in rules]
        # Note: after merging, rule_ids might not be strictly unique due to removal,
        # but the remaining ones should be distinct
        assert len(rule_ids) == len(set(rule_ids))


class TestContradictingRules:
    """Tests for contradiction detection and merging."""

    def test_contradicting_rules_detection(self):
        """Find contradictory rules: similar conditions, dissimilar actions."""
        rules = [
            Rule(
                rule_id="R-000",
                description="Rule A",
                condition="When the output contains error message in response to query",
                action="Remove the error message completely",
                priority=1,
            ),
            Rule(
                rule_id="R-001",
                description="Rule B",
                condition="When the output contains error message in response to query",
                action="XYZZY totally different action here with random words 12345",
                priority=2,
            ),
        ]

        contradictions = RuleSetSynthesizer._detect_contradictions(rules)

        # These rules have very similar conditions and very different actions
        assert len(contradictions) > 0
        assert (0, 1) in contradictions

    def test_no_contradictions_for_similar_rules(self):
        """Rules with similar conditions AND similar actions should NOT be contradictions."""
        rules = [
            Rule(
                rule_id="R-000",
                description="Rule A",
                condition="When output contains error",
                action="Remove the error message",
                priority=1,
            ),
            Rule(
                rule_id="R-001",
                description="Rule B",
                condition="When output contains error",
                action="Remove the error message and log it",
                priority=1,
            ),
        ]

        contradictions = RuleSetSynthesizer._detect_contradictions(rules)
        assert len(contradictions) == 0

    def test_contradicting_rules_merge(self):
        """Merge contradictions: higher priority rule wins."""
        rules = [
            Rule(
                rule_id="R-000",
                description="Low priority rule",
                condition="When output contains error",
                action="Remove error AAAA BBBB CCCC DDDD EEEE FFFF GGGG",
                priority=1,
            ),
            Rule(
                rule_id="R-001",
                description="High priority rule",
                condition="When output contains error",
                action="XYZZY 1234 abcde totally different action replace xyz",
                priority=3,
            ),
        ]

        contradictions = [(0, 1)]
        merged = RuleSetSynthesizer._merge_contradicting_rules(rules, contradictions)

        # Higher priority rule (R-001) should be kept
        assert len(merged) == 1
        assert merged[0].rule_id == "R-001"
        # Merged description should reference the removed rule
        assert "R-000" in merged[0].description

    def test_contradicting_rules_merge_equal_priority(self):
        """When priorities are equal, the first rule (rule_a) is kept."""
        rules = [
            Rule(
                rule_id="R-000",
                description="First rule",
                condition="same condition",
                action="action A totally unique unrelated stuff random words",
                priority=2,
            ),
            Rule(
                rule_id="R-001",
                description="Second rule",
                condition="same condition",
                action="XYZZY different action 12345 abcdef ghijkl mnopqr",
                priority=2,
            ),
        ]

        contradictions = [(0, 1)]
        merged = RuleSetSynthesizer._merge_contradicting_rules(rules, contradictions)

        assert len(merged) == 1
        # Equal priority -> first rule kept
        assert merged[0].rule_id == "R-000"
        assert "R-001" in merged[0].description


class TestHookScriptGeneration:
    """Tests for HookScriptGenerator."""

    def test_hook_script_generation(self, tmp_path):
        """Generate script file and verify it exists on disk."""
        rules = [
            Rule(
                rule_id="R-000",
                description="Test rule",
                condition="When output contains error",
                action='Replace "error" with "success"',
                priority=1,
            ),
        ]

        generator = HookScriptGenerator()
        script_path = generator.generate(rules, str(tmp_path))

        assert os.path.isfile(script_path)
        assert script_path.endswith("generated_prompt_hook.py")

        # File should have non-trivial content
        content = Path(script_path).read_text(encoding="utf-8")
        assert len(content) > 100

    def test_hook_script_ast_validity(self, tmp_path):
        """Parse the generated script with ast.parse -- must be valid Python."""
        rules = [
            Rule(
                rule_id="R-000",
                description="Test rule",
                condition='When output says "hello"',
                action='Replace "hello" with "goodbye"',
                priority=1,
            ),
            Rule(
                rule_id="R-001",
                description="Another rule",
                condition="always",
                action='Add: "suffix text"',
                priority=0,
            ),
        ]

        generator = HookScriptGenerator()
        script_path = generator.generate(rules, str(tmp_path))

        source = Path(script_path).read_text(encoding="utf-8")

        # Must not raise SyntaxError
        tree = ast.parse(source)
        assert tree is not None

    def test_hook_script_function_signatures(self, tmp_path):
        """Verify intercept_prompt and intercept_output functions exist in the AST."""
        rules = [
            Rule(
                rule_id="R-000",
                description="Test rule",
                condition="always",
                action="No action required",
                priority=0,
            ),
        ]

        generator = HookScriptGenerator()
        script_path = generator.generate(rules, str(tmp_path))

        source = Path(script_path).read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Collect top-level function names
        function_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]

        assert "intercept_prompt" in function_names
        assert "intercept_output" in function_names

        # Each should accept (user_input/model_output, context)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in (
                "intercept_prompt",
                "intercept_output",
            ):
                # 2 positional arguments
                assert len(node.args.args) == 2

    def test_hook_script_empty_rules(self, tmp_path):
        """Generating with an empty rules list should still produce a valid script."""
        generator = HookScriptGenerator()
        script_path = generator.generate([], str(tmp_path))

        source = Path(script_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert tree is not None

        # RULES variable should be an empty list
        assert "RULES" in source


class TestSynthesisPipelineIntegration:
    """Integration test for the full synthesis pipeline."""

    def test_synthesis_pipeline_integration(self, tmp_path, sample_feedback_pairs):
        """Full pipeline: FeedbackPairs -> diffs -> rules -> hook script file."""
        output_dir = str(tmp_path / "synthesis_output")

        script_path = run_synthesis_pipeline(
            pairs=sample_feedback_pairs,
            output_dir=output_dir,
        )

        # Script file should exist
        assert os.path.isfile(script_path)

        # Script should be valid Python
        source = Path(script_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert tree is not None

        # Should contain intercept functions
        func_names = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        ]
        assert "intercept_prompt" in func_names
        assert "intercept_output" in func_names

        # RULES should appear in the source
        assert "RULES" in source


# ============================================================================
# 3. Pipeline Tests
# ============================================================================


class TestBaseTrainerAbstract:
    """Tests for BaseTrainer abstraction."""

    def test_base_trainer_is_abstract(self):
        """Cannot instantiate BaseTrainer directly."""
        with pytest.raises(TypeError):
            BaseTrainer()


class TestOpenAITrainer:
    """Tests for OpenAITrainer with mocked OpenAI API."""

    def test_openai_trainer_prepare_data(self, tmp_path):
        """Mock openai.files.create and verify prepare_data returns file ID."""
        # Create a dummy JSONL file
        jsonl_file = tmp_path / "training.jsonl"
        jsonl_file.write_text('{"messages": []}\n', encoding="utf-8")

        trainer = OpenAITrainer(api_key="sk-test-fake-key")

        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_file_obj = MagicMock()
        mock_file_obj.id = "file-abc123"
        mock_client.files.create.return_value = mock_file_obj
        trainer._client = mock_client

        file_id = trainer.prepare_data(str(jsonl_file))

        assert file_id == "file-abc123"
        mock_client.files.create.assert_called_once()

        # Verify the call kwargs
        call_kwargs = mock_client.files.create.call_args
        assert call_kwargs[1]["purpose"] == "fine-tune"

    def test_openai_trainer_prepare_data_file_not_found(self):
        """prepare_data raises FileNotFoundError for missing files."""
        trainer = OpenAITrainer(api_key="sk-test-fake-key")
        trainer._client = MagicMock()

        with pytest.raises(FileNotFoundError):
            trainer.prepare_data("/nonexistent/path/training.jsonl")

    def test_openai_trainer_start_training(self):
        """Mock openai.fine_tuning.jobs.create and verify start_training returns job ID."""
        trainer = OpenAITrainer(api_key="sk-test-fake-key")

        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "ftjob-xyz789"
        mock_client.fine_tuning.jobs.create.return_value = mock_job
        trainer._client = mock_client

        job_id = trainer.start_training("file-abc123")

        assert job_id == "ftjob-xyz789"
        mock_client.fine_tuning.jobs.create.assert_called_once_with(
            training_file="file-abc123",
            model="gpt-4o-mini-2024-07-18",
        )

    def test_openai_trainer_get_status(self):
        """get_status returns a dict with expected keys."""
        trainer = OpenAITrainer(api_key="sk-test-fake-key")

        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "ftjob-xyz789"
        mock_job.status = "running"
        mock_job.model = "gpt-4o-mini-2024-07-18"
        mock_job.trained_tokens = 1000
        mock_job.created_at = 1700000000
        mock_job.finished_at = None
        mock_job.error = None
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
        trainer._client = mock_client

        status = trainer.get_status("ftjob-xyz789")

        assert status["job_id"] == "ftjob-xyz789"
        assert status["status"] == "running"
        assert status["model"] == "gpt-4o-mini-2024-07-18"

    def test_openai_trainer_get_result(self):
        """get_result returns a dict including fine_tuned_model."""
        trainer = OpenAITrainer(api_key="sk-test-fake-key")

        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "ftjob-xyz789"
        mock_job.status = "succeeded"
        mock_job.model = "gpt-4o-mini-2024-07-18"
        mock_job.fine_tuned_model = "ft:gpt-4o-mini:my-org::abc123"
        mock_job.trained_tokens = 5000
        mock_job.created_at = 1700000000
        mock_job.finished_at = 1700003600
        mock_job.result_files = ["file-result-001"]
        mock_job.error = None
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
        trainer._client = mock_client

        result = trainer.get_result("ftjob-xyz789")

        assert result["fine_tuned_model"] == "ft:gpt-4o-mini:my-org::abc123"
        assert result["status"] == "succeeded"


class TestTrainingPipelineRun:
    """Tests for TrainingPipeline.run orchestration."""

    def test_training_pipeline_run(self, tmp_path, mock_dataset_manager, sample_logs):
        """Mock all components and verify the pipeline orchestrates them correctly."""
        # Set up mock extractor
        weaviate_objects = [
            _make_weaviate_object(
                uuid=log.uuid,
                function_name=log.function_name,
                return_value=log.return_value,
                status=log.status,
                vector=log.vector,
            )
            for log in sample_logs
        ]
        fetch_response = _make_fetch_response(weaviate_objects)
        mock_dataset_manager.exec_col.query.fetch_objects.return_value = fetch_response

        extractor = DensityBasedExtractor(
            mock_dataset_manager, epsilon=0.3, top_k=50
        )

        # Set up mock trainer
        mock_trainer = MagicMock(spec=BaseTrainer)
        mock_trainer.prepare_data.return_value = "file-test-id"
        mock_trainer.start_training.return_value = "ftjob-test-id"

        # Set up mock monitor
        mock_monitor = MagicMock()

        output_dir = str(tmp_path / "pipeline_output")
        synth_dir = str(tmp_path / "synth_output")
        os.makedirs(output_dir, exist_ok=True)

        pipeline = TrainingPipeline(
            extractor=extractor,
            synthesizer_output_dir=synth_dir,
            trainer=mock_trainer,
            monitor=mock_monitor,
        )

        feedback_pairs = [
            FeedbackPair(
                bad_output="wrong answer",
                fixed_output="correct answer",
                input_prompt="test prompt",
            ),
        ]

        result = pipeline.run("my_func", feedback_pairs, output_dir)

        # Verify result structure
        assert "function_name" in result
        assert result["function_name"] == "my_func"
        assert "file_id" in result
        assert result["file_id"] == "file-test-id"
        assert "job_id" in result
        assert result["job_id"] == "ftjob-test-id"
        assert "extraction" in result
        assert "synthesis" in result
        assert "jsonl_path" in result

        # Verify trainer was called correctly
        mock_trainer.prepare_data.assert_called_once()
        mock_trainer.start_training.assert_called_once_with("file-test-id")

        # Verify monitor callbacks were invoked for all 4 stages
        stage_names_started = [
            call.args[0] for call in mock_monitor.on_stage_start.call_args_list
        ]
        stage_names_completed = [
            call.args[0] for call in mock_monitor.on_stage_complete.call_args_list
        ]

        for stage in ["extract", "synthesize", "prepare", "train"]:
            assert stage in stage_names_started
            assert stage in stage_names_completed


# ============================================================================
# 4. Dashboard Tests
# ============================================================================


class TestPipelineMonitorCallbacks:
    """Tests for PipelineMonitor callback methods."""

    def test_pipeline_monitor_callbacks(self):
        """on_stage_start, on_progress, on_stage_complete work correctly."""
        monitor = PipelineMonitor()

        # Initially no current stage
        assert monitor.current_stage is None
        assert monitor.stage_results == {}

        # Start a stage
        monitor.on_stage_start("extract")
        assert monitor.current_stage == "extract"
        assert "extract" in monitor._stage_start_times

        # Report progress
        monitor.on_progress("extract", 5, 10)
        # on_progress on PipelineMonitor just logs, no state change to verify
        # but it should not raise

        # Complete the stage
        monitor.on_stage_complete("extract", {"candidates": 42})
        assert monitor.current_stage is None
        assert "extract" in monitor.stage_results
        # The result should contain the original data plus _elapsed
        assert "_elapsed" in monitor.stage_results["extract"]

    def test_pipeline_monitor_on_log(self):
        """on_log stores log entries with timestamp, level, and message."""
        monitor = PipelineMonitor()

        monitor.on_log("Test message", "info")
        monitor.on_log("Warning message", "warning")

        assert len(monitor._logs) == 2
        assert monitor._logs[0]["message"] == "Test message"
        assert monitor._logs[0]["level"] == "info"
        assert "timestamp" in monitor._logs[0]

        assert monitor._logs[1]["message"] == "Warning message"
        assert monitor._logs[1]["level"] == "warning"

    def test_pipeline_monitor_multiple_stages(self):
        """Monitor correctly tracks multiple sequential stages."""
        monitor = PipelineMonitor()

        monitor.on_stage_start("extract")
        monitor.on_stage_complete("extract", {"count": 10})

        monitor.on_stage_start("synthesize")
        monitor.on_stage_complete("synthesize", {"rules": 3})

        monitor.on_stage_start("train")
        monitor.on_stage_complete("train", {"job_id": "ftjob-1"})

        assert len(monitor.stage_results) == 3
        assert "extract" in monitor.stage_results
        assert "synthesize" in monitor.stage_results
        assert "train" in monitor.stage_results
        assert monitor.current_stage is None


class TestCLIDashboard:
    """Tests for CLIDashboard."""

    def test_cli_dashboard_creation(self):
        """CLIDashboard instantiates without error."""
        dashboard = CLIDashboard()
        assert dashboard is not None
        assert dashboard.console is not None
        assert dashboard.current_stage is None
        assert dashboard.stage_results == {}

    def test_cli_dashboard_with_custom_console(self):
        """CLIDashboard accepts a custom Console instance."""
        from rich.console import Console

        custom_console = Console(quiet=True)
        dashboard = CLIDashboard(console=custom_console)
        assert dashboard.console is custom_console

    def test_cli_dashboard_inherits_pipeline_monitor(self):
        """CLIDashboard should be a subclass of PipelineMonitor."""
        assert issubclass(CLIDashboard, PipelineMonitor)

        dashboard = CLIDashboard()
        assert isinstance(dashboard, PipelineMonitor)


class TestMonitorSummary:
    """Tests for PipelineMonitor.summary."""

    def test_monitor_summary(self):
        """Verify summary returns correct structure after running stages."""
        monitor = PipelineMonitor()

        # Run through some stages
        monitor.on_stage_start("extract")
        monitor.on_stage_complete("extract", {"count": 10})

        monitor.on_stage_start("synthesize")
        monitor.on_stage_complete("synthesize", {"rules": 5})

        summary = monitor.summary()

        assert "total_elapsed" in summary
        assert isinstance(summary["total_elapsed"], float)
        assert summary["total_elapsed"] >= 0.0

        assert "stages_completed" in summary
        assert isinstance(summary["stages_completed"], list)
        assert "extract" in summary["stages_completed"]
        assert "synthesize" in summary["stages_completed"]

        assert "stage_results" in summary
        assert isinstance(summary["stage_results"], dict)

        assert "log_count" in summary
        assert isinstance(summary["log_count"], int)
        assert summary["log_count"] == 0  # no on_log calls made

    def test_monitor_summary_with_logs(self):
        """Summary log_count reflects the number of on_log calls."""
        monitor = PipelineMonitor()

        monitor.on_log("message 1", "info")
        monitor.on_log("message 2", "warning")
        monitor.on_log("message 3", "error")

        summary = monitor.summary()
        assert summary["log_count"] == 3

    def test_monitor_summary_empty(self):
        """Summary works even when no stages have been run."""
        monitor = PipelineMonitor()
        summary = monitor.summary()

        assert summary["stages_completed"] == []
        assert summary["stage_results"] == {}
        assert summary["log_count"] == 0
        assert summary["total_elapsed"] >= 0.0


# ============================================================================
# 5. Cost Guard Tests
# ============================================================================


class TestCountTokensInJsonl:
    """Tests for count_tokens_in_jsonl."""

    def test_count_tokens_in_jsonl(self, tmp_path):
        """Token count should be positive for a valid JSONL file."""
        jsonl_file = tmp_path / "train.jsonl"
        record = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        jsonl_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        count = count_tokens_in_jsonl(str(jsonl_file), "gpt-4o-mini-2024-07-18")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_file(self, tmp_path):
        """Empty JSONL yields zero tokens."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("", encoding="utf-8")

        count = count_tokens_in_jsonl(str(jsonl_file), "gpt-4o-mini-2024-07-18")
        assert count == 0


class TestEstimateTrainingCost:
    """Tests for estimate_training_cost."""

    def test_estimate_training_cost(self):
        """Verify cost formula: tokens * epochs * price_per_token."""
        # gpt-4o-mini: $3.00 per 1M tokens
        cost = estimate_training_cost(1_000_000, "gpt-4o-mini-2024-07-18", n_epochs=1)
        assert cost == pytest.approx(3.00)

        # With 3 epochs
        cost_3 = estimate_training_cost(1_000_000, "gpt-4o-mini-2024-07-18", n_epochs=3)
        assert cost_3 == pytest.approx(9.00)

    def test_estimate_unknown_model_uses_cheapest(self):
        """Unknown model falls back to the cheapest price."""
        cost = estimate_training_cost(1_000_000, "unknown-model-xyz", n_epochs=1)
        # Cheapest is gpt-4o-mini at $3.00/1M
        assert cost == pytest.approx(3.00)


class TestCheckBudget:
    """Tests for check_budget."""

    def test_check_budget_within_limit(self, tmp_path):
        """Budget within limit returns a CostEstimate with approved=True."""
        jsonl_file = tmp_path / "train.jsonl"
        record = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        jsonl_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = check_budget(
            str(jsonl_file), "gpt-4o-mini-2024-07-18", max_budget_usd=100.0
        )

        assert isinstance(result, CostEstimate)
        assert result.approved is True
        assert result.estimated_cost_usd <= result.budget_usd
        assert result.token_count > 0

    def test_check_budget_exceeded(self, tmp_path):
        """Budget exceeded raises BudgetExceededError."""
        jsonl_file = tmp_path / "train.jsonl"
        record = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        jsonl_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        with pytest.raises(BudgetExceededError) as exc_info:
            # $0.0000001 budget -- guaranteed to exceed
            check_budget(
                str(jsonl_file), "gpt-4o-mini-2024-07-18", max_budget_usd=0.0000001
            )

        assert exc_info.value.estimated_cost > exc_info.value.budget
        assert exc_info.value.token_count > 0


class TestOpenAITrainerWithBudgetGuard:
    """Integration test: OpenAITrainer respects max_budget_usd."""

    def test_openai_trainer_with_budget_guard(self, tmp_path):
        """prepare_data raises BudgetExceededError when budget is too low."""
        jsonl_file = tmp_path / "training.jsonl"
        record = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        jsonl_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        trainer = OpenAITrainer(api_key="sk-test-fake-key", max_budget_usd=0.0000001)
        trainer._client = MagicMock()

        with pytest.raises(BudgetExceededError):
            trainer.prepare_data(str(jsonl_file))

        # Ensure the upload was never called
        trainer._client.files.create.assert_not_called()


# ============================================================================
# 6. Hook Versioning Tests
# ============================================================================


class TestVersionManagerSaveVersion:
    """Tests for HookVersionManager.save_version."""

    def test_version_manager_save_version(self, tmp_path):
        """Saving a version creates a versioned file and manifest."""
        vm = HookVersionManager(str(tmp_path))
        vid = vm.save_version("# hook v1\n", rules_count=3)

        assert vid == "v001"

        # Manifest should exist
        manifest_path = tmp_path / ".hook_versions" / "manifest.json"
        assert manifest_path.is_file()

        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["versions"]) == 1
        assert manifest["active"] == "v001"

        # Version file should exist
        version_file = tmp_path / ".hook_versions" / manifest["versions"][0]["filename"]
        assert version_file.is_file()
        assert version_file.read_text() == "# hook v1\n"


class TestVersionManagerMultipleVersions:
    """Tests for sequential version numbering."""

    def test_version_manager_multiple_versions(self, tmp_path):
        """Multiple saves produce sequential version IDs."""
        vm = HookVersionManager(str(tmp_path))

        v1 = vm.save_version("# v1\n", rules_count=1)
        v2 = vm.save_version("# v2\n", rules_count=2)
        v3 = vm.save_version("# v3\n", rules_count=3)

        assert v1 == "v001"
        assert v2 == "v002"
        assert v3 == "v003"

        versions = vm.list_versions()
        assert len(versions) == 3
        assert versions[2]["active"] is True
        assert versions[0]["active"] is False


class TestVersionManagerRollback:
    """Tests for HookVersionManager.rollback."""

    def test_version_manager_rollback(self, tmp_path):
        """Rollback restores a previous version's content to the active file."""
        output_dir = str(tmp_path)
        vm = HookVersionManager(output_dir)

        # Generate first version via HookScriptGenerator
        gen = HookScriptGenerator()
        rules_v1 = [
            Rule(rule_id="R-000", description="v1 rule", condition="always",
                 action="No action", priority=0),
        ]
        gen.generate(rules_v1, output_dir)

        # Generate second (different) version
        rules_v2 = [
            Rule(rule_id="R-000", description="v1 rule", condition="always",
                 action="No action", priority=0),
            Rule(rule_id="R-001", description="v2 rule", condition="always",
                 action="No action", priority=1),
        ]
        gen.generate(rules_v2, output_dir)

        # Active file now has v2 content
        active_path = tmp_path / "generated_prompt_hook.py"
        v2_content = active_path.read_text()
        assert "R-001" in v2_content

        # Rollback to v001
        rolled = vm.rollback("v001")
        assert rolled == "v001"

        # Active file should now have v1 content (no R-001)
        v1_content = active_path.read_text()
        assert "R-001" not in v1_content
        assert "R-000" in v1_content


class TestVersionManagerListVersions:
    """Tests for HookVersionManager.list_versions."""

    def test_version_manager_list_versions(self, tmp_path):
        """list_versions returns all versions with active flag."""
        vm = HookVersionManager(str(tmp_path))

        vm.save_version("# a\n", rules_count=1)
        vm.save_version("# b\n", rules_count=2)

        versions = vm.list_versions()
        assert len(versions) == 2

        assert versions[0]["version_id"] == "v001"
        assert versions[0]["rules_count"] == 1
        assert versions[0]["active"] is False
        assert "timestamp" in versions[0]

        assert versions[1]["version_id"] == "v002"
        assert versions[1]["rules_count"] == 2
        assert versions[1]["active"] is True


class TestHookGeneratorWithVersioning:
    """Integration: HookScriptGenerator stores versions when enabled."""

    def test_hook_generator_with_versioning(self, tmp_path):
        """generate() with versioning creates .hook_versions dir and entries."""
        rules = [
            Rule(rule_id="R-000", description="Test", condition="always",
                 action="No action", priority=0),
        ]

        gen = HookScriptGenerator()
        gen.generate(rules, str(tmp_path), enable_versioning=True)

        # Version directory should exist
        versions_dir = tmp_path / ".hook_versions"
        assert versions_dir.is_dir()

        vm = HookVersionManager(str(tmp_path))
        versions = vm.list_versions()
        assert len(versions) == 1
        assert versions[0]["rules_count"] == 1

    def test_hook_generator_without_versioning(self, tmp_path):
        """generate() with versioning disabled skips version storage."""
        rules = [
            Rule(rule_id="R-000", description="Test", condition="always",
                 action="No action", priority=0),
        ]

        gen = HookScriptGenerator()
        gen.generate(rules, str(tmp_path), enable_versioning=False)

        # No .hook_versions directory should exist
        versions_dir = tmp_path / ".hook_versions"
        assert not versions_dir.exists()
