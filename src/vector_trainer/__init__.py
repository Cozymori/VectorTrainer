"""VectorTrainer: Self-Evolving AI Loop engine for automated LLM fine-tuning."""

__version__ = "0.1.0"

from .types import (
    CostEstimate,
    DatasetManagerProtocol,
    ExecutionLog,
    GoldenCandidate,
    SelectionStrategy,
    FeedbackPair,
    Rule,
)
from .cost_guard import BudgetExceededError
from .extractor import DensityBasedExtractor
from .synthesizer import (
    FeedbackDiffAnalyzer,
    RuleSetSynthesizer,
    HookScriptGenerator,
    HookVersionManager,
    run_synthesis_pipeline,
)
from .pipeline import BaseTrainer, OpenAITrainer, TrainingPipeline
from .dashboard import PipelineMonitor, CLIDashboard

__all__ = [
    "BudgetExceededError",
    "CostEstimate",
    "DensityBasedExtractor",
    "FeedbackDiffAnalyzer",
    "RuleSetSynthesizer",
    "HookScriptGenerator",
    "HookVersionManager",
    "run_synthesis_pipeline",
    "BaseTrainer",
    "OpenAITrainer",
    "TrainingPipeline",
    "PipelineMonitor",
    "CLIDashboard",
    "DatasetManagerProtocol",
    "ExecutionLog",
    "GoldenCandidate",
    "SelectionStrategy",
    "FeedbackPair",
    "Rule",
]
