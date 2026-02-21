"""Training pipeline: trainer abstraction (Strategy Pattern) and orchestration.

This module provides:

- ``BaseTrainer`` -- abstract interface for any fine-tuning backend.
- ``OpenAITrainer`` -- concrete implementation backed by the OpenAI fine-tuning API.
- ``TrainingPipeline`` -- orchestrator that chains extraction, synthesis, data
  upload, and training into a single ``run()`` call.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .types import FeedbackPair

if TYPE_CHECKING:
    from openai import OpenAI

    # Avoid circular imports -- these modules are only needed for type hints.
    from .extractor import DensityBasedExtractor  # noqa: F401
    from .synthesizer import run_synthesis_pipeline  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract Trainer (Strategy Pattern)
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """Abstract base class for fine-tuning backends.

    Every concrete trainer must implement the four lifecycle methods that
    mirror a typical fine-tuning workflow: **prepare** -> **start** ->
    **poll status** -> **retrieve result**.
    """

    @abstractmethod
    def prepare_data(self, jsonl_path: str) -> str:
        """Upload / preprocess training data.

        Parameters
        ----------
        jsonl_path:
            Local filesystem path to a ``.jsonl`` training file.

        Returns
        -------
        str
            A backend-specific file identifier (e.g. OpenAI file ID).
        """

    @abstractmethod
    def start_training(self, file_id: str, **kwargs: Any) -> str:
        """Start a fine-tuning job.

        Parameters
        ----------
        file_id:
            Identifier returned by :meth:`prepare_data`.
        **kwargs:
            Backend-specific hyperparameters.

        Returns
        -------
        str
            A backend-specific job identifier.
        """

    @abstractmethod
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Poll the current training status.

        Parameters
        ----------
        job_id:
            Identifier returned by :meth:`start_training`.

        Returns
        -------
        dict
            At minimum ``{"status": "<backend status string>"}``.
        """

    @abstractmethod
    def get_result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the final training result.

        Parameters
        ----------
        job_id:
            Identifier returned by :meth:`start_training`.

        Returns
        -------
        dict
            Includes the fine-tuned model name (if available) plus any
            metadata returned by the backend.
        """


# ---------------------------------------------------------------------------
# OpenAI Trainer
# ---------------------------------------------------------------------------


class OpenAITrainer(BaseTrainer):
    """Fine-tuning backend powered by the OpenAI API.

    Parameters
    ----------
    api_key:
        OpenAI API key.  When *None* the ``OPENAI_API_KEY`` environment
        variable is used instead.
    model:
        Base model to fine-tune (default ``gpt-4o-mini-2024-07-18``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-2024-07-18",
        max_budget_usd: Optional[float] = None,
    ) -> None:
        self._api_key: Optional[str] = api_key
        self.model: str = model
        self.max_budget_usd: Optional[float] = max_budget_usd
        self._client: Optional[OpenAI] = None

    # -- lazy client ----------------------------------------------------------

    @property
    def client(self) -> OpenAI:
        """Return the OpenAI client, creating it on first access."""

        if self._client is None:
            from openai import OpenAI as _OpenAI

            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError(
                    "No OpenAI API key provided.  Pass api_key to the "
                    "constructor or set the OPENAI_API_KEY environment variable."
                )
            self._client = _OpenAI(api_key=key)
            logger.debug("OpenAI client initialised (model=%s).", self.model)
        return self._client

    # -- BaseTrainer implementation -------------------------------------------

    def prepare_data(self, jsonl_path: str) -> str:
        """Upload a JSONL training file to OpenAI.

        When :attr:`max_budget_usd` is set, the estimated training cost is
        checked **before** uploading.  If the cost exceeds the budget a
        :class:`~vector_trainer.cost_guard.BudgetExceededError` is raised
        and no API call is made.

        Returns the OpenAI file ID (e.g. ``file-abc123``).
        """

        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"Training file not found: {jsonl_path}")

        if self.max_budget_usd is not None:
            from .cost_guard import check_budget

            cost_info = check_budget(jsonl_path, self.model, self.max_budget_usd)
            logger.info(
                "Cost estimate: %d tokens, $%.4f (budget: $%.2f)",
                cost_info.token_count,
                cost_info.estimated_cost_usd,
                cost_info.budget_usd,
            )

        logger.info("Uploading training data: %s", jsonl_path)
        with open(jsonl_path, "rb") as fh:
            file_obj = self.client.files.create(file=fh, purpose="fine-tune")

        logger.info("Upload complete. file_id=%s", file_obj.id)
        return file_obj.id

    def start_training(self, file_id: str, **kwargs: Any) -> str:
        """Create an OpenAI fine-tuning job.

        Any extra *kwargs* are forwarded to the API (e.g.
        ``hyperparameters``).
        """

        logger.info(
            "Starting fine-tuning job (model=%s, file_id=%s).",
            self.model,
            file_id,
        )
        job = self.client.fine_tuning.jobs.create(
            training_file=file_id,
            model=self.model,
            **kwargs,
        )
        logger.info("Fine-tuning job created. job_id=%s", job.id)
        return job.id

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the current status of a fine-tuning job."""

        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "trained_tokens": getattr(job, "trained_tokens", None),
            "created_at": getattr(job, "created_at", None),
            "finished_at": getattr(job, "finished_at", None),
            "error": getattr(job, "error", None),
        }

    def get_result(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the result of a completed fine-tuning job.

        The returned dictionary includes every field from :meth:`get_status`
        plus the ``fine_tuned_model`` name.
        """

        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "trained_tokens": getattr(job, "trained_tokens", None),
            "created_at": getattr(job, "created_at", None),
            "finished_at": getattr(job, "finished_at", None),
            "result_files": getattr(job, "result_files", None),
            "error": getattr(job, "error", None),
        }


# ---------------------------------------------------------------------------
# Training Pipeline Orchestrator
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """End-to-end orchestrator for the VectorTrainer fine-tuning pipeline.

    The pipeline chains four major stages:

    1. **extract** -- select golden training examples from the vector store.
    2. **synthesize** -- produce a hook script from user feedback pairs.
    3. **prepare** -- upload training data to the fine-tuning backend.
    4. **train** -- kick off a fine-tuning job.

    Parameters
    ----------
    extractor:
        A :class:`~vector_trainer.extractor.DensityBasedExtractor` instance
        used to select golden data.
    synthesizer_output_dir:
        Directory where synthesised hook scripts will be written.
    trainer:
        A :class:`BaseTrainer` implementation (e.g. :class:`OpenAITrainer`).
    monitor:
        Optional :class:`PipelineMonitor` (or duck-typed equivalent) that
        receives ``on_stage_start`` / ``on_stage_complete`` callbacks for
        dashboard integration.
    """

    def __init__(
        self,
        extractor: DensityBasedExtractor,
        synthesizer_output_dir: str,
        trainer: BaseTrainer,
        monitor: Optional[Any] = None,
    ) -> None:
        self.extractor = extractor
        self.synthesizer_output_dir: str = synthesizer_output_dir
        self.trainer: BaseTrainer = trainer
        self.monitor: Optional[Any] = monitor

    # -- helpers --------------------------------------------------------------

    def _notify_start(self, stage: str) -> None:
        """Notify the monitor that *stage* is starting (if one is set)."""

        logger.info("Pipeline stage starting: %s", stage)
        if self.monitor is not None:
            try:
                self.monitor.on_stage_start(stage)
            except Exception:
                logger.warning(
                    "Monitor callback on_stage_start(%r) raised an exception.",
                    stage,
                    exc_info=True,
                )

    def _notify_complete(self, stage: str, result: Any) -> None:
        """Notify the monitor that *stage* has completed (if one is set)."""

        logger.info("Pipeline stage complete: %s", stage)
        if self.monitor is not None:
            try:
                self.monitor.on_stage_complete(stage, result)
            except Exception:
                logger.warning(
                    "Monitor callback on_stage_complete(%r, ...) raised an exception.",
                    stage,
                    exc_info=True,
                )

    # -- main entry point -----------------------------------------------------

    def run(
        self,
        function_name: str,
        feedback_pairs: List[FeedbackPair],
        output_dir: str,
    ) -> Dict[str, Any]:
        """Execute the full fine-tuning pipeline.

        Parameters
        ----------
        function_name:
            The target function whose golden data will be extracted.
        feedback_pairs:
            User-supplied ``(bad_output, fixed_output)`` pairs for rule
            synthesis.
        output_dir:
            Directory where intermediate artefacts (e.g. the JSONL file)
            will be stored.

        Returns
        -------
        dict
            A summary containing keys for each stage's output:
            ``extraction``, ``synthesis``, ``file_id``, ``job_id``.
        """

        # Lazy import to break circular dependency at runtime.
        from .synthesizer import run_synthesis_pipeline

        jsonl_path = str(Path(output_dir) / f"{function_name}_golden.jsonl")

        # -- Stage 1: Extract golden data ------------------------------------
        stage = "extract"
        self._notify_start(stage)
        extraction_result = self.extractor.extract_golden_data(
            function_name, jsonl_path
        )
        self._notify_complete(stage, {"jsonl_path": extraction_result})

        # -- Stage 2: Synthesize hook script ---------------------------------
        stage = "synthesize"
        self._notify_start(stage)
        synthesis_result = run_synthesis_pipeline(
            feedback_pairs, self.synthesizer_output_dir
        )
        self._notify_complete(stage, {"hook_script": synthesis_result})

        # -- Stage 3: Prepare (upload) training data -------------------------
        stage = "prepare"
        self._notify_start(stage)
        file_id: str = self.trainer.prepare_data(jsonl_path)
        self._notify_complete(stage, {"file_id": file_id})

        # -- Stage 4: Start training -----------------------------------------
        stage = "train"
        self._notify_start(stage)
        job_id: str = self.trainer.start_training(file_id)
        self._notify_complete(stage, {"job_id": job_id})

        # -- Summary ---------------------------------------------------------
        summary: Dict[str, Any] = {
            "function_name": function_name,
            "jsonl_path": jsonl_path,
            "extraction": extraction_result,
            "synthesis": synthesis_result,
            "file_id": file_id,
            "job_id": job_id,
        }
        logger.info("Pipeline run complete. job_id=%s", job_id)
        return summary
