"""Real-time monitoring dashboard for the VectorTrainer fine-tuning pipeline.

Provides two interfaces:
- **CLIDashboard**: Rich-powered terminal UI with progress bars, live tables,
  and formatted log output.
- **WebDashboard**: Streamlit-powered browser UI with stage indicators,
  metrics, and data-visualization placeholders.

Both classes inherit from ``PipelineMonitor``, a callback-based base that
TrainingPipeline invokes at each stage transition.

Usage (CLI)::

    from vector_trainer.dashboard import run_cli_dashboard

    result = run_cli_dashboard(pipeline, "my_func", feedback_pairs, "./out")

Usage (Streamlit)::

    streamlit run src/vector_trainer/dashboard.py
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage display helpers
# ---------------------------------------------------------------------------

_STAGE_ICONS: Dict[str, str] = {
    "extract": "[bold cyan]EXTRACT[/bold cyan]",
    "synthesize": "[bold magenta]SYNTHESIZE[/bold magenta]",
    "train": "[bold green]TRAIN[/bold green]",
}

_LEVEL_STYLES: Dict[str, str] = {
    "debug": "dim",
    "info": "white",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PipelineMonitor:
    """Callback-based monitor that ``TrainingPipeline`` calls at each stage.

    Subclasses override the ``on_*`` hooks to provide visual feedback.
    """

    stages: List[str] = ["extract", "synthesize", "train"]

    def __init__(self) -> None:
        self.current_stage: Optional[str] = None
        self.stage_results: Dict[str, Any] = {}
        self.start_time: float = time.time()
        self._stage_start_times: Dict[str, float] = {}
        self._logs: List[Dict[str, Any]] = []

    # -- callbacks -----------------------------------------------------------

    def on_stage_start(self, stage_name: str) -> None:
        """Called when a pipeline stage begins."""
        self.current_stage = stage_name
        self._stage_start_times[stage_name] = time.time()
        logger.info("Stage started: %s", stage_name)

    def on_progress(self, stage: str, current: int, total: int) -> None:
        """Called to report incremental progress within a stage."""
        logger.debug("Progress [%s]: %d / %d", stage, current, total)

    def on_stage_complete(self, stage_name: str, result: Dict[str, Any]) -> None:
        """Called when a pipeline stage finishes successfully."""
        self.stage_results[stage_name] = result
        elapsed = time.time() - self._stage_start_times.get(stage_name, self.start_time)
        self.stage_results.setdefault(stage_name, {})
        self.stage_results[stage_name]["_elapsed"] = elapsed
        if self.current_stage == stage_name:
            self.current_stage = None
        logger.info("Stage completed: %s (%.2fs)", stage_name, elapsed)

    def on_log(self, message: str, level: str = "info") -> None:
        """Called to emit a log line during pipeline execution."""
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "message": message,
        }
        self._logs.append(entry)
        getattr(logger, level, logger.info)(message)

    def summary(self) -> Dict[str, Any]:
        """Return an overall summary of the pipeline run."""
        total_elapsed = time.time() - self.start_time
        return {
            "total_elapsed": total_elapsed,
            "stages_completed": list(self.stage_results.keys()),
            "stage_results": self.stage_results,
            "log_count": len(self._logs),
        }


# ---------------------------------------------------------------------------
# CLI Dashboard (Rich)
# ---------------------------------------------------------------------------


class CLIDashboard(PipelineMonitor):
    """Rich-powered terminal dashboard for pipeline monitoring.

    Renders progress bars, formatted log lines, and a final summary table
    directly to the terminal.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        super().__init__()
        self.console: Console = console or Console()
        self._progress: Progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self._task_ids: Dict[str, TaskID] = {}
        self._progress_started: bool = False

    # -- helpers -------------------------------------------------------------

    def _ensure_progress_started(self) -> None:
        if not self._progress_started:
            self._progress.start()
            self._progress_started = True

    def _stop_progress(self) -> None:
        if self._progress_started:
            self._progress.stop()
            self._progress_started = False

    @staticmethod
    def _stage_label(stage_name: str) -> str:
        return _STAGE_ICONS.get(stage_name, f"[bold]{stage_name.upper()}[/bold]")

    # -- callbacks -----------------------------------------------------------

    def on_stage_start(self, stage_name: str) -> None:
        super().on_stage_start(stage_name)

        header = Panel(
            Text.from_markup(
                f"  Stage: {self._stage_label(stage_name)}  |  "
                f"Started at {datetime.now().strftime('%H:%M:%S')}"
            ),
            title="[bold]VectorTrainer Pipeline[/bold]",
            border_style="bright_blue",
            expand=False,
        )
        self.console.print()
        self.console.print(header)

    def on_progress(self, stage: str, current: int, total: int) -> None:
        super().on_progress(stage, current, total)

        self._ensure_progress_started()

        if stage not in self._task_ids:
            label = self._stage_label(stage)
            task_id = self._progress.add_task(
                description=f"  {label}",
                total=total,
                completed=current,
            )
            self._task_ids[stage] = task_id
        else:
            task_id = self._task_ids[stage]
            self._progress.update(task_id, completed=current, total=total)

    def on_stage_complete(self, stage_name: str, result: Dict[str, Any]) -> None:
        super().on_stage_complete(stage_name, result)

        # Finish associated progress bar if present
        if stage_name in self._task_ids:
            task_id = self._task_ids[stage_name]
            self._progress.update(task_id, completed=self._progress.tasks[task_id].total)

        elapsed = self.stage_results[stage_name].get("_elapsed", 0.0)

        # Build a compact result string for display
        display_items: List[str] = []
        for key, value in result.items():
            if key.startswith("_"):
                continue
            display_items.append(f"[bold]{key}[/bold]={value}")
        result_str = ", ".join(display_items) if display_items else "OK"

        self.console.print(
            f"  [green]>[/green] {self._stage_label(stage_name)} completed in "
            f"[bold]{elapsed:.2f}s[/bold]  |  {result_str}"
        )

    def on_log(self, message: str, level: str = "info") -> None:
        super().on_log(message, level)
        style = _LEVEL_STYLES.get(level, "white")
        ts = datetime.now().strftime("%H:%M:%S")
        self.console.print(
            f"  [{style}][{ts}] [{level.upper():>8s}] {message}[/{style}]"
        )

    def summary(self) -> Dict[str, Any]:
        result = super().summary()

        self._stop_progress()

        self.console.print()

        # -- Summary table ---------------------------------------------------
        table = Table(
            title="Pipeline Summary",
            show_header=True,
            header_style="bold bright_blue",
            border_style="bright_blue",
            expand=False,
        )
        table.add_column("Stage", style="bold", min_width=14)
        table.add_column("Status", justify="center", min_width=10)
        table.add_column("Elapsed", justify="right", min_width=10)
        table.add_column("Details", min_width=30)

        for stage in self.stages:
            if stage in self.stage_results:
                status = "[green]DONE[/green]"
                elapsed = f"{self.stage_results[stage].get('_elapsed', 0.0):.2f}s"
                details_parts: List[str] = []
                for k, v in self.stage_results[stage].items():
                    if k.startswith("_"):
                        continue
                    details_parts.append(f"{k}={v}")
                details = ", ".join(details_parts) if details_parts else "-"
            elif stage == self.current_stage:
                status = "[yellow]RUNNING[/yellow]"
                elapsed = "-"
                details = "in progress"
            else:
                status = "[dim]PENDING[/dim]"
                elapsed = "-"
                details = "-"
            table.add_row(self._stage_label(stage), status, elapsed, details)

        self.console.print(table)

        # -- Footer ----------------------------------------------------------
        total = result["total_elapsed"]
        footer = Panel(
            Text.from_markup(
                f"Total elapsed: [bold]{total:.2f}s[/bold]  |  "
                f"Stages completed: [bold]{len(result['stages_completed'])}/{len(self.stages)}[/bold]  |  "
                f"Log entries: [bold]{result['log_count']}[/bold]"
            ),
            border_style="green",
            expand=False,
        )
        self.console.print(footer)
        self.console.print()

        return result


# ---------------------------------------------------------------------------
# Web Dashboard (Streamlit)
# ---------------------------------------------------------------------------


class WebDashboard(PipelineMonitor):
    """Streamlit-powered browser dashboard for pipeline monitoring.

    All state is stored internally so that repeated Streamlit reruns can
    reconstruct the UI from the latest snapshot.
    """

    def __init__(self) -> None:
        super().__init__()
        self._progress_state: Dict[str, Dict[str, int]] = {}
        self._st: Optional[Any] = None  # lazy-imported streamlit module

    @property
    def st(self) -> Any:
        """Lazily import streamlit so the module can be loaded without it."""
        if self._st is None:
            try:
                import streamlit as _st  # noqa: N812
                self._st = _st
            except ImportError as exc:
                raise ImportError(
                    "Streamlit is required for WebDashboard. "
                    "Install it with: pip install streamlit"
                ) from exc
        return self._st

    # -- callbacks -----------------------------------------------------------

    def on_stage_start(self, stage_name: str) -> None:
        super().on_stage_start(stage_name)

    def on_progress(self, stage: str, current: int, total: int) -> None:
        super().on_progress(stage, current, total)
        self._progress_state[stage] = {"current": current, "total": total}

    def on_stage_complete(self, stage_name: str, result: Dict[str, Any]) -> None:
        super().on_stage_complete(stage_name, result)
        # Mark progress as 100% for the completed stage
        if stage_name in self._progress_state:
            total = self._progress_state[stage_name]["total"]
            self._progress_state[stage_name]["current"] = total

    def on_log(self, message: str, level: str = "info") -> None:
        super().on_log(message, level)

    # -- rendering -----------------------------------------------------------

    def render(self) -> None:
        """Render the full dashboard using Streamlit components.

        Call this method inside a Streamlit script context.  It reads from
        internal state and writes all UI elements.
        """
        st = self.st

        st.set_page_config(
            page_title="VectorTrainer Dashboard",
            page_icon="::chart_with_upwards_trend::",
            layout="wide",
        )

        st.title("VectorTrainer Pipeline Dashboard")
        st.caption("Self-Evolving AI Loop -- real-time pipeline monitor")

        # -- Top-level metrics -----------------------------------------------
        col1, col2, col3 = st.columns(3)
        completed_count = len(self.stage_results)
        total_elapsed = time.time() - self.start_time

        col1.metric("Stages Completed", f"{completed_count}/{len(self.stages)}")
        col2.metric("Elapsed Time", f"{total_elapsed:.1f}s")
        col3.metric("Log Entries", str(len(self._logs)))

        st.divider()

        # -- Stage status indicators -----------------------------------------
        st.subheader("Stage Status")
        stage_cols = st.columns(len(self.stages))

        for idx, stage in enumerate(self.stages):
            with stage_cols[idx]:
                if stage in self.stage_results:
                    st.success(f"{stage.upper()} -- DONE")
                    elapsed = self.stage_results[stage].get("_elapsed", 0.0)
                    st.metric(f"{stage} time", f"{elapsed:.2f}s")
                elif stage == self.current_stage:
                    st.warning(f"{stage.upper()} -- RUNNING")
                    progress_info = self._progress_state.get(stage, {})
                    current = progress_info.get("current", 0)
                    total = progress_info.get("total", 1)
                    pct = current / total if total > 0 else 0.0
                    st.progress(pct, text=f"{current}/{total}")
                else:
                    st.info(f"{stage.upper()} -- PENDING")

        st.divider()

        # -- Progress bars for all stages ------------------------------------
        st.subheader("Progress")
        for stage in self.stages:
            progress_info = self._progress_state.get(stage)
            if progress_info is not None:
                current = progress_info["current"]
                total = progress_info["total"]
                pct = current / total if total > 0 else 0.0
                st.progress(pct, text=f"{stage}: {current}/{total}")

        st.divider()

        # -- Data visualization placeholders ---------------------------------
        st.subheader("Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("**Vector Distribution**")
            train_result = self.stage_results.get("train", {})
            if "vector_distribution" in train_result:
                st.dataframe(train_result["vector_distribution"])
            else:
                st.info("Vector distribution will appear after training completes.")

        with viz_col2:
            st.markdown("**Training Curves**")
            train_result = self.stage_results.get("train", {})
            if "training_curves" in train_result:
                st.line_chart(train_result["training_curves"])
            else:
                st.info("Training curves will appear after training completes.")

        st.divider()

        # -- Stage result details --------------------------------------------
        st.subheader("Stage Results")
        for stage in self.stages:
            if stage in self.stage_results:
                with st.expander(f"{stage.upper()} results", expanded=False):
                    display_result = {
                        k: v
                        for k, v in self.stage_results[stage].items()
                        if not k.startswith("_")
                    }
                    if display_result:
                        st.json(display_result)
                    else:
                        st.write("No detailed results available.")

        st.divider()

        # -- Log display -----------------------------------------------------
        st.subheader("Logs")
        if self._logs:
            log_lines: List[str] = []
            for entry in self._logs:
                ts = entry["timestamp"]
                lvl = entry["level"].upper()
                msg = entry["message"]
                log_lines.append(f"[{ts}] {lvl:>8s}  {msg}")
            st.code("\n".join(log_lines), language="log")
        else:
            st.info("No log entries yet.")

    def summary(self) -> Dict[str, Any]:
        result = super().summary()

        st = self.st
        st.divider()
        st.subheader("Pipeline Summary")
        st.metric("Total Elapsed", f"{result['total_elapsed']:.2f}s")
        st.metric("Stages Completed", f"{len(result['stages_completed'])}/{len(self.stages)}")
        st.metric("Log Entries", str(result["log_count"]))

        return result


# ---------------------------------------------------------------------------
# Standalone runner functions
# ---------------------------------------------------------------------------


def run_cli_dashboard(
    pipeline: Any,
    function_name: str,
    feedback_pairs: List[Any],
    output_dir: str,
) -> Any:
    """Run the pipeline with CLI dashboard monitoring.

    Parameters
    ----------
    pipeline:
        A ``TrainingPipeline`` (or compatible) instance that exposes a
        ``monitor`` attribute and a ``run()`` method.
    function_name:
        Target function name passed through to the pipeline.
    feedback_pairs:
        List of ``FeedbackPair`` instances (or compatible dicts).
    output_dir:
        Directory for pipeline output artefacts.

    Returns
    -------
    Any
        The result returned by ``pipeline.run()``.
    """
    dashboard = CLIDashboard()
    pipeline.monitor = dashboard

    dashboard.console.print(
        Panel(
            Text.from_markup(
                f"[bold]Function:[/bold] {function_name}  |  "
                f"[bold]Feedback pairs:[/bold] {len(feedback_pairs)}  |  "
                f"[bold]Output:[/bold] {output_dir}"
            ),
            title="[bold bright_blue]VectorTrainer[/bold bright_blue]",
            subtitle="Self-Evolving AI Loop",
            border_style="bright_blue",
        )
    )

    result = pipeline.run(function_name, feedback_pairs, output_dir)
    dashboard.summary()
    return result


def create_streamlit_app(
    pipeline: Any,
    function_name: str,
    feedback_pairs: List[Any],
    output_dir: str,
) -> None:
    """Create and render a Streamlit app for pipeline monitoring.

    Intended to be called from a script executed via::

        streamlit run dashboard.py

    The function wires up a :class:`WebDashboard`, runs the pipeline, and
    renders the final dashboard state.

    Parameters
    ----------
    pipeline:
        A ``TrainingPipeline`` (or compatible) instance.
    function_name:
        Target function name passed through to the pipeline.
    feedback_pairs:
        List of ``FeedbackPair`` instances (or compatible dicts).
    output_dir:
        Directory for pipeline output artefacts.
    """
    import streamlit as st  # noqa: N812

    dashboard = WebDashboard()
    pipeline.monitor = dashboard

    # Use Streamlit session_state to avoid re-running the pipeline on every
    # script rerun.  The pipeline is executed only once; subsequent reruns
    # just re-render the stored dashboard state.
    if "vt_dashboard" not in st.session_state:
        with st.spinner("Running pipeline..."):
            result = pipeline.run(function_name, feedback_pairs, output_dir)
        st.session_state["vt_dashboard"] = dashboard
        st.session_state["vt_result"] = result
    else:
        dashboard = st.session_state["vt_dashboard"]

    dashboard.render()
    dashboard.summary()
