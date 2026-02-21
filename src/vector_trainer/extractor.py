"""Density-based golden data extraction using graph-theoretic methods.

This module models execution logs as nodes in a weighted graph where edge
weights represent cosine similarity.  High-density nodes correspond to
stable operational clusters (steady state), while low-density SUCCESS nodes
represent novel patterns worth capturing (anomalies).

Mathematical foundation
-----------------------
* **Graph G = (V, E, W)** — V: execution logs, E: edges between similar
  logs, W: cosine similarity weights.
* **Density D(L_i)** = sum_{j in N(L_i)} W_ij — the total edge weight
  incident on node L_i.
* **Epsilon neighbourhood** — only edges with similarity >= (1 - epsilon)
  are retained, controlling the sparsity of G.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .types import (
    DatasetManagerProtocol,
    ExecutionLog,
    GoldenCandidate,
    SelectionStrategy,
)

logger = logging.getLogger(__name__)


class DensityBasedExtractor:
    """Select golden training data via density analysis on the vector graph.

    Parameters
    ----------
    dataset_manager:
        An object satisfying ``DatasetManagerProtocol`` (injected via DI).
    epsilon:
        Neighbour threshold in cosine *distance* space.  Two nodes are
        connected when their cosine similarity >= (1 - epsilon).
    top_k:
        Maximum number of data points to return per selection strategy.
    """

    def __init__(
        self,
        dataset_manager: DatasetManagerProtocol,
        epsilon: float = 0.3,
        top_k: int = 50,
    ) -> None:
        self.dataset_manager = dataset_manager
        self.epsilon = epsilon
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Vector arithmetic
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns a value in [-1, 1].  For normalised embeddings this
        simplifies to the dot product.
        """
        a = np.asarray(v1, dtype=np.float64)
        b = np.asarray(v2, dtype=np.float64)

        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_weighted_graph(
        self, logs: List[ExecutionLog]
    ) -> Dict[str, Dict[str, float]]:
        """Build an adjacency dict for the similarity graph.

        Each ``ExecutionLog`` becomes a **node** (keyed by UUID).  An
        **edge** (u, v) is added with weight W_uv = cosine_similarity(u, v)
        when W_uv >= (1 - epsilon).

        Returns
        -------
        adjacency : dict
            ``{node_uuid: {neighbour_uuid: weight, ...}, ...}``
        """
        similarity_threshold: float = 1.0 - self.epsilon

        # Initialise adjacency list for every node
        adjacency: Dict[str, Dict[str, float]] = {log.uuid: {} for log in logs}

        node_count = len(logs)
        edge_count = 0

        for i in range(node_count):
            for j in range(i + 1, node_count):
                node_i = logs[i]
                node_j = logs[j]

                weight = self._cosine_similarity(node_i.vector, node_j.vector)

                if weight >= similarity_threshold:
                    adjacency[node_i.uuid][node_j.uuid] = weight
                    adjacency[node_j.uuid][node_i.uuid] = weight
                    edge_count += 1

        logger.info(
            "Built weighted graph: |V|=%d, |E|=%d (epsilon=%.3f, threshold=%.3f)",
            node_count,
            edge_count,
            self.epsilon,
            similarity_threshold,
        )
        return adjacency

    # ------------------------------------------------------------------
    # Density computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_density(node_id: str, graph: Dict[str, Dict[str, float]]) -> float:
        """Compute the density of a node in the weighted graph.

        D(L_i) = sum_{j in N(L_i)} W_ij

        A high density indicates the node sits inside a dense cluster of
        similar execution logs.
        """
        neighbours = graph.get(node_id, {})
        return sum(neighbours.values())

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_execution_logs(self, function_name: str) -> List[ExecutionLog]:
        """Retrieve execution logs from VectorWave and convert to dataclasses.

        Uses ``dataset_manager.exec_col.query.fetch_objects()`` to pull
        stored logs, then filters for entries that contain an embedding
        vector (required for graph construction).
        """
        try:
            response = self.dataset_manager.exec_col.query.fetch_objects(
                include_vector=True,
                limit=1000,
            )
        except Exception:
            logger.exception("Failed to fetch execution logs from VectorWave")
            return []

        objects = getattr(response, "objects", [])
        logs: List[ExecutionLog] = []

        for obj in objects:
            properties: Dict[str, Any] = getattr(obj, "properties", {})

            # Filter by target function
            if properties.get("function_name") != function_name:
                continue

            # Extract vector — Weaviate stores vectors in a dict keyed by
            # the vectoriser name; the default key is "default".
            raw_vector = getattr(obj, "vector", None)
            if isinstance(raw_vector, dict):
                vector = raw_vector.get("default")
            elif isinstance(raw_vector, list):
                vector = raw_vector
            else:
                vector = None

            if not vector:
                continue

            log = ExecutionLog(
                uuid=str(obj.uuid),
                function_name=properties.get("function_name", ""),
                return_value=properties.get("return_value") or "",
                status=properties.get("status", ""),
                vector=vector,
                properties=properties,
            )
            logs.append(log)

        logger.info(
            "Fetched %d execution logs with vectors for function '%s'",
            len(logs),
            function_name,
        )
        return logs

    # ------------------------------------------------------------------
    # Centroid helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_centroid(logs: List[ExecutionLog]) -> np.ndarray:
        """Return the centroid (mean vector) of a set of execution logs."""
        matrix = np.array([log.vector for log in logs], dtype=np.float64)
        return matrix.mean(axis=0)

    @staticmethod
    def _distance_to_centroid(vector: List[float], centroid: np.ndarray) -> float:
        """Euclidean distance from a vector to the centroid."""
        return float(np.linalg.norm(np.asarray(vector, dtype=np.float64) - centroid))

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def select_steady_state(self, function_name: str) -> List[GoldenCandidate]:
        """Select high-density nodes representing stable operational clusters.

        Nodes with the highest density D(L_i) sit at the core of dense
        clusters — these represent the *steady state* of the function's
        behaviour and make reliable golden training examples.
        """
        logs = self._fetch_execution_logs(function_name)
        if not logs:
            logger.warning("No execution logs found for steady-state selection")
            return []

        graph = self._build_weighted_graph(logs)
        centroid = self._compute_centroid(logs)

        # Map UUID -> ExecutionLog for quick lookup
        log_index: Dict[str, ExecutionLog] = {log.uuid: log for log in logs}

        # Compute density for every node and sort descending
        densities: List[tuple[str, float]] = [
            (node_id, self._compute_density(node_id, graph)) for node_id in graph
        ]
        densities.sort(key=lambda pair: pair[1], reverse=True)

        # Select top-k highest-density nodes
        candidates: List[GoldenCandidate] = []
        for node_id, density in densities[: self.top_k]:
            if density <= 0.0:
                # Skip isolated nodes (no neighbours within epsilon)
                continue

            log = log_index[node_id]
            candidates.append(
                GoldenCandidate(
                    uuid=log.uuid,
                    strategy=SelectionStrategy.STEADY,
                    density=density,
                    distance_to_centroid=self._distance_to_centroid(
                        log.vector, centroid
                    ),
                    return_value=log.return_value,
                    vector=log.vector,
                )
            )

        logger.info(
            "Steady-state selection: %d candidates from %d nodes",
            len(candidates),
            len(logs),
        )
        return candidates

    def select_anomalies(self, function_name: str) -> List[GoldenCandidate]:
        """Select low-density SUCCESS nodes representing novel patterns.

        Successful executions that lie in sparse regions of the vector space
        are *anomalies* — they indicate new behavioural patterns that the
        model has not yet learned.  Including them in training data improves
        coverage.
        """
        logs = self._fetch_execution_logs(function_name)
        if not logs:
            logger.warning("No execution logs found for anomaly selection")
            return []

        graph = self._build_weighted_graph(logs)
        centroid = self._compute_centroid(logs)

        log_index: Dict[str, ExecutionLog] = {log.uuid: log for log in logs}

        # Only consider nodes that executed successfully
        success_densities: List[tuple[str, float]] = []
        for node_id in graph:
            log = log_index[node_id]
            if log.status != "SUCCESS":
                continue
            density = self._compute_density(node_id, graph)
            success_densities.append((node_id, density))

        # Sort by density ascending — lowest density first (most anomalous)
        success_densities.sort(key=lambda pair: pair[1])

        candidates: List[GoldenCandidate] = []
        for node_id, density in success_densities[: self.top_k]:
            log = log_index[node_id]
            candidates.append(
                GoldenCandidate(
                    uuid=log.uuid,
                    strategy=SelectionStrategy.ANOMALY,
                    density=density,
                    distance_to_centroid=self._distance_to_centroid(
                        log.vector, centroid
                    ),
                    return_value=log.return_value,
                    vector=log.vector,
                )
            )

        logger.info(
            "Anomaly selection: %d candidates from %d SUCCESS nodes",
            len(candidates),
            len(success_densities),
        )
        return candidates

    # ------------------------------------------------------------------
    # Golden data export
    # ------------------------------------------------------------------

    def extract_golden_data(self, function_name: str, output_path: str) -> str:
        """Extract golden data and write ChatML-format JSONL to *output_path*.

        Combines steady-state and anomaly selections, deduplicates by UUID,
        and serialises each candidate as a ChatML message triple::

            {"messages": [
                {"role": "system",    "content": "<system prompt>"},
                {"role": "user",      "content": "<input description>"},
                {"role": "assistant", "content": "<return value>"}
            ]}

        Returns
        -------
        str
            The path the JSONL file was written to (same as *output_path*).
        """
        steady_candidates = self.select_steady_state(function_name)
        anomaly_candidates = self.select_anomalies(function_name)

        # Deduplicate — a node may theoretically appear in both lists
        seen_uuids: set[str] = set()
        all_candidates: List[GoldenCandidate] = []

        for candidate in steady_candidates + anomaly_candidates:
            if candidate.uuid not in seen_uuids:
                seen_uuids.add(candidate.uuid)
                all_candidates.append(candidate)

        logger.info(
            "Total golden candidates for '%s': %d (steady=%d, anomaly=%d, deduplicated=%d)",
            function_name,
            len(all_candidates),
            len(steady_candidates),
            len(anomaly_candidates),
            len(steady_candidates) + len(anomaly_candidates) - len(all_candidates),
        )

        # Ensure the output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        system_prompt = (
            f"You are a function executor for '{function_name}'. "
            "Produce the correct return value given the user's input."
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            for candidate in all_candidates:
                user_content = (
                    f"function_name: {function_name}\n"
                    f"strategy: {candidate.strategy.value}\n"
                    f"density: {candidate.density:.6f}\n"
                    f"distance_to_centroid: {candidate.distance_to_centroid:.6f}"
                )

                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": candidate.return_value or ""},
                    ]
                }

                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Wrote %d ChatML records to %s", len(all_candidates), output_path)
        return output_path
