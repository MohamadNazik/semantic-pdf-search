"""
evaluation/logger.py
=====================
Module 9 — Experiment Logging

ExperimentLogger writes structured CSV records for every indexing and
search experiment so results can be compared across models and runs
in a spreadsheet or analysis notebook.

Two log files are maintained:
    experiments.csv  — one row per index-build / experiment run
    search_history.csv — one row per interactive search query
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── ExperimentLogger ──────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Appends structured records to CSV log files.

    Parameters
    ----------
    logs_dir : directory where CSV files are written
    """

    EXPERIMENT_COLUMNS = [
        "timestamp",
        "run_id",
        "model_key",
        "model_name",
        "search_type",         # "semantic" | "keyword"
        "num_documents",
        "num_chunks",
        "embedding_dim",
        "indexing_time_s",
        "embedding_time_s",
        "total_build_time_s",
        "index_size_vectors",
        "use_chunking",
        "chunk_size",
        "chunk_overlap",
        "notes",
    ]

    SEARCH_COLUMNS = [
        "timestamp",
        "run_id",
        "model_key",
        "search_type",
        "query",
        "top_k",
        "query_time_ms",
        "num_results",
    ]

    EVAL_SUMMARY_COLUMNS = [
        "timestamp",
        "run_id",
        "model_key",
        "search_type",
        "top_k",
        "num_queries",
        "MAP",
        "P@1", "P@3", "P@5", "P@10",
        "R@1", "R@3", "R@5", "R@10",
        "mean_query_time_ms",
        "notes",
    ]

    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._exp_path   = self.logs_dir / "experiments.csv"
        self._search_path = self.logs_dir / "search_history.csv"
        self._eval_path  = self.logs_dir / "eval_summary.csv"

        self._ensure_headers()

    # ── public logging methods ────────────────────────────────────────────────

    def log_experiment(
        self,
        run_id:            str,
        model_key:         str,
        model_name:        str,
        search_type:       str,
        num_documents:     int,
        num_chunks:        int,
        embedding_dim:     int,
        indexing_time_s:   float,
        embedding_time_s:  float,
        use_chunking:      bool,
        chunk_size:        int,
        chunk_overlap:     int,
        index_size_vectors: int,
        notes:             str = "",
    ) -> None:
        """Record one index-build experiment."""
        row = {
            "timestamp":           self._now(),
            "run_id":              run_id,
            "model_key":           model_key,
            "model_name":          model_name,
            "search_type":         search_type,
            "num_documents":       num_documents,
            "num_chunks":          num_chunks,
            "embedding_dim":       embedding_dim,
            "indexing_time_s":     round(indexing_time_s, 3),
            "embedding_time_s":    round(embedding_time_s, 3),
            "total_build_time_s":  round(indexing_time_s + embedding_time_s, 3),
            "index_size_vectors":  index_size_vectors,
            "use_chunking":        use_chunking,
            "chunk_size":          chunk_size,
            "chunk_overlap":       chunk_overlap,
            "notes":               notes,
        }
        self._append(self._exp_path, self.EXPERIMENT_COLUMNS, row)
        logger.info("Experiment logged — run_id=%s  model=%s", run_id, model_key)

    def log_search(
        self,
        run_id:        str,
        model_key:     str,
        search_type:   str,
        query:         str,
        top_k:         int,
        query_time_ms: float,
        num_results:   int,
    ) -> None:
        """Record one search query."""
        row = {
            "timestamp":     self._now(),
            "run_id":        run_id,
            "model_key":     model_key,
            "search_type":   search_type,
            "query":         query,
            "top_k":         top_k,
            "query_time_ms": round(query_time_ms, 1),
            "num_results":   num_results,
        }
        self._append(self._search_path, self.SEARCH_COLUMNS, row)

    def log_eval_summary(
        self,
        run_id:     str,
        aggregate:  Dict[str, Any],
        notes:      str = "",
    ) -> None:
        """Record aggregate evaluation metrics for one experiment run."""
        row = {
            "timestamp": self._now(),
            "run_id":    run_id,
            "notes":     notes,
            **aggregate,
        }
        self._append(self._eval_path, self.EVAL_SUMMARY_COLUMNS, row)
        logger.info(
            "Eval summary logged — run_id=%s  MAP=%.4f",
            run_id, aggregate.get("MAP", 0.0),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ensure_headers(self) -> None:
        """Write CSV headers if the files do not yet exist."""
        for path, columns in [
            (self._exp_path,    self.EXPERIMENT_COLUMNS),
            (self._search_path, self.SEARCH_COLUMNS),
            (self._eval_path,   self.EVAL_SUMMARY_COLUMNS),
        ]:
            if not path.exists():
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=columns).writeheader()

    @staticmethod
    def _append(path: Path, columns: list, row: Dict) -> None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writerow(row)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def make_run_id(model_key: str) -> str:
        """Generate a unique run identifier: model_key + UTC timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_key}_{ts}"
