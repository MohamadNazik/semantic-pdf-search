"""
evaluation/evaluator.py
========================
Module 8 — Evaluation Framework

SearchEvaluator computes standard IR metrics over a labelled query set:

    * Precision@K  — fraction of top-K results that are relevant
    * Recall@K     — fraction of known relevant docs retrieved in top-K
    * Average Precision (AP) per query
    * Mean Average Precision (MAP) across all queries
    * Mean Query Response Time

Query set format (queries.json):
    [
      {
        "query_id": "q1",
        "query_text": "machine learning classification",
        "relevant_docs": ["paper_a.pdf", "paper_b.pdf"]
      },
      ...
    ]

`relevant_docs` are matched against result filenames (case-insensitive).
"""

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Holds the search results for one evaluated query."""
    query_id:      str
    query_text:    str
    relevant_docs: List[str]           # ground-truth filenames (lowercase)
    retrieved:     List[str]           # retrieved filenames in ranked order
    scores:        List[float]         # similarity / TF-IDF scores
    query_time_ms: float


@dataclass
class MetricSet:
    """Per-query metrics."""
    query_id:  str
    ap:        float                   # Average Precision
    precisions: Dict[int, float]       # Precision@K for each K in eval_ks
    recalls:    Dict[int, float]       # Recall@K for each K in eval_ks
    query_time_ms: float


# ── SearchEvaluator ───────────────────────────────────────────────────────────

class SearchEvaluator:
    """
    Runs a labelled query set through a search function and computes
    Precision@K, Recall@K, and MAP.

    Parameters
    ----------
    eval_ks      : list of K values to compute Precision@K and Recall@K
    results_dir  : directory where CSV reports are saved
    """

    def __init__(
        self,
        eval_ks:     List[int] = None,
        results_dir: Path = None,
    ):
        self.eval_ks     = eval_ks or [1, 3, 5, 10]
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── query set loading ─────────────────────────────────────────────────────

    @staticmethod
    def load_queries(queries_file: Path) -> List[Dict]:
        """Load the JSON query set; returns an empty list if file missing."""
        queries_file = Path(queries_file)
        if not queries_file.exists():
            logger.warning("Queries file not found: %s", queries_file)
            return []
        with open(queries_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── evaluation entry point ────────────────────────────────────────────────

    def evaluate(
        self,
        queries:       List[Dict],
        search_fn:     Callable[[str, int], Dict],
        top_k:         int,
        model_label:   str = "unknown",
        search_type:   str = "semantic",
    ) -> Dict:
        """
        Run all queries through *search_fn* and compute aggregate metrics.

        Parameters
        ----------
        queries     : list of query dicts from load_queries()
        search_fn   : callable(query_text, top_k) → search result dict
                      (as returned by SemanticSearch.search or KeywordSearch.search)
        top_k       : number of results to retrieve per query
        model_label : label for this run (e.g. "minilm")
        search_type : "semantic" or "keyword"

        Returns
        -------
        dict with aggregate metrics and per-query details
        """
        if not queries:
            logger.warning("No queries to evaluate.")
            return {}

        # Warm up the search function to discard PyTorch/initialization latency
        try:
            logger.info("Warming up search engine to discard initialization latency...")
            search_fn("warmup query", 1)
        except Exception:
            pass

        per_query: List[MetricSet] = []

        for q in queries:
            qid   = q.get("query_id", "?")
            qtext = q.get("query_text", "")
            rel   = [r.lower() for r in q.get("relevant_docs", [])]

            try:
                result = search_fn(qtext, top_k)
            except Exception as exc:
                logger.error("Query %s failed: %s", qid, exc)
                continue

            retrieved = [
                r["filename"].lower()
                for r in result.get("results", [])
            ]
            scores = [r["score"] for r in result.get("results", [])]
            qtime  = result.get("query_time_ms", 0.0)

            ms = self._compute_metrics(qid, rel, retrieved, scores, qtime)
            per_query.append(ms)

            logger.debug(
                "Query %s — AP=%.3f  P@5=%.3f  R@5=%.3f",
                qid, ms.ap,
                ms.precisions.get(5, 0.0),
                ms.recalls.get(5, 0.0),
            )

        if not per_query:
            return {}

        aggregate = self._aggregate(per_query)
        aggregate.update({
            "model":       model_label,
            "search_type": search_type,
            "top_k":       top_k,
            "num_queries": len(per_query),
        })

        return {
            "aggregate":  aggregate,
            "per_query":  [asdict(m) for m in per_query],
        }

    # ── metric computation ────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        query_id:   str,
        relevant:   List[str],
        retrieved:  List[str],
        scores:     List[float],
        query_time: float,
    ) -> MetricSet:
        """Compute AP, Precision@K, Recall@K for one query."""
        precisions: Dict[int, float] = {}
        recalls:    Dict[int, float] = {}

        # Deduplicate retrieved list while preserving rank order for document-level metrics
        seen = set()
        dedup_retrieved = []
        for r in retrieved:
            if r not in seen:
                seen.add(r)
                dedup_retrieved.append(r)

        for k in self.eval_ks:
            top_k_ret = dedup_retrieved[:k]
            hits       = sum(1 for r in top_k_ret if r in relevant)
            precisions[k] = hits / k if k > 0 else 0.0
            recalls[k]    = hits / len(relevant) if relevant else 0.0

        # Average Precision
        ap = self._average_precision(relevant, dedup_retrieved)

        return MetricSet(
            query_id      = query_id,
            ap            = ap,
            precisions    = precisions,
            recalls       = recalls,
            query_time_ms = query_time,
        )

    @staticmethod
    def _average_precision(relevant: List[str], retrieved: List[str]) -> float:
        """Compute AP for a single query."""
        if not relevant:
            return 0.0
        hits       = 0
        sum_prec   = 0.0
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                hits += 1
                sum_prec += hits / i
        return sum_prec / len(relevant)

    def _aggregate(self, per_query: List[MetricSet]) -> Dict:
        """Compute MAP and mean Precision@K / Recall@K over all queries."""
        n = len(per_query)
        result = {}

        result["MAP"] = sum(m.ap for m in per_query) / n

        for k in self.eval_ks:
            result[f"P@{k}"] = sum(
                m.precisions.get(k, 0.0) for m in per_query
            ) / n
            result[f"R@{k}"] = sum(
                m.recalls.get(k, 0.0) for m in per_query
            ) / n

        result["mean_query_time_ms"] = sum(
            m.query_time_ms for m in per_query
        ) / n

        return {k: round(v, 4) for k, v in result.items()}

    # ── CSV export ────────────────────────────────────────────────────────────

    def save_results_csv(
        self,
        eval_output: Dict,
        filename: str = "evaluation_results.csv",
    ) -> Path:
        """Write per-query metrics to a CSV file in results_dir."""
        per_query = eval_output.get("per_query", [])
        if not per_query:
            logger.warning("No per-query data to save.")
            return self.results_dir / filename

        out_path = self.results_dir / filename
        fieldnames = ["query_id", "ap", "query_time_ms"] + [
            f"{m}@{k}"
            for m in ("P", "R")
            for k in self.eval_ks
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in per_query:
                flat = {
                    "query_id":    row["query_id"],
                    "ap":          row["ap"],
                    "query_time_ms": row["query_time_ms"],
                }
                for k in self.eval_ks:
                    flat[f"P@{k}"] = row["precisions"].get(k, 0.0)
                    flat[f"R@{k}"] = row["recalls"].get(k, 0.0)
                writer.writerow(flat)

        logger.info("Evaluation results saved to %s", out_path)
        return out_path
