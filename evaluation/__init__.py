"""evaluation — metrics computation and experiment CSV logging."""
from .evaluator import SearchEvaluator, QueryResult
from .logger    import ExperimentLogger

__all__ = ["SearchEvaluator", "QueryResult", "ExperimentLogger"]
