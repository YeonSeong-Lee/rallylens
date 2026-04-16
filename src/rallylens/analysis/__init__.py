"""Deterministic match-level metrics derived from pipeline artifacts.

This layer produces a compact numerical summary of a match without any LLM
in the loop. The result is consumed by `rallylens.llm` for report generation
but is equally useful to dashboards, notebooks, or quality-eval harnesses.
"""

from rallylens.analysis.metrics import (
    MatchMetrics,
    PlayerMetrics,
    ShuttleMetrics,
    compute_match_metrics,
)

__all__ = [
    "MatchMetrics",
    "PlayerMetrics",
    "ShuttleMetrics",
    "compute_match_metrics",
]
