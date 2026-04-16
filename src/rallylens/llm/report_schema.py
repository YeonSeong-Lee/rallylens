"""Pydantic response schema for the LLM-generated rally report.

Fields use Korean text (the user explicitly selected Korean-only output).
Only JSON-schema-friendly types are used so the schema serializes cleanly
for Gemini `response_schema` structured output: str, int, float, list, and
Literal unions — no tuple, datetime, or custom Enum.

Derived numerics like duration or fps are intentionally not on this schema
— they belong to `MatchMetrics` and are stitched in deterministically by
`render_report_markdown`. Keeping the LLM schema minimal reduces token
waste and eliminates the risk of the model hallucinating numbers it was
supposed to copy from input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PlayerInsight(BaseModel):
    """Per-player qualitative analysis written by the LLM."""

    model_config = ConfigDict(frozen=True)

    track_id: int
    summary_ko: str
    strengths_ko: list[str]
    weaknesses_ko: list[str]


class ReportOutput(BaseModel):
    """Top-level structured LLM response for a rally analysis report."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    headline_ko: str
    summary_ko: str
    key_observations_ko: list[str]
    player_analysis: list[PlayerInsight]
    tactical_suggestions_ko: list[str]
