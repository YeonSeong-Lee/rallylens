"""Vertex AI Gemini report generation + deterministic markdown rendering.

`generate_report` is the only place in the codebase that calls Gemini. It
takes a deterministic `MatchMetrics` bundle, renders a Korean analysis via
structured output, and returns a typed `ReportOutput`. The client is
injected for testability; absent a client the function builds one from
environment variables via `create_vertex_client`.

`render_report_markdown` composes the final `report.md` from the LLM output
plus the source metrics. It runs no LLM — it is a pure f-string template
so the Markdown shape is stable and cheap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from google import genai
from google.auth.exceptions import DefaultCredentialsError
from google.genai import types
from google.genai.errors import APIError

from rallylens.analysis.metrics import MatchMetrics, PlayerMetrics
from rallylens.common import get_logger
from rallylens.llm.prompt import SYSTEM_PROMPT_KO
from rallylens.llm.report_schema import PlayerInsight, ReportOutput
from rallylens.llm.vertex_client import create_vertex_client

_log = get_logger(__name__)

DEFAULT_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_TEMPERATURE: Final[float] = 0.4
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 4096


def generate_report(
    metrics: MatchMetrics,
    *,
    client: genai.Client | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> ReportOutput:
    """Generate a structured Korean rally report via Vertex AI Gemini.

    Wraps credential errors from the underlying ADC flow into a RuntimeError
    with an actionable message so the CLI can surface it cleanly.
    """
    client = client or create_vertex_client()
    user_text = (
        "Input MatchMetrics:\n"
        + metrics.model_dump_json(indent=2)
        + "\n\n이 수치를 바탕으로 한국어 랠리 분석 보고서를 JSON으로 생성하세요."
    )
    _log.info(
        "generate_report: model=%s players=%d hits=%d",
        model,
        len(metrics.players),
        metrics.shuttle.total_hit_events,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_KO,
                response_mime_type="application/json",
                response_schema=ReportOutput,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "Vertex AI credentials are not configured. Run "
            "`gcloud auth application-default login` and verify "
            "GOOGLE_CLOUD_PROJECT is set in .env."
        ) from exc
    except APIError as exc:
        raise RuntimeError(f"Vertex AI API error: {exc}") from exc

    if isinstance(response.parsed, ReportOutput):
        return response.parsed
    if response.text is None:
        raise RuntimeError(
            "Vertex AI returned no parsed ReportOutput and no text body. "
            "Check the model ID and your schema."
        )
    return ReportOutput.model_validate_json(response.text)


def render_report_markdown(
    report: ReportOutput,
    metrics: MatchMetrics,
    *,
    model: str,
    md_path: Path,
    court_diagram_path: Path | None,
    heatmap_path: Path | None,
) -> str:
    """Compose the final `report.md` deterministically from report + metrics."""
    sections: list[str] = [
        f"# {report.headline_ko}",
        _summary_table(metrics),
        _section("## 경기 요약", [report.summary_ko]),
        _section("## 핵심 관찰", _bullet_list(report.key_observations_ko)),
    ]
    if court_diagram_path is not None:
        sections.append(_image_section(
            "## 코트 이동 다이어그램", "코트 이동 다이어그램", md_path, court_diagram_path,
        ))
    if heatmap_path is not None:
        sections.append(_image_section(
            "## 히트맵", "선수·셔틀 히트맵", md_path, heatmap_path,
        ))
    sections.append(_player_analysis_section(report, metrics))
    sections.append(_section("## 전술 제안", _bullet_list(report.tactical_suggestions_ko)))
    sections.append(
        f"---\n생성 모델: `{model}` · metrics schema v{metrics.schema_version} · "
        f"report schema v{report.schema_version}"
    )
    return "\n\n".join(sections) + "\n"


def _section(heading: str, body: list[str]) -> str:
    return heading + "\n" + "\n".join(body)


def _image_section(heading: str, alt: str, md_path: Path, target: Path) -> str:
    rel = os.path.relpath(str(target), start=str(md_path.parent))
    return _section(heading, [f"![{alt}]({rel})"])


def _player_analysis_section(report: ReportOutput, metrics: MatchMetrics) -> str:
    insights_by_id = {i.track_id: i for i in report.player_analysis}
    blocks: list[str] = ["## 선수별 분석"]
    for player in metrics.players:
        block = [f"### {player.track_id}번 선수", _player_stats_table(player)]
        insight = insights_by_id.get(player.track_id)
        if insight is not None:
            block.extend([
                "",
                insight.summary_ko,
                "",
                "**강점**",
                *_bullet_list(insight.strengths_ko),
                "",
                "**개선 과제**",
                *_bullet_list(insight.weaknesses_ko),
            ])
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _summary_table(metrics: MatchMetrics) -> str:
    return (
        "| 항목 | 값 |\n"
        "|---|---|\n"
        f"| 영상 ID | `{metrics.video_id}` |\n"
        f"| 길이 | {metrics.duration_seconds:.1f} 초 ({metrics.frame_count} 프레임) |\n"
        f"| FPS | {metrics.fps:.1f} |\n"
        f"| 총 접촉 추정 | {metrics.shuttle.total_hit_events} 회 |\n"
        f"| 추정 랠리 수 | {metrics.shuttle.rally_count} |\n"
        f"| 가장 긴 랠리 | {metrics.shuttle.longest_rally_shots} 샷 |\n"
        f"| 평균 셔틀 속도 | {metrics.shuttle.avg_shuttle_speed_mps:.2f} m/s |"
    )


def _player_stats_table(player: PlayerMetrics) -> str:
    return (
        "| 항목 | 값 |\n"
        "|---|---|\n"
        f"| 감지 프레임 | {player.detection_frame_count} |\n"
        f"| 총 이동 거리 | {player.total_distance_m:.2f} m |\n"
        f"| 평균 / 최대 속도 | "
        f"{player.avg_speed_mps:.2f} / {player.max_speed_mps:.2f} m/s |\n"
        f"| 코트 커버리지 | {player.convex_hull_area_m2:.2f} m² |\n"
        f"| 샷 수 | {player.shot_count} |\n"
        f"| 전/중/후 비율 | "
        f"{player.front_third_pct:.0%} / {player.mid_third_pct:.0%} / "
        f"{player.back_third_pct:.0%} |\n"
        f"| 좌/중/우 비율 | "
        f"{player.left_third_pct:.0%} / {player.center_third_pct:.0%} / "
        f"{player.right_third_pct:.0%} |"
    )


def _bullet_list(items: list[str]) -> list[str]:
    if not items:
        return ["_(데이터 없음)_"]
    return [f"- {item}" for item in items]


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "PlayerInsight",
    "ReportOutput",
    "generate_report",
    "render_report_markdown",
]
