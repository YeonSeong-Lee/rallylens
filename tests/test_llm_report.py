"""Unit tests for rallylens.llm.report.

Uses a fake Gemini client injected via the `client=` parameter of
`generate_report` so no network call is made. The client records the
`model`, `config`, and `contents` of each call for assertion.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rallylens.analysis.metrics import (
    MatchMetrics,
    PlayerMetrics,
    ShuttleMetrics,
    compute_match_metrics,
)
from rallylens.domain.video import VideoProperties
from rallylens.llm.prompt import SYSTEM_PROMPT_KO
from rallylens.llm.report import (
    DEFAULT_MODEL,
    generate_report,
    render_report_markdown,
)
from rallylens.llm.report_schema import PlayerInsight, ReportOutput
from rallylens.vision.court_detector import CourtCorners


def _make_sample_report() -> ReportOutput:
    return ReportOutput(
        headline_ko="랠리 분석 샘플",
        summary_ko="1번 선수와 2번 선수 간 활발한 랠리가 관찰되었습니다.",
        key_observations_ko=[
            "1번 선수는 전위에서 주도권을 잡았습니다.",
            "2번 선수는 후위 드라이브가 강했습니다.",
            "두 선수 모두 좌측 라인 활용도가 높았습니다.",
        ],
        player_analysis=[
            PlayerInsight(
                track_id=1,
                summary_ko="네트 앞 공세적 포지셔닝이 돋보였습니다.",
                strengths_ko=["전위 주도권", "빠른 리시브"],
                weaknesses_ko=["후위 커버 부족"],
            ),
            PlayerInsight(
                track_id=2,
                summary_ko="후위 안정감이 높은 선수입니다.",
                strengths_ko=["긴 랠리 유지력"],
                weaknesses_ko=["전위 대응 속도"],
            ),
        ],
        tactical_suggestions_ko=[
            "1번 선수는 후위 풋워크 훈련을 권장합니다.",
            "2번 선수는 네트 앞 대응 드릴을 추가하세요.",
            "양 선수 모두 우측 라인 커버리지를 늘려야 합니다.",
        ],
    )


def _make_metrics() -> MatchMetrics:
    return MatchMetrics(
        video_id="fixture_clip",
        fps=30.0,
        duration_seconds=10.0,
        frame_count=300,
        players=[
            PlayerMetrics(
                track_id=1,
                detection_frame_count=250,
                total_distance_m=34.5,
                avg_speed_mps=2.1,
                max_speed_mps=5.8,
                convex_hull_area_m2=14.3,
                shot_count=5,
                front_third_pct=0.6,
                mid_third_pct=0.3,
                back_third_pct=0.1,
                left_third_pct=0.5,
                center_third_pct=0.3,
                right_third_pct=0.2,
            ),
            PlayerMetrics(
                track_id=2,
                detection_frame_count=240,
                total_distance_m=28.2,
                avg_speed_mps=1.9,
                max_speed_mps=4.7,
                convex_hull_area_m2=12.1,
                shot_count=4,
                front_third_pct=0.2,
                mid_third_pct=0.35,
                back_third_pct=0.45,
                left_third_pct=0.3,
                center_third_pct=0.4,
                right_third_pct=0.3,
            ),
        ],
        shuttle=ShuttleMetrics(
            total_hit_events=9,
            rally_count=2,
            longest_rally_shots=6,
            avg_rally_shots=4.5,
            avg_inter_hit_seconds=0.8,
            avg_shuttle_speed_mps=11.2,
            max_shuttle_speed_mps=18.3,
        ),
    )


# ---------------------------------------------------------------------------
# Fake Gemini client
#
# The real SDK exposes `client.models.generate_content(...)` returning an
# object with `parsed` and `text` attributes. Tests mimic that shape via
# nested `SimpleNamespace` so they can both preconfigure the response
# (parsed and/or text) and inspect the recorded calls.
# ---------------------------------------------------------------------------


def _make_fake_client(
    parsed: ReportOutput | None = None,
    text: str | None = None,
) -> SimpleNamespace:
    recorded: list[dict[str, Any]] = []

    def generate_content(*, model: str, contents: Any, config: Any) -> SimpleNamespace:
        recorded.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(parsed=parsed, text=text)

    models = SimpleNamespace(generate_content=generate_content, recorded=recorded)
    return SimpleNamespace(models=models)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_passes_schema_and_system_prompt() -> None:
    fake = _make_fake_client(parsed=_make_sample_report())
    metrics = _make_metrics()

    result = generate_report(metrics, client=fake)  # type: ignore[arg-type]

    assert isinstance(result, ReportOutput)
    assert result.headline_ko == "랠리 분석 샘플"
    assert len(fake.models.recorded) == 1
    call = fake.models.recorded[0]
    assert call["model"] == DEFAULT_MODEL
    config = call["config"]
    assert config.system_instruction == SYSTEM_PROMPT_KO
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ReportOutput
    # Thinking disabled for deterministic JSON output
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0
    # User contents contain the metrics payload
    assert "fixture_clip" in call["contents"]


def test_generate_report_prefers_parsed_over_text_fallback() -> None:
    sample = _make_sample_report()
    fake = _make_fake_client(parsed=sample)
    result = generate_report(_make_metrics(), client=fake)  # type: ignore[arg-type]
    assert result == sample


def test_generate_report_falls_back_to_text_when_parsed_is_none() -> None:
    sample = _make_sample_report()
    fake = _make_fake_client(parsed=None, text=sample.model_dump_json())
    result = generate_report(_make_metrics(), client=fake)  # type: ignore[arg-type]
    assert result == sample


def test_generate_report_raises_when_no_parsed_and_no_text() -> None:
    fake = _make_fake_client(parsed=None, text=None)
    with pytest.raises(RuntimeError, match="no parsed"):
        generate_report(_make_metrics(), client=fake)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Optional-dependency plumbing
# ---------------------------------------------------------------------------


def test_load_llm_returns_callables_when_genai_installed() -> None:
    """In the project's venv google-genai is installed (see pyproject extras),
    so _load_llm should return the two real functions."""
    from rallylens.pipeline.report import _load_llm

    gen, render = _load_llm()
    assert callable(gen)
    assert callable(render)


def test_genai_missing_hint_mentions_install_and_metrics_only() -> None:
    """Contract: the error surfaced to CLI users must tell them both how to
    install and how to skip the LLM entirely."""
    from rallylens.pipeline.report import _GENAI_MISSING_HINT

    assert "uv sync --extra report" in _GENAI_MISSING_HINT
    assert "--metrics-only" in _GENAI_MISSING_HINT


def test_llm_package_import_does_not_load_google_genai() -> None:
    """Regression guard: importing rallylens.llm must not pull in google.genai
    so that rallylens.pipeline.io (which imports ReportOutput) stays usable
    without the optional `report` extra installed."""
    import importlib

    # rallylens.llm is likely already imported by other tests; reload it to
    # re-run its __init__.py under the assertion.
    import rallylens.llm

    importlib.reload(rallylens.llm)

    # The package itself must expose only the pure-pydantic schema.
    assert hasattr(rallylens.llm, "ReportOutput")
    assert hasattr(rallylens.llm, "PlayerInsight")
    assert not hasattr(rallylens.llm, "generate_report")
    assert not hasattr(rallylens.llm, "render_report_markdown")


# ---------------------------------------------------------------------------
# render_report_markdown
# ---------------------------------------------------------------------------


def test_render_report_markdown_contains_all_sections(tmp_path: Path) -> None:
    report = _make_sample_report()
    metrics = _make_metrics()
    md_path = tmp_path / "reports" / "fixture_clip" / "report.md"
    gif_path = tmp_path / "viz" / "fixture_clip" / "viz_court.gif"
    md = render_report_markdown(
        report,
        metrics,
        model="gemini-2.5-flash",
        md_path=md_path,
        court_gif_path=gif_path,
    )

    assert "# 랠리 분석 샘플" in md
    assert "## 경기 요약" in md
    assert "## 핵심 관찰" in md
    assert "## 코트 이동 다이어그램" in md
    assert "## 선수별 분석" in md
    assert "## 전술 제안" in md
    assert "생성 모델: `gemini-2.5-flash`" in md

    # Metrics numbers stitched into tables (not hallucinated by LLM)
    assert "34.50 m" in md or "34.5" in md  # player 1 distance (table uses .2f)
    assert "9 회" in md  # total hit events
    assert "fixture_clip" in md
    # Images embedded with relative paths (reports/<id>/ → viz/<id>/)
    assert "../../viz/fixture_clip/viz_court.gif" in md


def test_render_report_markdown_skips_images_when_paths_are_none(
    tmp_path: Path,
) -> None:
    report = _make_sample_report()
    metrics = _make_metrics()
    md_path = tmp_path / "reports" / "fixture_clip" / "report.md"

    md = render_report_markdown(
        report,
        metrics,
        model="gemini-2.5-flash",
        md_path=md_path,
        court_gif_path=None,
    )

    assert "## 코트 이동 다이어그램" not in md
    assert "viz_court.gif" not in md


def test_render_report_markdown_handles_missing_player_insight(
    tmp_path: Path,
) -> None:
    metrics = _make_metrics()
    # LLM returned insight only for track 1 — track 2 should still render stats
    partial_report = ReportOutput(
        headline_ko="부분 보고서",
        summary_ko="일부 분석",
        key_observations_ko=["단일 관찰"],
        player_analysis=[
            PlayerInsight(
                track_id=1,
                summary_ko="1번 요약",
                strengths_ko=["강점"],
                weaknesses_ko=["약점"],
            ),
        ],
        tactical_suggestions_ko=["제안 1"],
    )
    md_path = tmp_path / "reports" / "fixture_clip" / "report.md"
    md = render_report_markdown(
        partial_report,
        metrics,
        model="gemini-2.5-flash",
        md_path=md_path,
        court_gif_path=None,
    )
    assert "### 1번 선수" in md
    assert "### 2번 선수" in md
    # Player 2 has no insight — stats present but no LLM prose
    assert "1번 요약" in md


def test_integration_with_real_metrics_fixture(tmp_path: Path) -> None:
    """End-to-end: compute_match_metrics → generate_report → render."""
    # Build a minimal but real metrics artifact via compute_match_metrics
    corners = CourtCorners(
        top_left=(60.0, 60.0),
        top_right=(670.0, 60.0),
        bottom_left=(60.0, 1400.0),
        bottom_right=(670.0, 1400.0),
    )
    props = VideoProperties(fps=30.0, width=1000, height=800, frame_count=300)
    metrics = compute_match_metrics(
        detections=[],
        shuttle_track=[],
        corners=corners,
        video_props=props,
        video_id="empty_clip",
    )
    fake = _make_fake_client(parsed=_make_sample_report())
    report = generate_report(metrics, client=fake)  # type: ignore[arg-type]

    md_path = tmp_path / "report.md"
    md = render_report_markdown(
        report,
        metrics,
        model="gemini-2.5-flash",
        md_path=md_path,
        court_gif_path=None,
    )
    assert "empty_clip" in md
