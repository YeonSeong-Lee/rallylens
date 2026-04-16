"""Orchestration for the `rallylens report` pipeline.

Loads the three CV artifacts (player detections, shuttle track, court
corners), computes deterministic match metrics, optionally renders the
court diagram GIF so the report can embed it, then calls the Vertex AI
report generator and writes the outputs.

The Vertex AI SDK is imported lazily to keep `rallylens --help` and
non-report subcommands free of `google.genai` loading cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rallylens.analysis.metrics import compute_match_metrics
from rallylens.common import ensure_dir, get_logger, read_video_properties
from rallylens.pipeline.io import (
    load_court_corners,
    load_player_detections,
    load_shuttle_track,
    report_markdown_path,
    save_match_metrics,
    save_report,
    viz_court_path,
)
from rallylens.viz import render_viz_court

_log = get_logger(__name__)

_GENAI_MISSING_HINT = (
    "`google-genai` is not installed. Install the optional report "
    "dependency with `uv sync --extra report`, or pass `--metrics-only` "
    "to skip the LLM call and only compute metrics.json."
)


@dataclass
class ReportResult:
    """File paths produced by a `run_report_pipeline` invocation."""

    video_id: str
    metrics_path: Path
    report_json_path: Path | None
    report_md_path: Path | None
    court_gif_path: Path | None


def run_report_pipeline(
    video_path: Path,
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.4,
    metrics_only: bool = False,
    skip_viz: bool = False,
) -> ReportResult:
    """Execute the full report pipeline for a single video.

    Requires detections + court corners to already exist on disk (from
    `rallylens detect` and `rallylens calibrate`). Shuttle track is
    optional — if missing, the shuttle metrics fall back to zeros.
    """
    video_id = video_path.stem

    detections = load_player_detections(video_id)
    if not detections:
        raise FileNotFoundError(
            f"player detections missing for {video_id!r} — "
            f"run `rallylens detect {video_path}` first"
        )
    shuttle_track = load_shuttle_track(video_id)
    try:
        corners = load_court_corners(video_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"court corners missing for {video_id!r} — "
            f"run `rallylens calibrate {video_path}` first"
        ) from exc

    props = read_video_properties(video_path)
    metrics = compute_match_metrics(
        detections, shuttle_track, corners, props, video_id
    )
    m_path = save_match_metrics(metrics, video_id)
    _log.info("wrote metrics: %s", m_path)

    gif_path: Path | None = None
    if not skip_viz:
        gif_path = _ensure_artifact(
            "court diagram",
            viz_court_path(video_id),
            lambda path: render_viz_court(
                detections, shuttle_track, corners, path, fps=props.fps
            ),
        )

    if metrics_only:
        return ReportResult(
            video_id=video_id,
            metrics_path=m_path,
            report_json_path=None,
            report_md_path=None,
            court_gif_path=gif_path,
        )

    generate_report, render_report_markdown = _load_llm()

    report = generate_report(metrics, model=model, temperature=temperature)
    r_json_path = save_report(report, video_id)

    md_path = report_markdown_path(video_id)
    ensure_dir(md_path.parent)
    md = render_report_markdown(
        report,
        metrics,
        model=model,
        md_path=md_path,
        court_gif_path=gif_path,
    )
    md_path.write_text(md, encoding="utf-8")
    _log.info("wrote report: %s and %s", r_json_path, md_path)

    return ReportResult(
        video_id=video_id,
        metrics_path=m_path,
        report_json_path=r_json_path,
        report_md_path=md_path,
        court_gif_path=gif_path,
    )


def _ensure_artifact(
    label: str,
    path: Path,
    render: Callable[[Path], Path],
) -> Path | None:
    """Return `path` if it exists, otherwise render it, logging any failure."""
    if path.exists():
        _log.info("%s already exists: %s", label, path)
        return path
    try:
        return render(path)
    except (OSError, RuntimeError, ValueError) as exc:
        _log.warning("%s render failed, continuing without: %s", label, exc)
        return None


def _load_llm() -> tuple[Any, Any]:
    """Lazy-load the Vertex AI report functions.

    `google-genai` is an optional dependency, so the actual import happens
    here — at the moment the LLM is needed — rather than at module load.
    A missing install is translated into an actionable `RuntimeError` the
    CLI can surface via `click.ClickException`.
    """
    try:
        from rallylens.llm.report import generate_report, render_report_markdown
    except ModuleNotFoundError as exc:
        raise RuntimeError(_GENAI_MISSING_HINT) from exc
    return generate_report, render_report_markdown
