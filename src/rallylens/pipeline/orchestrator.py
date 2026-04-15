"""End-to-end orchestration: download → segment → track → events → heatmaps → report."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from rallylens.analysis.events import RallyStats
from rallylens.common import (
    OUTPUTS_DEMO_DIR,
    OVERLAYS_DIR,
    RALLIES_DIR,
    REPORTS_DIR,
    ensure_dir,
    get_logger,
)
from rallylens.ingest.downloader import download_video
from rallylens.llm.report_generator import generate_match_report
from rallylens.pipeline.events import run_events_pipeline
from rallylens.pipeline.heatmaps import render_match_heatmaps
from rallylens.pipeline.io import save_player_detections, save_video_meta
from rallylens.pipeline.shuttle import run_shuttle_pipeline
from rallylens.preprocess.rally_segmenter import segment_rallies
from rallylens.vision.detect_track import TrackerName, detect_and_track_players
from rallylens.vision.kalman_tracker import ShuttleTrackPoint
from rallylens.viz.overlay import render_overlay

_log = get_logger(__name__)


@dataclass
class PipelineResult:
    video_id: str
    rally_count: int
    stats_list: list[RallyStats]
    heatmap_path: Path | None
    report_path: Path | None
    report_error: str | None


def run_full_pipeline(
    url: str,
    *,
    tracker: TrackerName = "bytetrack",
    with_shuttle: bool = True,
    with_report: bool = True,
) -> PipelineResult:
    """Run the full match analysis pipeline end-to-end for a single URL.

    Raises:
        RuntimeError: when segmentation produces zero rallies.

    The report step is best-effort: failures are captured in `report_error` so
    the caller can decide how to surface them instead of crashing the pipeline.
    """
    meta = download_video(url)
    save_video_meta(meta)

    rallies = segment_rallies(
        meta.source_path,
        ensure_dir(RALLIES_DIR / meta.video_id),
    )
    if not rallies:
        raise RuntimeError("no rallies detected")

    shuttle_tracks: dict[int, list[ShuttleTrackPoint]] = {}
    for rally in rallies:
        player_detections = detect_and_track_players(rally.path, tracker=tracker)
        save_player_detections(player_detections, meta.video_id, rally.path.stem)

        shuttle_track: list[ShuttleTrackPoint] = []
        if with_shuttle:
            shuttle_track = run_shuttle_pipeline(rally.path, meta.video_id)
        shuttle_tracks[rally.index] = shuttle_track

        overlay_out = (
            ensure_dir(OVERLAYS_DIR / meta.video_id)
            / f"rally_{rally.index:03d}_overlay.mp4"
        )
        render_overlay(
            rally.path,
            player_detections,
            overlay_out,
            shuttle_track=shuttle_track or None,
        )

    stats_list = run_events_pipeline(meta.video_id, rallies, shuttle_tracks)

    heatmap_path = render_match_heatmaps(meta.video_id)
    if heatmap_path is not None:
        ensure_dir(OUTPUTS_DEMO_DIR)
        demo_copy = OUTPUTS_DEMO_DIR / f"{meta.video_id}_heatmaps.png"
        demo_copy.write_bytes(heatmap_path.read_bytes())

    report_path: Path | None = None
    report_error: str | None = None
    if with_report and stats_list:
        try:
            report_path = ensure_dir(REPORTS_DIR / meta.video_id) / "match_report.md"
            generate_match_report(meta, stats_list, report_path)
        except Exception as exc:  # noqa: BLE001 — best-effort, reason captured
            report_error = f"{type(exc).__name__}: {exc}"
            _log.error("report generation failed:\n%s", traceback.format_exc())
            report_path = None

    return PipelineResult(
        video_id=meta.video_id,
        rally_count=len(rallies),
        stats_list=stats_list,
        heatmap_path=heatmap_path,
        report_path=report_path,
        report_error=report_error,
    )
