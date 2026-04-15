"""Match-level heatmap rendering pipeline stage."""

from __future__ import annotations

from pathlib import Path

from rallylens.analysis.heatmap import render_heatmaps
from rallylens.common import (
    CALIBRATION_DIR,
    HEATMAPS_DIR,
    RALLIES_DIR,
    ensure_dir,
    get_logger,
    read_video_properties,
)
from rallylens.pipeline.io import (
    detections_path,
    load_all_stats,
    load_homography,
    load_player_detections,
    load_shuttle_track,
    shuttle_track_path,
)
from rallylens.preprocess.rally_segmenter import RallyClip, load_manifest
from rallylens.vision.detect_track import Detection
from rallylens.vision.kalman_tracker import ShuttleTrackPoint

_log = get_logger(__name__)


def _rally_center_proxy(rally_path: Path) -> Detection:
    """Fallback placeholder for the player heatmap when real detections are absent.

    Used when `detect` / `run` has not been invoked yet, so we can still render
    a heatmap without re-running YOLO — at the cost of player heatmap accuracy.
    """
    props = read_video_properties(rally_path)
    w = float(props.width)
    h = float(props.height)
    return Detection(
        frame_idx=0,
        bbox_xyxy=(w * 0.25, h * 0.25, w * 0.75, h * 0.75),
        confidence=1.0,
        keypoints_xy=[],
        keypoints_conf=[],
    )


def _collect_player_detections(
    video_id: str, rallies: list[RallyClip]
) -> list[Detection]:
    """Prefer cached per-rally detections; fall back to a single center proxy."""
    real: list[Detection] = []
    missing_rallies: list[RallyClip] = []
    for rally in rallies:
        if detections_path(video_id, rally.path.stem).exists():
            real.extend(load_player_detections(video_id, rally.path.stem))
        else:
            missing_rallies.append(rally)

    if missing_rallies and not real:
        _log.warning(
            "no cached player detections for %s — using center-proxy heatmap. "
            "Run `rallylens detect` or `rallylens run` first for an accurate player heatmap.",
            video_id,
        )
        return [_rally_center_proxy(r.path) for r in missing_rallies]

    if missing_rallies:
        _log.warning(
            "player detections missing for %d rally clips — heatmap only covers "
            "the rallies with cached detections",
            len(missing_rallies),
        )
    return real


def render_match_heatmaps(video_id: str) -> Path | None:
    """Render heatmaps.png from cached rally manifest + stats + shuttle tracks.

    Returns None if rallies or stats have not been computed yet — callers can
    translate that into a user-facing error.
    """
    rallies = load_manifest(RALLIES_DIR / video_id)
    if not rallies:
        return None

    stats_list = load_all_stats(video_id)
    if not stats_list:
        return None

    all_shuttle: list[ShuttleTrackPoint] = []
    for rally in rallies:
        all_shuttle.extend(load_shuttle_track(shuttle_track_path(video_id, rally.path.stem)))

    homography_path = CALIBRATION_DIR / video_id / "homography.json"
    homography = load_homography(homography_path) if homography_path.exists() else None

    all_player_detections = _collect_player_detections(video_id, rallies)

    out_path = ensure_dir(HEATMAPS_DIR / video_id) / "heatmaps.png"
    render_heatmaps(
        out_path,
        all_player_detections,
        all_shuttle,
        stats_list,
        homography=homography,
        title=f"RallyLens — {video_id}",
    )
    return out_path
