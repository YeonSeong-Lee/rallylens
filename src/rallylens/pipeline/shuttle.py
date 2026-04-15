"""Shuttle detection + Kalman tracking pipeline stage."""

from __future__ import annotations

from pathlib import Path

from rallylens.common import TRACKS_DIR, ensure_dir, get_logger, read_video_properties
from rallylens.pipeline.io import save_track_jsonl, shuttle_track_path
from rallylens.vision.kalman_tracker import (
    ShuttleTrackPoint,
    observations_from_detections,
    track_shuttle,
)
from rallylens.vision.shuttlecock_detector import detect_shuttlecocks, resolve_weights

_log = get_logger(__name__)


def run_shuttle_pipeline(clip_path: Path, video_id: str) -> list[ShuttleTrackPoint]:
    """Detect shuttlecocks in a rally clip and produce a Kalman-smoothed track."""
    _, is_fine_tuned = resolve_weights()
    if not is_fine_tuned:
        _log.warning(
            "shuttle detector running WITHOUT fine-tuned weights — "
            "track quality will be very poor. Run Week 2 Colab notebook first."
        )

    candidates = detect_shuttlecocks(clip_path)
    obs = observations_from_detections(
        (c.frame_idx, c.bbox_xyxy, c.confidence) for c in candidates
    )
    props = read_video_properties(clip_path)
    track = track_shuttle(obs, props.frame_count)

    ensure_dir(TRACKS_DIR / video_id)
    out_path = shuttle_track_path(video_id, clip_path.stem)
    save_track_jsonl(track, out_path)
    _log.info("saved shuttle track -> %s", out_path)
    return track
