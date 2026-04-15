"""Shuttle tracking pipeline stage.

Opens a video, feeds every frame through ShuttleTracker, and persists
the detected ShuttlePoints to disk as a JSONL file.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from rallylens.common import ensure_dir, get_logger
from rallylens.config import TRACKS_DIR
from rallylens.pipeline.io import save_shuttle_track
from rallylens.vision.shuttle_tracker import ShuttlePoint, ShuttleTracker

_log = get_logger(__name__)


def run_shuttle_pipeline(
    video_path: Path,
    video_id: str,
    weights_path: Path | None = None,
) -> list[ShuttlePoint]:
    """Detect shuttlecock positions in every frame of a video.

    Args:
        video_path:   Path to the source video file.
        video_id:     Identifier used for the output subdirectory.
        weights_path: Optional explicit path to TrackNet weights.
                      Falls back to ``models/shuttle_tracknet.pth``.

    Returns:
        list[ShuttlePoint] — one entry per detected frame, sorted by frame_idx.
    """
    tracker = ShuttleTracker(weights_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    # Dedup across overlapping sliding windows: each full window returns points
    # for SEQ_LEN frame indices; successive windows overlap by SEQ_LEN-1 frames.
    points_by_frame: dict[int, ShuttlePoint] = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for point in tracker.detect(frame, frame_idx):
            points_by_frame.setdefault(point.frame_idx, point)
        frame_idx += 1

    cap.release()

    all_points = [points_by_frame[k] for k in sorted(points_by_frame)]

    ensure_dir(TRACKS_DIR / video_id)
    out_path = save_shuttle_track(all_points, video_id)
    _log.info("saved %d shuttle points → %s", len(all_points), out_path)
    return all_points
