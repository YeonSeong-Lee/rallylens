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
from rallylens.vision.shuttle_tracker import ShuttleTracker

_log = get_logger(__name__)


def run_shuttle_pipeline(
    video_path: Path,
    video_id: str,
    weights_path: Path | None = None,
) -> list:
    """Detect shuttlecock positions in every frame of a video.

    Args:
        video_path:   Path to the source video file.
        video_id:     Identifier used for the output subdirectory.
        weights_path: Optional explicit path to TrackNet weights.
                      Falls back to ``models/shuttle_tracknet.pth``.

    Returns:
        list[ShuttlePoint] — one entry per detected frame.
    """
    tracker = ShuttleTracker(weights_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    all_points = []
    frame_idx = 0
    seen_frame_idxs: set[int] = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for point in tracker.detect(frame, frame_idx):
            if point.frame_idx not in seen_frame_idxs:
                all_points.append(point)
                seen_frame_idxs.add(point.frame_idx)
        frame_idx += 1

    cap.release()

    # Sort by frame index (detect() returns points for the whole window)
    all_points.sort(key=lambda p: p.frame_idx)

    ensure_dir(TRACKS_DIR / video_id)
    out_path = save_shuttle_track(all_points, video_id, video_path.stem)
    _log.info("saved %d shuttle points → %s", len(all_points), out_path)
    return all_points
