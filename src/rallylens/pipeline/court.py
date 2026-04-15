"""Court detection pipeline stage.

Samples evenly-spaced frames from a video, applies detect_court_corners()
to each, and persists the first successful detection to disk.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from rallylens.common import ensure_dir, get_logger, read_video_properties
from rallylens.config import CALIBRATION_DIR
from rallylens.pipeline.io import save_court_corners
from rallylens.vision.court_detector import CourtCorners, detect_court_corners

_log = get_logger(__name__)


def run_court_detection(
    video_path: Path,
    video_id: str,
    sample_count: int = 20,
) -> CourtCorners | None:
    """Detect badminton court corners from a video.

    Samples ``sample_count`` evenly-spaced frames and returns the first
    successful detection.  Saves the result to
    ``data/calibration/{video_id}/corners.json``.

    Args:
        video_path:   Source video file.
        video_id:     Identifier for the output subdirectory.
        sample_count: Number of frames to try before giving up.

    Returns:
        CourtCorners on success, None if detection fails on all sampled frames.
    """
    props = read_video_properties(video_path)
    step = max(1, props.frame_count // sample_count)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    corners: CourtCorners | None = None
    for i in range(sample_count):
        target = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            continue
        result = detect_court_corners(frame)
        if result is not None:
            corners = result
            _log.info("court corners detected from frame %d", target)
            break
        _log.debug("frame %d: no court corners detected", target)

    cap.release()

    if corners is None:
        _log.warning("court detection failed on all %d sampled frames", sample_count)
        return None

    ensure_dir(CALIBRATION_DIR / video_id)
    save_court_corners(corners, video_id)
    return corners
