"""Court detection pipeline stage.

Samples evenly-spaced frames from a video, applies detect_court_corners()
to each, and persists the first successful detection to disk.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from rallylens.common import ensure_dir, get_logger, open_video, read_video_properties
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

    corners: CourtCorners | None = None
    with open_video(video_path) as cap:
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

    if corners is None:
        _log.warning("court detection failed on all %d sampled frames", sample_count)
        return None

    ensure_dir(CALIBRATION_DIR / video_id)
    save_court_corners(corners, video_id)
    return corners


def run_court_detection_interactive(
    video_path: Path,
    video_id: str,
    sample_count: int = 20,
) -> CourtCorners | None:
    """Detect court corners with an interactive OpenCV confirmation window.

    Runs auto-detection first (same logic as :func:`run_court_detection`).
    Then opens a window showing the best candidate frame:

    * If auto-detection succeeded the detected corners are shown as a cyan
      overlay — press **Enter** to accept or **R** to redo manually.
    * If auto-detection failed the user must click all 4 corners by hand.

    Returns None only when the user explicitly cancels (ESC / Q).
    """
    from rallylens.vision.court_picker import pick_court_corners_interactively

    props = read_video_properties(video_path)
    step = max(1, props.frame_count // sample_count)

    auto_corners: CourtCorners | None = None
    best_frame = None

    with open_video(video_path) as cap:
        for i in range(sample_count):
            target = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                continue
            if best_frame is None:
                best_frame = frame
            result = detect_court_corners(frame)
            if result is not None:
                auto_corners = result
                best_frame = frame
                _log.info("court corners auto-detected from frame %d", target)
                break
            _log.debug("frame %d: no court corners detected", target)

    if best_frame is None:
        raise RuntimeError(f"could not read any frame from {video_path}")

    if auto_corners is None:
        _log.warning("auto-detection failed; opening interactive picker")
    else:
        _log.info("auto-detection succeeded; opening interactive picker for confirmation")

    corners = pick_court_corners_interactively(best_frame, initial_corners=auto_corners)

    if corners is None:
        _log.info("user cancelled interactive court calibration")
        return None

    ensure_dir(CALIBRATION_DIR / video_id)
    save_court_corners(corners, video_id)
    return corners
