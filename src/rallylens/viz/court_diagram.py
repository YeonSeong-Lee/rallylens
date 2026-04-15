"""Court coordinate trajectory visualization.

Projects player foot positions and shuttlecock positions into standard court
diagram space via homography and saves the result as a PNG.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rallylens.common import ensure_dir
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint
from rallylens.viz._utils import (
    compute_homography,
    draw_court_background,
    project_point,
    track_color,
)

_SHUTTLE_COLOR: tuple[int, int, int] = (0, 255, 255)  # yellow

__all__ = ["render_court_diagram"]


def render_court_diagram(
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    corners: CourtCorners,
    out_path: Path,
    *,
    kp_conf_thresh: float = 0.3,
    player_radius: int = 4,
    shuttle_radius: int = 3,
) -> Path:
    """Project detections and shuttle track onto a court diagram and save as PNG."""
    H = compute_homography(corners)
    img = draw_court_background()

    # Player foot positions
    for det in detections:
        if len(det.keypoints_xy) < 17:
            continue
        l_conf = det.keypoints_conf[15] if len(det.keypoints_conf) > 15 else 0.0
        r_conf = det.keypoints_conf[16] if len(det.keypoints_conf) > 16 else 0.0
        if l_conf > kp_conf_thresh and r_conf > kp_conf_thresh:
            lx, ly = det.keypoints_xy[15]
            rx, ry = det.keypoints_xy[16]
            pt = project_point(H, (lx + rx) / 2, (ly + ry) / 2)
        elif l_conf > kp_conf_thresh:
            kx, ky = det.keypoints_xy[15]
            pt = project_point(H, kx, ky)
        elif r_conf > kp_conf_thresh:
            kx, ky = det.keypoints_xy[16]
            pt = project_point(H, kx, ky)
        else:
            continue
        color = track_color(det.track_id)
        cv2.circle(img, pt, player_radius, color, -1, cv2.LINE_AA)

    # Shuttle trajectory
    prev: tuple[int, int] | None = None
    for sp in shuttle_track:
        pt = project_point(H, float(sp.x), float(sp.y))
        cv2.circle(img, pt, shuttle_radius, _SHUTTLE_COLOR, -1, cv2.LINE_AA)
        if prev is not None:
            cv2.line(img, prev, pt, _SHUTTLE_COLOR, 1, cv2.LINE_AA)
        prev = pt

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), img)
    return out_path
