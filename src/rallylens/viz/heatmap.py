"""Position heatmap visualization.

Accumulates player foot positions and shuttle positions over all frames,
applies a Gaussian blur to create smooth hotspots, blends the result over the
court diagram with a JET colormap, and saves a side-by-side PNG.
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
    IMG_H,
    build_heatmap_over_court,
    compute_homography,
    draw_court_background,
    extract_foot_positions,
    project_point,
)

__all__ = ["render_heatmap"]


def render_heatmap(
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    corners: CourtCorners,
    out_path: Path,
    *,
    kp_conf_thresh: float = 0.3,
    blur_sigma: int = 12,
) -> Path:
    """Render player and shuttle position heatmaps as a side-by-side PNG."""
    H = compute_homography(corners)
    court_bg = draw_court_background()

    foot_positions = extract_foot_positions(detections, H, kp_conf_thresh)
    player_panel = build_heatmap_over_court(court_bg, foot_positions, blur_sigma=blur_sigma)
    cv2.putText(
        player_panel, "Players", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    shuttle_positions = [project_point(H, float(sp.x), float(sp.y)) for sp in shuttle_track]
    shuttle_panel = build_heatmap_over_court(court_bg, shuttle_positions, blur_sigma=blur_sigma)
    cv2.putText(
        shuttle_panel, "Shuttle", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    gap: np.ndarray = np.full((IMG_H, 20, 3), 20, dtype=np.uint8)
    combined = np.hstack([player_panel, gap, shuttle_panel])

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), combined)
    return out_path
