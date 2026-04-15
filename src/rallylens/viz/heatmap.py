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
    IMG_W,
    compute_homography,
    draw_court_background,
    extract_foot_positions,
    project_point,
)

__all__ = ["render_heatmap"]


def _accumulate(positions: list[tuple[int, int]], sigma: int) -> np.ndarray:  # type: ignore[type-arg]
    grid: np.ndarray = np.zeros((IMG_H, IMG_W), dtype=np.float32)  # type: ignore[type-arg]
    for x, y in positions:
        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            grid[y, x] += 1.0
    kernel_size = sigma * 4 + 1
    return cv2.GaussianBlur(grid, (kernel_size, kernel_size), sigma)  # type: ignore[return-value]


def _blend_onto_court(
    court: np.ndarray,  # type: ignore[type-arg]
    grid: np.ndarray,  # type: ignore[type-arg]
    alpha_max: float = 0.75,
) -> np.ndarray:  # type: ignore[type-arg]
    max_val = float(grid.max())
    if max_val == 0:
        return court.copy()
    normalized = (grid / max_val * 255).astype(np.uint8)
    heatmap_rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    alpha_map = (grid / max_val * alpha_max).astype(np.float32)[:, :, np.newaxis]
    blended = (
        court.astype(np.float32) * (1 - alpha_map)
        + heatmap_rgb.astype(np.float32) * alpha_map
    ).astype(np.uint8)
    return blended


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

    # Player panel
    foot_positions = extract_foot_positions(detections, H, kp_conf_thresh)
    player_grid = _accumulate(foot_positions, blur_sigma)
    player_panel = _blend_onto_court(court_bg, player_grid)
    cv2.putText(
        player_panel, "Players", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # Shuttle panel
    shuttle_positions = [project_point(H, float(sp.x), float(sp.y)) for sp in shuttle_track]
    shuttle_grid = _accumulate(shuttle_positions, blur_sigma)
    shuttle_panel = _blend_onto_court(court_bg, shuttle_grid)
    cv2.putText(
        shuttle_panel, "Shuttle", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    gap: np.ndarray = np.full((IMG_H, 20, 3), 20, dtype=np.uint8)  # type: ignore[type-arg]
    combined = np.hstack([player_panel, gap, shuttle_panel])

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), combined)
    return out_path
