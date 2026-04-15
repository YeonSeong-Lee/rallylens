"""Internal shared helpers for the viz package.

Court diagram constants, homography computation, and foot-position extraction
are centralised here so overlay.py, court_diagram.py, and heatmap.py all stay
thin.
"""

from __future__ import annotations

import cv2
import numpy as np

from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection

# ---------------------------------------------------------------------------
# Court diagram constants (1 px = 1 cm)
# ---------------------------------------------------------------------------

COURT_W: int = 610   # 6.10 m — doubles width
COURT_H: int = 1340  # 13.40 m — court length
MARGIN: int = 60     # blank border around the court lines

IMG_W: int = COURT_W + 2 * MARGIN   # 730
IMG_H: int = COURT_H + 2 * MARGIN   # 1460

# Y-offsets from the top of the court (cm / px)
_NET_Y: int = 670
_SHORT_SVC_FAR: int = _NET_Y - 198   # 472
_SHORT_SVC_NEAR: int = _NET_Y + 198  # 868
_DBL_LONG_FAR: int = 76
_DBL_LONG_NEAR: int = COURT_H - 76   # 1264

# X-offsets
_SINGLES_LEFT: int = 46
_SINGLES_RIGHT: int = COURT_W - 46   # 564
_CENTER_X: int = COURT_W // 2        # 305

# ---------------------------------------------------------------------------
# Track-color palette (BGR)
# ---------------------------------------------------------------------------

TRACK_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),      # green
    (0, 128, 255),    # sky-blue / orange
    (255, 128, 0),    # orange-blue
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 255, 0),    # yellow
]


def track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


# ---------------------------------------------------------------------------
# Court diagram drawing
# ---------------------------------------------------------------------------


def draw_court_background() -> np.ndarray:  # type: ignore[type-arg]
    """Return a (IMG_H, IMG_W, 3) BGR image of a standard badminton court."""
    img: np.ndarray = np.full((IMG_H, IMG_W, 3), (34, 85, 34), dtype=np.uint8)  # type: ignore[type-arg]
    m = MARGIN

    def hline(y: int, x1: int = 0, x2: int = COURT_W) -> None:
        cv2.line(img, (m + x1, m + y), (m + x2, m + y), (255, 255, 255), 2)

    def vline(x: int, y1: int = 0, y2: int = COURT_H) -> None:
        cv2.line(img, (m + x, m + y1), (m + x, m + y2), (255, 255, 255), 2)

    # Outer doubles boundary
    cv2.rectangle(img, (m, m), (m + COURT_W, m + COURT_H), (255, 255, 255), 2)
    # Singles sidelines
    vline(_SINGLES_LEFT)
    vline(_SINGLES_RIGHT)
    # Net
    hline(_NET_Y)
    # Short service lines
    hline(_SHORT_SVC_FAR)
    hline(_SHORT_SVC_NEAR)
    # Doubles long service lines
    hline(_DBL_LONG_FAR)
    hline(_DBL_LONG_NEAR)
    # Center service line (full court length for clarity)
    vline(_CENTER_X)

    return img


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------


def compute_homography(corners: CourtCorners) -> np.ndarray:  # type: ignore[type-arg]
    """Compute homography mapping video pixel corners to the court diagram."""
    src = np.array(
        [
            [corners.top_left[0], corners.top_left[1]],
            [corners.top_right[0], corners.top_right[1]],
            [corners.bottom_left[0], corners.bottom_left[1]],
            [corners.bottom_right[0], corners.bottom_right[1]],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [MARGIN, MARGIN],
            [MARGIN + COURT_W, MARGIN],
            [MARGIN, MARGIN + COURT_H],
            [MARGIN + COURT_W, MARGIN + COURT_H],
        ],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError(
            "could not compute homography from court corners — corners may be collinear"
        )
    return H  # type: ignore[return-value]


def project_point(H: np.ndarray, px: float, py: float) -> tuple[int, int]:  # type: ignore[type-arg]
    """Project a single pixel coordinate into court diagram space."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return (int(result[0, 0, 0]), int(result[0, 0, 1]))


# ---------------------------------------------------------------------------
# Foot-position extraction
# ---------------------------------------------------------------------------


def extract_foot_positions(
    detections: list[Detection],
    H: np.ndarray,  # type: ignore[type-arg]
    kp_conf_thresh: float = 0.3,
) -> list[tuple[int, int]]:
    """Extract foot midpoints from detections and project to court diagram space."""
    positions: list[tuple[int, int]] = []
    for det in detections:
        if len(det.keypoints_xy) < 17:
            continue
        l_conf = det.keypoints_conf[15] if len(det.keypoints_conf) > 15 else 0.0
        r_conf = det.keypoints_conf[16] if len(det.keypoints_conf) > 16 else 0.0
        if l_conf > kp_conf_thresh and r_conf > kp_conf_thresh:
            lx, ly = det.keypoints_xy[15]
            rx, ry = det.keypoints_xy[16]
            positions.append(project_point(H, (lx + rx) / 2, (ly + ry) / 2))
        elif l_conf > kp_conf_thresh:
            kx, ky = det.keypoints_xy[15]
            positions.append(project_point(H, kx, ky))
        elif r_conf > kp_conf_thresh:
            kx, ky = det.keypoints_xy[16]
            positions.append(project_point(H, kx, ky))
    return positions
