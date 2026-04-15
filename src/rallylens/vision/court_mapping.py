"""Badminton court perspective mapping (top-down view).

Reference: https://github.com/hgupt3/TRACE  CourtMapping.py

Transforms detected court corners into a bird's-eye court diagram using
OpenCV's perspective transform (getPerspectiveTransform / warpPerspective).

Badminton court dimensions (BWF standard, doubles):
  Total length : 13.40 m
  Doubles width:  6.70 m
  Singles width:  6.10 m
  Net position : 6.70 m from each baseline (centre)
  Short service line: 1.98 m from net
  Long service line (doubles): 0.76 m from baseline
  Singles long service line: same as baseline
  Side alley (doubles): (6.70 - 6.10) / 2 = 0.30 m each side
  Centre line: splits court lengthwise (parallel to long axis)

Canvas: 1 pixel ≈ 1 cm  →  670 × 1340 px (doubles court)
"""

from __future__ import annotations

import cv2
import numpy as np

from rallylens.vision.court_detector import CourtCorners

# ---------------------------------------------------------------------------
# Canvas constants  (pixels, doubles court, 1 px ≈ 1 cm)
# ---------------------------------------------------------------------------

CANVAS_W = 670   # doubles court width  (6.70 m × 100)
CANVAS_H = 1340  # court length         (13.40 m × 100)

# Padding around the court on the canvas
_PAD_X = 60
_PAD_Y = 80

_FULL_W = CANVAS_W + 2 * _PAD_X   # 790
_FULL_H = CANVAS_H + 2 * _PAD_Y   # 1500

# Court corners on the canvas (top-left origin, y increases downward)
_CTL = (_PAD_X,            _PAD_Y)
_CTR = (_PAD_X + CANVAS_W, _PAD_Y)
_CBL = (_PAD_X,            _PAD_Y + CANVAS_H)
_CBR = (_PAD_X + CANVAS_W, _PAD_Y + CANVAS_H)

# ── Line positions in canvas pixels ─────────────────────────────────────────
# Net (centre of court)
_NET_Y = _PAD_Y + CANVAS_H // 2

# Short service lines (1.98 m = 198 px from net)
_SSL_TOP_Y    = _NET_Y - 198
_SSL_BOTTOM_Y = _NET_Y + 198

# Long service line – doubles (0.76 m = 76 px from baseline)
_LSL_TOP_Y    = _PAD_Y + 76
_LSL_BOTTOM_Y = _PAD_Y + CANVAS_H - 76

# Side alleys (0.30 m = 30 px each side)
_ALLEY_LEFT_X  = _PAD_X + 30
_ALLEY_RIGHT_X = _PAD_X + CANVAS_W - 30

# Centre line (vertical, splits court width in half)
_CTR_LINE_X = _PAD_X + CANVAS_W // 2

_WHITE = (255, 255, 255)
_LINE_T = 2


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CourtMapper:
    """Applies a perspective transform from frame coordinates to a top-down court.

    Args:
        corners: Four court corners detected in the video frame.
    """

    def __init__(self, corners: CourtCorners) -> None:
        src = np.float32([
            list(corners.top_left),
            list(corners.top_right),
            list(corners.bottom_left),
            list(corners.bottom_right),
        ])
        dst = np.float32([list(_CTL), list(_CTR), list(_CBL), list(_CBR)])
        self._M = cv2.getPerspectiveTransform(src, dst)

    # ------------------------------------------------------------------

    def to_court_px(self, point: tuple[float, float]) -> tuple[int, int]:
        """Map a pixel coordinate in the original frame to the top-down canvas."""
        pts = np.float32([[list(point)]])
        transformed = cv2.perspectiveTransform(pts, self._M)[0][0]
        return int(transformed[0]), int(transformed[1])

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Warp the entire frame to the top-down view."""
        return cv2.warpPerspective(frame, self._M, (_FULL_W, _FULL_H))

    def render_empty_court(self) -> np.ndarray:
        """Return a blank BGR image with the badminton court lines drawn."""
        canvas = np.zeros((_FULL_H, _FULL_W, 3), dtype=np.uint8)
        _draw_court(canvas)
        return canvas


# ---------------------------------------------------------------------------
# Internal drawing
# ---------------------------------------------------------------------------


def _draw_court(img: np.ndarray) -> None:
    """Draw all standard badminton court lines onto img (in-place)."""

    def line(p1: tuple[int, int], p2: tuple[int, int]) -> None:
        cv2.line(img, p1, p2, _WHITE, _LINE_T)

    def rect(tl: tuple[int, int], br: tuple[int, int]) -> None:
        cv2.rectangle(img, tl, br, _WHITE, _LINE_T)

    # Outer boundary (doubles)
    rect(_CTL, _CBR)

    # Singles side lines
    rect((_ALLEY_LEFT_X, _PAD_Y), (_ALLEY_RIGHT_X, _PAD_Y + CANVAS_H))

    # Net line
    line((_PAD_X, _NET_Y), (_PAD_X + CANVAS_W, _NET_Y))

    # Short service lines (both halves)
    line((_PAD_X, _SSL_TOP_Y),    (_PAD_X + CANVAS_W, _SSL_TOP_Y))
    line((_PAD_X, _SSL_BOTTOM_Y), (_PAD_X + CANVAS_W, _SSL_BOTTOM_Y))

    # Long service lines – doubles (both ends)
    line((_PAD_X, _LSL_TOP_Y),    (_PAD_X + CANVAS_W, _LSL_TOP_Y))
    line((_PAD_X, _LSL_BOTTOM_Y), (_PAD_X + CANVAS_W, _LSL_BOTTOM_Y))

    # Centre line (between short service lines only — standard badminton rule)
    line((_CTR_LINE_X, _SSL_TOP_Y), (_CTR_LINE_X, _SSL_BOTTOM_Y))
