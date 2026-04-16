"""Badminton court corner detection using classical computer vision.

Pipeline per frame:
  1. Grayscale → binary threshold → Canny edges → HoughLinesP
  2. Count intersections per line; pick top-8 most-intersecting lines.
  3. floodFill those lines to isolate court region; mask out non-court areas.
  4. Dilate → erode → Canny → HoughLines on the cleaned mask.
  5. Find the four extreme boundary lines (leftmost, rightmost, topmost, bottommost).
  6. Compute the four court corners as pairwise intersections of those lines.

Returns CourtCorners or None if detection fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, pi, sin
from typing import Final

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

# ── Stage 1: edge detection ────────────────────────────────────────────────
_GRAY_THRESHOLD: Final[int] = 156
_CANNY_LOW: Final[int] = 100
_CANNY_HIGH: Final[int] = 200
_CANNY_ERODED_LOW: Final[int] = 90
_CANNY_ERODED_HIGH: Final[int] = 100

# ── Stage 1: HoughLinesP ───────────────────────────────────────────────────
_HOUGH_THRESHOLD: Final[int] = 150
_HOUGH_MIN_LINE_LENGTH: Final[int] = 100
_HOUGH_MAX_LINE_GAP: Final[int] = 10

# ── Stage 2: line intersection ─────────────────────────────────────────────
_INTERSECTION_BOUNDARY_PAD: Final[int] = 200

# ── Stage 3: morphology + line ranking ─────────────────────────────────────
_MORPHOLOGY_KERNEL_SIZE: Final[tuple[int, int]] = (5, 5)
_TOP_INTERSECTING_LINES: Final[int] = 8

# ── Stage 4: HoughLines on cleaned mask ────────────────────────────────────
_HOUGH_MAIN_THRESHOLD: Final[int] = 300

# ── Small vertical correction applied to the top boundary lines ────────────
_TOP_LINE_VERTICAL_CORRECTION: Final[int] = 4


class CourtCorners(BaseModel):
    """Four corner pixel coordinates of the detected badminton court."""

    model_config = ConfigDict(frozen=True)

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

_Line = list[list[float]]  # [[x1,y1],[x2,y2]]


def _determinant(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _find_intersection(
    line1: _Line,
    line2: _Line,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[int, int] | None:
    """Return the integer intersection point of two lines, bounded by the given box."""
    xd = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    yd = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = _determinant(xd, yd)
    if div == 0:
        return None
    d = (_determinant(*line1), _determinant(*line2))
    x = int(_determinant(d, xd) / div)
    y = int(_determinant(d, yd) / div)
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return None
    return x, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_court_corners(frame: np.ndarray) -> CourtCorners | None:
    """Detect the four corners of a badminton court in a video frame.

    Args:
        frame: BGR image (any resolution).

    Returns:
        CourtCorners with pixel coordinates, or None if detection fails.
    """
    height, width = frame.shape[:2]
    extra = width // 3  # extension beyond frame edges for line intersection search

    # ── Stage 1: edge detection ──────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, _GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(bw, _CANNY_LOW, _CANNY_HIGH)

    h_lines_p = cv2.HoughLinesP(
        canny,
        1,
        pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=_HOUGH_MAX_LINE_GAP,
    )
    if h_lines_p is None or len(h_lines_p) == 0:
        return None

    # ── Stage 2: rank lines by intersection count ────────────────────────────
    n = len(h_lines_p)
    intersect_count = np.zeros((n, 2))  # [count, original_index]
    for i, l1 in enumerate(h_lines_p):
        x1, y1, x2, y2 = l1[0]
        seg1: _Line = [[float(x1), float(y1)], [float(x2), float(y2)]]
        lx1, lx2 = (x1, x2) if x1 <= x2 else (x2, x1)
        ly1, ly2 = (y1, y2) if y1 <= y2 else (y2, y1)
        for l2 in h_lines_p:
            x3, y3, x4, y4 = l2[0]
            seg2: _Line = [[float(x3), float(y3)], [float(x4), float(y4)]]
            if l1 is l2:
                continue
            if _find_intersection(
                seg1,
                seg2,
                lx1 - _INTERSECTION_BOUNDARY_PAD,
                ly1 - _INTERSECTION_BOUNDARY_PAD,
                lx2 + _INTERSECTION_BOUNDARY_PAD,
                ly2 + _INTERSECTION_BOUNDARY_PAD,
            ):
                intersect_count[i][0] += 1
        intersect_count[i][1] = i

    # ── Stage 3: flood-fill the top-8 most intersecting lines to isolate court
    dilation = cv2.dilate(bw, np.ones(_MORPHOLOGY_KERNEL_SIZE, np.uint8), iterations=1)
    non_rect = dilation.copy()
    ranked = intersect_count[(-intersect_count)[:, 0].argsort()]

    for p in range(min(_TOP_INTERSECTING_LINES, n)):
        orig_idx = int(ranked[p][1])
        # Preserve original (quirky) gating: look up ranked row at orig_idx slot.
        if orig_idx >= n or ranked[orig_idx][0] <= 0:
            continue
        x1, y1, x2, y2 = h_lines_p[orig_idx][0]
        for seed in ((x1, y1), (x2, y2)):
            mask = np.zeros((height + 2, width + 2), np.uint8)
            cv2.floodFill(non_rect, mask, seed, 1)

    dilation[non_rect == 255] = 0
    dilation[non_rect == 1] = 255
    eroded = cv2.erode(dilation, np.ones(_MORPHOLOGY_KERNEL_SIZE, np.uint8))
    canny_main = cv2.Canny(eroded, _CANNY_ERODED_LOW, _CANNY_ERODED_HIGH)

    # ── Stage 4: find extreme boundary lines via HoughLines ──────────────────
    h_lines = cv2.HoughLines(canny_main, 2, pi / 180, _HOUGH_MAIN_THRESHOLD)
    if h_lines is None or len(h_lines) == 0:
        return None

    # Reference border axes (extended beyond frame)
    axis_top: _Line    = [[-extra, 0], [width + extra, 0]]
    axis_right: _Line  = [[width + extra, 0], [width + extra, height]]
    axis_bottom: _Line = [[-extra, height], [width + extra, height]]
    axis_left: _Line   = [[-extra, 0], [-extra, height]]

    xo_left = width + extra
    xo_right = -extra
    xf_left = width + extra
    xf_right = -extra
    yo_top = height
    yo_bottom = 0
    yf_top = height
    yf_bottom = 0

    xo_left_line = xo_right_line = yo_top_line = yo_bottom_line = None
    xf_left_line = xf_right_line = yf_top_line = yf_bottom_line = None

    bounds = (-extra, 0, width + extra, height)

    for h_line in h_lines:
        rho, theta = h_line[0]
        a, b = cos(theta), sin(theta)
        x0, y0 = a * rho, b * rho
        seg: _Line = [
            [x0 + width * (-b), y0 + width * a],
            [x0 - width * (-b), y0 - width * a],
        ]

        pt_xo = _find_intersection(axis_top, seg, *bounds)
        pt_yo = _find_intersection(axis_left, seg, *bounds)
        pt_xf = _find_intersection(axis_bottom, seg, *bounds)
        pt_yf = _find_intersection(axis_right, seg, *bounds)

        if pt_xo:
            if pt_xo[0] < xo_left:
                xo_left, xo_left_line = pt_xo[0], seg
            if pt_xo[0] > xo_right:
                xo_right, xo_right_line = pt_xo[0], seg
        if pt_yo:
            if pt_yo[1] < yo_top:
                yo_top, yo_top_line = pt_yo[1], seg
            if pt_yo[1] > yo_bottom:
                yo_bottom, yo_bottom_line = pt_yo[1], seg
        if pt_xf:
            if pt_xf[0] < xf_left:
                xf_left, xf_left_line = pt_xf[0], seg
            if pt_xf[0] > xf_right:
                xf_right, xf_right_line = pt_xf[0], seg
        if pt_yf:
            if pt_yf[1] < yf_top:
                yf_top, yf_top_line = pt_yf[1], seg
            if pt_yf[1] > yf_bottom:
                yf_bottom, yf_bottom_line = pt_yf[1], seg

    required = [xo_left_line, xo_right_line, yo_top_line, yo_bottom_line,
                xf_left_line, xf_right_line, yf_top_line, yf_bottom_line]
    if any(seg_line is None for seg_line in required):
        return None

    assert xo_left_line is not None
    assert xo_right_line is not None
    assert yo_top_line is not None
    assert yo_bottom_line is not None
    assert xf_left_line is not None
    assert xf_right_line is not None
    assert yf_top_line is not None
    assert yf_bottom_line is not None

    # Apply a small vertical correction to the top lines (same as TRACE)
    for seg in (yo_top_line, yf_top_line):
        seg[0][1] += _TOP_LINE_VERTICAL_CORRECTION
        seg[1][1] += _TOP_LINE_VERTICAL_CORRECTION

    # ── Stage 5: compute four corners ────────────────────────────────────────
    tl = _find_intersection(xo_left_line,  yo_top_line,    *bounds)
    tr = _find_intersection(xo_right_line, yf_top_line,    *bounds)
    bl = _find_intersection(xf_left_line,  yo_bottom_line, *bounds)
    br = _find_intersection(xf_right_line, yf_bottom_line, *bounds)

    if None in (tl, tr, bl, br):
        return None

    return CourtCorners(
        top_left=tl,
        top_right=tr,
        bottom_left=bl,
        bottom_right=br,
    )
