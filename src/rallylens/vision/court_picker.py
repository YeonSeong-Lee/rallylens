"""Interactive court corner picker using an OpenCV window.

Opens a resizable OpenCV window on a video frame and lets the user click
4 corner points in order: top-left → top-right → bottom-right → bottom-left.

If auto-detected corners are provided they are pre-filled as a yellow overlay
so the user can confirm with Enter or redo manually.

Keyboard shortcuts
------------------
Enter / Space  — confirm (requires 4 points placed)
Backspace      — undo last point
R              — reset all points
ESC / Q        — cancel (returns None)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

import cv2
import numpy as np

from rallylens.vision.court_detector import CourtCorners

# Click order → CourtCorners field names
CORNER_ORDER: tuple[str, ...] = ("top-left", "top-right", "bottom-right", "bottom-left")

# Display window size cap
_MAX_W = 1280
_MAX_H = 720

# Drawing constants
_STATUS_BAR_HEIGHT: Final[int] = 40
_HEADER_BAR_HEIGHT: Final[int] = 28
_CORNER_POINT_RADIUS: Final[int] = 10
_WAITKEY_MS: Final[int] = 20
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICK = 1


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _PickerState:
    clicks: list[tuple[int, int]] = field(default_factory=list)  # original-resolution coords
    confirmed: bool = False
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def _corners_to_display_pts(
    corners: CourtCorners,
    scale: float,
) -> list[tuple[int, int]]:
    """Convert CourtCorners (original resolution) to display-space points in click order."""
    orig = [
        corners.top_left,
        corners.top_right,
        corners.bottom_right,
        corners.bottom_left,
    ]
    return [(int(x * scale), int(y * scale)) for x, y in orig]


def _draw_overlay(
    display_frame: np.ndarray,
    clicks: list[tuple[int, int]],
    initial_corners: CourtCorners | None,
    scale: float,
) -> np.ndarray:
    """Return a copy of *display_frame* with interactive overlay drawn on it."""
    img = display_frame.copy()
    h, w = img.shape[:2]

    # ── Ghost overlay for auto-detected corners (shown only when no user clicks yet) ──
    if initial_corners is not None and len(clicks) == 0:
        auto_pts = _corners_to_display_pts(initial_corners, scale)
        labels = ("TL", "TR", "BR", "BL")
        # Draw quad outline
        for i in range(4):
            cv2.line(img, auto_pts[i], auto_pts[(i + 1) % 4], (0, 220, 220), 2)
        # Draw circles and labels
        for pt, label in zip(auto_pts, labels, strict=True):
            cv2.circle(img, pt, _CORNER_POINT_RADIUS, (0, 220, 220), 2)
            cv2.putText(img, f"AUTO:{label}", (pt[0] + 12, pt[1] - 6),
                        _FONT, _FONT_SCALE - 0.1, (0, 220, 220), _FONT_THICK, cv2.LINE_AA)

    # ── User-placed click points ──
    display_clicks = [(int(x * scale), int(y * scale)) for x, y in clicks]

    # Connecting lines / quad
    color_line = (0, 255, 255)  # cyan while incomplete
    if len(display_clicks) >= 2:
        for i in range(len(display_clicks) - 1):
            cv2.line(img, display_clicks[i], display_clicks[i + 1], color_line, 2)
    if len(display_clicks) == 4:
        cv2.line(img, display_clicks[3], display_clicks[0], (0, 220, 0), 2)

    # Corner circles and labels
    for i, pt in enumerate(display_clicks):
        label = CORNER_ORDER[i]
        circle_color = (0, 220, 0) if len(clicks) == 4 else (255, 255, 255)
        cv2.circle(img, pt, _CORNER_POINT_RADIUS, (0, 0, 0), -1)   # black border
        cv2.circle(img, pt, 8, circle_color, -1)
        cv2.putText(img, f"{i + 1}:{label}", (pt[0] + 12, pt[1] - 6),
                    _FONT, _FONT_SCALE - 0.05, circle_color, _FONT_THICK, cv2.LINE_AA)

    # ── Status bar ──
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - _STATUS_BAR_HEIGHT), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    if initial_corners is not None and len(clicks) == 0:
        status = "Auto-detected corners shown.  ENTER=confirm  R=redo  Q=cancel"
    elif len(clicks) < 4:
        next_corner = CORNER_ORDER[len(clicks)]
        status = f"Click {next_corner} corner  ({len(clicks)}/4)   BKSP=undo  R=reset  Q=cancel"
    else:
        status = "4 corners set.  ENTER=save  BKSP=undo  R=reset  Q=cancel"

    cv2.putText(img, status, (10, h - 12), _FONT, _FONT_SCALE, (220, 220, 220),
                _FONT_THICK, cv2.LINE_AA)

    # ── Top header bar ──
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (0, 0), (w, _HEADER_BAR_HEIGHT), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.75, img, 0.25, 0, img)
    cv2.putText(img, "RallyLens  Court Calibration", (10, 20),
                _FONT, _FONT_SCALE, (200, 200, 200), _FONT_THICK, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Mouse callback factory
# ---------------------------------------------------------------------------


def _make_mouse_callback(
    state: _PickerState,
    base_display: np.ndarray,
    initial_corners: CourtCorners | None,
    scale: float,
    window_name: str,
) -> Callable[[int, int, int, int, Any | None], None]:
    def _callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state.confirmed or state.cancelled:
            return
        if len(state.clicks) >= 4:
            return
        # Store in original-resolution coordinates
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        state.clicks.append((orig_x, orig_y))
        img = _draw_overlay(base_display, state.clicks, initial_corners, scale)
        cv2.imshow(window_name, img)

    return _callback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pick_court_corners_interactively(
    frame: np.ndarray,
    initial_corners: CourtCorners | None = None,
    window_name: str = "RallyLens — Court Calibration",
) -> CourtCorners | None:
    """Open an OpenCV window for the user to click 4 court corners.

    Args:
        frame:           BGR image (original resolution) to display.
        initial_corners: Pre-fill with auto-detected corners if available.
                         Shown as a cyan overlay; user can confirm (Enter) or redo (R).
        window_name:     Title of the OpenCV window.

    Returns:
        CourtCorners from user input (or confirmed auto corners), or None if cancelled.
    """
    h, w = frame.shape[:2]
    scale = min(_MAX_W / w, _MAX_H / h, 1.0)
    disp_w = int(w * scale)
    disp_h = int(h * scale)

    base_display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    state = _PickerState()

    # Pre-fill with auto-detected corners so the user can confirm immediately
    if initial_corners is not None:
        orig_pts = [
            (int(initial_corners.top_left[0]),    int(initial_corners.top_left[1])),
            (int(initial_corners.top_right[0]),   int(initial_corners.top_right[1])),
            (int(initial_corners.bottom_right[0]), int(initial_corners.bottom_right[1])),
            (int(initial_corners.bottom_left[0]),  int(initial_corners.bottom_left[1])),
        ]
        # We do NOT pre-fill state.clicks; instead we draw the ghost and let the user
        # decide to confirm (Enter) or redo (R).  The pre-fill into state.clicks happens
        # when the user presses Enter without any manual clicks.
        _auto_pts = orig_pts  # kept for confirm-without-click path
    else:
        _auto_pts = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(
        window_name,
        _make_mouse_callback(state, base_display, initial_corners, scale, window_name),
    )

    while True:
        img = _draw_overlay(base_display, state.clicks, initial_corners, scale)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(_WAITKEY_MS) & 0xFF

        # Confirm
        if key in (13, 32):  # Enter or Space
            if len(state.clicks) == 4:
                state.confirmed = True
                break
            elif len(state.clicks) == 0 and initial_corners is not None:
                # User accepted auto-detected corners without clicking
                state.clicks = list(_auto_pts)  # type: ignore[arg-type]
                state.confirmed = True
                break

        # Undo last point
        elif key in (8, 127):  # Backspace or Delete
            if state.clicks:
                state.clicks.pop()

        # Reset all points
        elif key in (ord("r"), ord("R")):
            state.clicks.clear()
            # Also clear the initial_corners ghost so user starts fresh
            initial_corners = None
            _auto_pts = None

        # Cancel
        elif key in (27, ord("q"), ord("Q")):  # ESC or Q
            state.cancelled = True
            break

        # Window closed by OS
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            state.cancelled = True
            break

    cv2.destroyWindow(window_name)

    if state.cancelled or len(state.clicks) != 4:
        return None

    c = state.clicks
    return CourtCorners(
        top_left=(float(c[0][0]), float(c[0][1])),
        top_right=(float(c[1][0]), float(c[1][1])),
        bottom_right=(float(c[2][0]), float(c[2][1])),
        bottom_left=(float(c[3][0]), float(c[3][1])),
    )
