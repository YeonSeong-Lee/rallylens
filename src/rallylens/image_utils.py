"""OpenCV image helpers shared by LLM vision QA and overlay rendering."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np


def crop_around_bbox(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    padding_px: int = 80,
    draw_bbox: bool = True,
    bbox_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Crop a square region of `frame` around `bbox` with `padding_px` margin.

    If `draw_bbox` is True, the original bbox is drawn on the crop (in BGR)
    so the caller can visually see which region was flagged.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) / 2 + padding_px
    cx1 = int(max(0, cx - half))
    cy1 = int(max(0, cy - half))
    cx2 = int(min(w, cx + half))
    cy2 = int(min(h, cy + half))
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if draw_bbox:
        rel_x1 = int(x1 - cx1)
        rel_y1 = int(y1 - cy1)
        rel_x2 = int(x2 - cx1)
        rel_y2 = int(y2 - cy1)
        cv2.rectangle(crop, (rel_x1, rel_y1), (rel_x2, rel_y2), bbox_color, 2)
    return crop


def encode_jpeg_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV BGR image as a standard base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")
