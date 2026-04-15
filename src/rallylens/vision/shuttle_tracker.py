"""Shuttlecock detection using TrackNetV3.

Reference: https://github.com/qaz812345/TrackNetV3

Wraps TrackNet inference with a rolling frame buffer, coordinate extraction
from sigmoid heatmaps (contour-based), and coordinate rescaling to the
original video resolution.

Weights path: models/shuttle_tracknet.pth
If the weights file is missing, ShuttleTracker logs a warning and
detect() always returns None (graceful fallback).
"""

from __future__ import annotations

import contextlib
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from rallylens.common import get_logger
from rallylens.config import MODELS_DIR
from rallylens.vision.tracknet import TrackNet

_log = get_logger(__name__)

# TrackNetV3 inference resolution
INFER_H = 288
INFER_W = 512

# Default rolling window (sequence length)
SEQ_LEN = 8

# Heatmap detection threshold
HEATMAP_THRESH = 0.5

# Default weights filename under MODELS_DIR
DEFAULT_WEIGHTS = "shuttle_tracknet.pth"


class ShuttlePoint(BaseModel):
    """Single shuttlecock detection in one frame."""

    model_config = ConfigDict(frozen=True)

    frame_idx: int
    x: int  # pixel x in original video resolution
    y: int  # pixel y in original video resolution


def _stack_frames(frames: list[np.ndarray]) -> torch.Tensor:
    """Stack a list of BGR frames into a (1, 3*seq_len, H, W) float tensor."""
    channels = []
    for frame in frames:
        resized = cv2.resize(frame, (INFER_W, INFER_H)).astype(np.float32) / 255.0
        # BGR → RGB channel order
        channels.append(resized[:, :, ::-1].transpose(2, 0, 1))  # (3, H, W)
    stacked = np.concatenate(channels, axis=0)  # (3*seq_len, H, W)
    return torch.from_numpy(stacked).unsqueeze(0)  # (1, 3*seq_len, H, W)


def _predict_location(heatmap: np.ndarray) -> tuple[int, int] | None:
    """Extract (x, y) from a single sigmoid heatmap via contour bounding box.

    Returns the center of the largest contour whose pixel value exceeds the
    threshold, or None if no shuttlecock is detected.
    """
    binary = (heatmap > HEATMAP_THRESH).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    return int(bx + bw / 2), int(by + bh / 2)


class ShuttleTracker:
    """Stateful shuttlecock tracker backed by TrackNetV3.

    Usage::

        tracker = ShuttleTracker()
        for idx, frame in enumerate(frames):
            point = tracker.detect(frame, idx)
            if point:
                print(point.x, point.y)
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        resolved = weights_path or (MODELS_DIR / DEFAULT_WEIGHTS)
        self._model: TrackNet | None = None
        self._enabled = False

        if not resolved.exists():
            _log.warning(
                "shuttle weights not found at %s — detection disabled. "
                "Provide models/%s to enable.",
                resolved,
                DEFAULT_WEIGHTS,
            )
            return

        model = TrackNet(in_dim=3 * SEQ_LEN, out_dim=SEQ_LEN)
        with contextlib.chdir(MODELS_DIR):
            model.load_state_dict(torch.load(resolved, map_location="cpu"))
        model.eval()
        self._model = model
        self._enabled = True
        _log.info("shuttle TrackNet loaded from %s", resolved)

        self._buffer: deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        # frame_idx corresponding to each buffer slot (oldest → newest)
        self._idx_buffer: deque[int] = deque(maxlen=SEQ_LEN)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[ShuttlePoint]:
        """Add a frame to the buffer and run detection when buffer is full.

        Args:
            frame:     BGR numpy array (any resolution).
            frame_idx: 0-based frame index in the source video.

        Returns:
            List of ShuttlePoint — one per detected frame in the current
            window (up to SEQ_LEN points), or an empty list if the buffer
            is not yet full or the model is disabled.
        """
        if not self._enabled or self._model is None:
            return []

        orig_h, orig_w = frame.shape[:2]
        self._buffer.append(frame)
        self._idx_buffer.append(frame_idx)

        if len(self._buffer) < SEQ_LEN:
            return []

        tensor = _stack_frames(list(self._buffer))

        with torch.no_grad():
            output = self._model(tensor)  # (1, SEQ_LEN, H, W)

        points: list[ShuttlePoint] = []
        for i in range(SEQ_LEN):
            heatmap = output[0, i].cpu().numpy()  # (H, W) float in [0,1]
            xy = _predict_location(heatmap)
            if xy is None:
                continue
            # Rescale from inference resolution to original
            x = int(xy[0] * orig_w / INFER_W)
            y = int(xy[1] * orig_h / INFER_H)
            points.append(ShuttlePoint(frame_idx=self._idx_buffer[i], x=x, y=y))

        return points
