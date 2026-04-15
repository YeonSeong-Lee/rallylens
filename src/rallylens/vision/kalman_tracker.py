"""Kalman filter + nearest-neighbor association for shuttlecock tracking.

Single-object assumption (one shuttle per rally). The 4-state filter
predicts [x, y, vx, vy] and the measurement is [x, y] of the highest-confidence
bbox center from the shuttlecock detector.

Why Kalman here (not ByteTrack): the shuttle is a single object with high-speed
motion, frequent missed detections from the single-frame YOLO, and occasional
occlusion. Kalman with NN association interpolates missed frames — a cheap
approximation of TrackNetV3's 3-frame temporal input.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from rallylens.common import get_logger

_log = get_logger(__name__)


class ShuttleObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_idx: int
    x: float
    y: float
    confidence: float


class ShuttleTrackPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_idx: int
    x: float
    y: float
    vx: float
    vy: float
    interpolated: bool
    residual: float  # distance between predicted and observed (NaN if interpolated)

    @field_validator("residual", mode="before")
    @classmethod
    def _parse_residual(cls, v: Any) -> float:
        if v is None:
            return float("nan")
        return float(v)

    @field_serializer("residual")
    def _serialize_residual(self, v: float) -> float | None:
        return None if np.isnan(v) else v


class ShuttleKalmanTracker:
    """Constant-velocity Kalman filter for a single shuttlecock.

    State: x = [px, py, vx, vy]^T
    Transition: x_{k+1} = F x_k with F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    Measurement: z = H x + noise, H = [[1,0,0,0],[0,1,0,0]]
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_std: float = 4.0,      # pixels / frame^2 (shuttle acceleration scale)
        measurement_std: float = 3.0,  # pixels (YOLO bbox center noise)
        init_var: float = 500.0,
        association_max_distance: float = 80.0,
        max_missed_frames: int = 10,
    ) -> None:
        self.dt = dt
        self.association_max_distance = association_max_distance
        self.max_missed_frames = max_missed_frames

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        q = process_std**2
        self.Q = np.diag([q * dt**4 / 4, q * dt**4 / 4, q * dt**2, q * dt**2])
        r = measurement_std**2
        self.R = np.diag([r, r])
        self.init_P = np.eye(4) * init_var

        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._missed = 0

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def _initialize(self, obs: ShuttleObservation) -> None:
        self._x = np.array([obs.x, obs.y, 0.0, 0.0])
        self._P = self.init_P.copy()
        self._missed = 0

    def _predict(self) -> np.ndarray:
        if self._x is None or self._P is None:
            raise RuntimeError("Kalman filter not initialized — call step() with a candidate first")
        self._x = self.F @ self._x
        self._P = self.F @ self._P @ self.F.T + self.Q
        return self._x

    def _update(self, z: np.ndarray) -> float:
        if self._x is None or self._P is None:
            raise RuntimeError("Kalman filter not initialized")
        y = z - self.H @ self._x
        S = self.H @ self._P @ self.H.T + self.R
        K = self._P @ self.H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self.H) @ self._P
        return float(sqrt(float(y @ y)))

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._missed = 0

    def step(
        self, frame_idx: int, candidates: list[ShuttleObservation]
    ) -> ShuttleTrackPoint | None:
        """Advance one frame.

        Candidates: all shuttle bbox hypotheses detected in this frame (may be empty).
        Returns the selected track point, or None if no state and no candidates.
        """
        if not self.initialized:
            if not candidates:
                return None
            best = max(candidates, key=lambda o: o.confidence)
            self._initialize(best)
            if self._x is None:
                raise RuntimeError("Kalman filter failed to initialize")
            return ShuttleTrackPoint(
                frame_idx=frame_idx,
                x=float(self._x[0]),
                y=float(self._x[1]),
                vx=float(self._x[2]),
                vy=float(self._x[3]),
                interpolated=False,
                residual=0.0,
            )

        predicted = self._predict()
        associated: ShuttleObservation | None = None
        best_dist = self.association_max_distance
        for obs in candidates:
            d = sqrt((obs.x - predicted[0]) ** 2 + (obs.y - predicted[1]) ** 2)
            if d <= best_dist:
                best_dist = d
                associated = obs

        if associated is not None:
            residual = self._update(np.array([associated.x, associated.y]))
            self._missed = 0
            interpolated = False
        else:
            self._missed += 1
            residual = float("nan")
            interpolated = True
            if self._missed > self.max_missed_frames:
                self.reset()
                return None

        if self._x is None:
            raise RuntimeError("Kalman filter state is unexpectedly None")
        return ShuttleTrackPoint(
            frame_idx=frame_idx,
            x=float(self._x[0]),
            y=float(self._x[1]),
            vx=float(self._x[2]),
            vy=float(self._x[3]),
            interpolated=interpolated,
            residual=residual,
        )


def track_shuttle(
    observations_per_frame: dict[int, list[ShuttleObservation]],
    total_frames: int,
    **tracker_kwargs: Any,
) -> list[ShuttleTrackPoint]:
    """Run a single-object Kalman track across all frames."""
    tracker = ShuttleKalmanTracker(**tracker_kwargs)
    track: list[ShuttleTrackPoint] = []
    for frame_idx in range(total_frames):
        cand = observations_per_frame.get(frame_idx, [])
        point = tracker.step(frame_idx, cand)
        if point is not None:
            track.append(point)
    _log.info(
        "shuttle track: %d points (%d interpolated) across %d frames",
        len(track),
        sum(1 for p in track if p.interpolated),
        total_frames,
    )
    return track


def observations_from_detections(
    detections: Iterable[tuple[int, tuple[float, float, float, float], float]],
) -> dict[int, list[ShuttleObservation]]:
    """Helper: group `(frame_idx, bbox_xyxy, conf)` tuples into per-frame buckets."""
    out: dict[int, list[ShuttleObservation]] = {}
    for frame_idx, bbox, conf in detections:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        out.setdefault(frame_idx, []).append(
            ShuttleObservation(frame_idx=frame_idx, x=cx, y=cy, confidence=conf)
        )
    return out


