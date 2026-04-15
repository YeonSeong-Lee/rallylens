"""Hit-event detection from shuttle Kalman track + player detections.

Two signals combined:

1. **Velocity vector direction reversal** — dot product of consecutive velocity
   vectors goes negative. This happens when the shuttle bounces off a racket.
   Using *magnitude* changes is insufficient because gravity + drag also cause
   speed changes between hits; only direction reversal is specific to a strike.

2. **Kalman residual spike** — when the actual shuttle position diverges
   sharply from the Kalman prediction (z-score threshold over the running mean
   residual). Residuals spike at hits because the constant-velocity motion model
   breaks down during the racket impact.

The two are combined with OR by default (either signal triggers a hit), but
with a refractory window (min_gap_frames) to collapse duplicate triggers from
adjacent frames.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from rallylens.common import get_logger
from rallylens.vision.kalman_tracker import ShuttleTrackPoint

_log = get_logger(__name__)


class HitEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_idx: int
    time_s: float
    kind: str  # "hit"
    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    signals: tuple[str, ...]
    player_side: str | None  # "top" | "bottom" | None


class RallyStats(BaseModel):
    video_id: str
    rally_index: int
    duration_s: float
    total_frames: int
    shot_count: int
    first_shot_frame: int | None
    last_shot_frame: int | None
    avg_inter_shot_gap_s: float | None
    top_side_shots: int
    bottom_side_shots: int
    events: list[HitEvent] = Field(default_factory=list)


def _velocity_direction_reversals(
    track: list[ShuttleTrackPoint],
    min_speed: float = 1.0,
) -> set[int]:
    """Return frame indices where the velocity vector flipped direction."""
    hits: set[int] = set()
    for prev, curr in zip(track, track[1:], strict=False):
        speed_prev = math.hypot(prev.vx, prev.vy)
        speed_curr = math.hypot(curr.vx, curr.vy)
        if speed_prev < min_speed or speed_curr < min_speed:
            continue
        dot = prev.vx * curr.vx + prev.vy * curr.vy
        if dot < 0:
            hits.add(curr.frame_idx)
    return hits


def _residual_spikes(
    track: list[ShuttleTrackPoint],
    z_threshold: float = 3.0,
) -> set[int]:
    """Return frame indices where the Kalman residual exceeded mean + z*std."""
    residuals = [p.residual for p in track if not p.interpolated and not math.isnan(p.residual)]
    if len(residuals) < 5:
        return set()
    mean_r = sum(residuals) / len(residuals)
    var_r = sum((r - mean_r) ** 2 for r in residuals) / max(1, len(residuals) - 1)
    std_r = math.sqrt(var_r) if var_r > 0 else 0.0
    threshold = mean_r + z_threshold * std_r
    return {
        p.frame_idx
        for p in track
        if not p.interpolated
        and not math.isnan(p.residual)
        and p.residual > threshold
    }


def _side_from_position(
    y: float,
    frame_height: int | None = None,
    threshold: float = 0.5,
) -> str | None:
    """Return "top" if in the upper half of the frame, "bottom" otherwise."""
    if frame_height is None:
        return None
    return "top" if y < frame_height * threshold else "bottom"


def detect_hit_events(
    track: list[ShuttleTrackPoint],
    fps: float,
    min_gap_frames: int = 5,
    frame_height: int | None = None,
    z_threshold: float = 3.0,
) -> list[HitEvent]:
    if len(track) < 3:
        return []

    reversals = _velocity_direction_reversals(track)
    spikes = _residual_spikes(track, z_threshold=z_threshold)
    combined = sorted(reversals | spikes)
    _log.info(
        "event signals: %d direction reversals, %d residual spikes, %d union",
        len(reversals),
        len(spikes),
        len(combined),
    )

    by_frame = {p.frame_idx: p for p in track}
    events: list[HitEvent] = []
    last_frame = -(min_gap_frames + 1)
    for frame_idx in combined:
        if frame_idx - last_frame < min_gap_frames:
            continue
        point = by_frame.get(frame_idx)
        if point is None:
            continue
        signals: list[str] = []
        if frame_idx in reversals:
            signals.append("velocity_reversal")
        if frame_idx in spikes:
            signals.append("residual_spike")
        events.append(
            HitEvent(
                frame_idx=frame_idx,
                time_s=frame_idx / fps if fps > 0 else 0.0,
                kind="hit",
                position_xy=(point.x, point.y),
                velocity_xy=(point.vx, point.vy),
                signals=tuple(signals),
                player_side=_side_from_position(point.y, frame_height),
            )
        )
        last_frame = frame_idx
    return events


def aggregate_rally_stats(
    video_id: str,
    rally_index: int,
    events: list[HitEvent],
    duration_s: float,
    total_frames: int,
) -> RallyStats:
    top = sum(1 for e in events if e.player_side == "top")
    bottom = sum(1 for e in events if e.player_side == "bottom")
    if len(events) >= 2:
        gaps = [
            events[i].time_s - events[i - 1].time_s for i in range(1, len(events))
        ]
        avg_gap = sum(gaps) / len(gaps)
    else:
        avg_gap = None
    return RallyStats(
        video_id=video_id,
        rally_index=rally_index,
        duration_s=duration_s,
        total_frames=total_frames,
        shot_count=len(events),
        first_shot_frame=events[0].frame_idx if events else None,
        last_shot_frame=events[-1].frame_idx if events else None,
        avg_inter_shot_gap_s=avg_gap,
        top_side_shots=top,
        bottom_side_shots=bottom,
        events=events,
    )


