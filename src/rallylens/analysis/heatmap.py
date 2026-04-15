"""Match-level visualizations: player positions, shuttle trajectory, rally lengths."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from rallylens.analysis.events import RallyStats
from rallylens.common import ensure_dir, get_logger
from rallylens.vision.court_homography import (
    COURT_CORNERS_M,
    COURT_LENGTH_M,
    SINGLES_WIDTH_M,
    CourtHomography,
)
from rallylens.vision.detect_track import Detection
from rallylens.vision.kalman_tracker import ShuttleTrackPoint

_log = get_logger(__name__)


def _player_positions_court(
    detections: list[Detection], homography: CourtHomography | None
) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for d in detections:
        cx = (d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2
        cy = (d.bbox_xyxy[1] + d.bbox_xyxy[3]) / 2
        if homography is not None:
            x, y = homography.image_to_court((cx, cy))
        else:
            x, y = cx, cy
        pts.append((x, y))
    return np.array(pts) if pts else np.empty((0, 2))


def _shuttle_positions_court(
    track: list[ShuttleTrackPoint], homography: CourtHomography | None
) -> np.ndarray:
    if not track:
        return np.empty((0, 2))
    pts = []
    for p in track:
        if homography is not None:
            x, y = homography.image_to_court((p.x, p.y))
        else:
            x, y = p.x, p.y
        pts.append((x, y))
    return np.array(pts)


def _draw_court(ax: plt.Axes) -> None:
    corners = np.vstack([COURT_CORNERS_M, COURT_CORNERS_M[:1]])
    ax.plot(corners[:, 0], corners[:, 1], "-", color="white", linewidth=2)
    ax.axhline(COURT_LENGTH_M / 2, color="white", linewidth=1.5)  # net
    ax.set_xlim(-0.5, SINGLES_WIDTH_M + 0.5)
    ax.set_ylim(COURT_LENGTH_M + 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#2a7f3b")
    ax.set_xticks([])
    ax.set_yticks([])


def render_heatmaps(
    out_path: Path,
    player_detections: list[Detection],
    shuttle_track: list[ShuttleTrackPoint],
    rally_stats: list[RallyStats],
    homography: CourtHomography | None = None,
    title: str | None = None,
) -> Path:
    """Write a 1x3 panel png: player heatmap | shuttle trajectory | rally length hist."""
    ensure_dir(out_path.parent)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title or "RallyLens — match analysis", fontsize=14)

    player_pts = _player_positions_court(player_detections, homography)
    ax = axes[0]
    _draw_court(ax)
    if len(player_pts) > 0 and homography is not None:
        ax.hist2d(
            player_pts[:, 0],
            player_pts[:, 1],
            bins=(16, 30),
            range=[[0, SINGLES_WIDTH_M], [0, COURT_LENGTH_M]],
            cmap="Reds",
            alpha=0.85,
            cmin=1,
        )
    elif len(player_pts) > 0:
        ax.scatter(player_pts[:, 0], player_pts[:, 1], s=2, c="red", alpha=0.4)
    ax.set_title(
        f"Player positions  (n={len(player_pts)})",
        fontsize=11,
    )

    shuttle_pts = _shuttle_positions_court(shuttle_track, homography)
    ax = axes[1]
    _draw_court(ax)
    if len(shuttle_pts) > 0:
        ax.plot(
            shuttle_pts[:, 0],
            shuttle_pts[:, 1],
            "-",
            color="yellow",
            linewidth=1,
            alpha=0.6,
        )
        ax.scatter(
            shuttle_pts[:, 0], shuttle_pts[:, 1], s=4, c="yellow", edgecolors="none"
        )
    ax.set_title(
        f"Shuttle trajectory  (n={len(shuttle_pts)})",
        fontsize=11,
    )

    ax = axes[2]
    durations = [r.duration_s for r in rally_stats]
    shot_counts = [r.shot_count for r in rally_stats]
    if durations:
        ax.hist(durations, bins=min(20, max(5, len(durations))), color="#4c72b0", edgecolor="black")
        ax.set_xlabel("Rally duration (s)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Rally durations  (n={len(durations)}, total shots={sum(shot_counts)})",
            fontsize=11,
        )
    else:
        ax.text(0.5, 0.5, "no rallies", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Rally durations", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    _log.info("wrote heatmap panel -> %s", out_path)
    return out_path
