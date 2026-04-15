"""Court homography: 4-point image->court (meters) transform.

Badminton singles court dimensions (BWF standard):
    length = 13.40 m  (baseline to baseline)
    singles width = 5.18 m  (sideline to sideline)

The 4 clicked points are the four corners of the SINGLES playing area as
visible in the match broadcast, in a fixed order:
    0: top-left (far-left baseline corner as seen by the camera)
    1: top-right (far-right baseline corner)
    2: bottom-right (near-right baseline corner)
    3: bottom-left (near-left baseline corner)

Court coordinate convention:
    x in [0, 5.18] meters (left = 0)
    y in [0, 13.40] meters (far baseline = 0, near baseline = 13.40)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rallylens.common import get_logger

_log = get_logger(__name__)

COURT_LENGTH_M = 13.40
SINGLES_WIDTH_M = 5.18

COURT_CORNERS_M: np.ndarray = np.array(
    [
        [0.0, 0.0],                # top-left (far baseline, left sideline)
        [SINGLES_WIDTH_M, 0.0],    # top-right
        [SINGLES_WIDTH_M, COURT_LENGTH_M],  # bottom-right (near baseline, right)
        [0.0, COURT_LENGTH_M],     # bottom-left (near baseline, left)
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CourtHomography:
    image_points: list[tuple[float, float]]
    court_points_m: list[tuple[float, float]]
    H: np.ndarray  # 3x3 image -> court

    def image_to_court(self, xy: tuple[float, float]) -> tuple[float, float]:
        pt = np.array([xy[0], xy[1], 1.0])
        out = self.H @ pt
        return float(out[0] / out[2]), float(out[1] / out[2])

    def to_json_dict(self) -> dict[str, object]:
        return {
            "image_points": [list(p) for p in self.image_points],
            "court_points_m": [list(p) for p in self.court_points_m],
            "H": self.H.tolist(),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> CourtHomography:
        return cls(
            image_points=[tuple(p) for p in data["image_points"]],  # type: ignore[misc]
            court_points_m=[tuple(p) for p in data["court_points_m"]],  # type: ignore[misc]
            H=np.array(data["H"], dtype=np.float64),
        )


def compute_homography(
    image_points: list[tuple[float, float]],
    court_points_m: np.ndarray = COURT_CORNERS_M,
) -> CourtHomography:
    if len(image_points) != 4:
        raise ValueError(f"expected 4 image points, got {len(image_points)}")
    src = np.array(image_points, dtype=np.float64)
    dst = court_points_m
    H, _mask = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed (points may be collinear)")
    return CourtHomography(
        image_points=[tuple(p) for p in image_points],
        court_points_m=[tuple(p) for p in dst.tolist()],
        H=H,
    )


def save_homography(h: CourtHomography, path: Path) -> None:
    path.write_text(json.dumps(h.to_json_dict(), indent=2), encoding="utf-8")


def load_homography(path: Path) -> CourtHomography:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CourtHomography.from_json_dict(data)


def pick_points_interactive(frame_path: Path) -> list[tuple[float, float]]:
    """Open a preview window; user clicks 4 court corners in TL,TR,BR,BL order.

    Not used by tests — requires a display + mouse. CLI calls this for calibration.
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(frame_path)

    display = frame.copy()
    clicks: list[tuple[float, float]] = []
    labels = ("TL", "TR", "BR", "BL")
    window = "rallylens calibrate — click TL, TR, BR, BL"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= 4:
            return
        clicks.append((float(x), float(y)))
        cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            display,
            labels[len(clicks) - 1],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, display)
    cv2.setMouseCallback(window, on_mouse)
    while len(clicks) < 4:
        if cv2.waitKey(20) & 0xFF == 27:  # esc to abort
            break
    cv2.destroyWindow(window)
    if len(clicks) != 4:
        raise RuntimeError("calibration aborted before 4 points")
    _log.info("picked points: %s", clicks)
    return clicks
