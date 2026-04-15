from pathlib import Path

import numpy as np
import pytest

from rallylens.pipeline.io import load_homography, save_homography
from rallylens.vision.court_homography import (
    COURT_CORNERS_M,
    COURT_LENGTH_M,
    SINGLES_WIDTH_M,
    compute_homography,
)


def test_compute_homography_round_trip():
    image_points = [(100.0, 50.0), (500.0, 50.0), (500.0, 350.0), (100.0, 350.0)]
    h = compute_homography(image_points)
    for img_pt, court_pt in zip(image_points, COURT_CORNERS_M.tolist(), strict=True):
        cx, cy = h.image_to_court(img_pt)
        assert cx == pytest.approx(court_pt[0], abs=1e-6)
        assert cy == pytest.approx(court_pt[1], abs=1e-6)


def test_homography_center_maps_to_court_center():
    image_points = [(100.0, 50.0), (500.0, 50.0), (500.0, 350.0), (100.0, 350.0)]
    h = compute_homography(image_points)
    cx, cy = h.image_to_court((300.0, 200.0))
    assert cx == pytest.approx(SINGLES_WIDTH_M / 2, abs=1e-3)
    assert cy == pytest.approx(COURT_LENGTH_M / 2, abs=1e-3)


def test_compute_homography_wrong_point_count():
    with pytest.raises(ValueError, match="4 image points"):
        compute_homography([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])


def test_save_load_homography_round_trip(tmp_path: Path):
    image_points = [(100.0, 50.0), (500.0, 50.0), (500.0, 350.0), (100.0, 350.0)]
    h = compute_homography(image_points)
    path = tmp_path / "h.json"
    save_homography(h, path)
    loaded = load_homography(path)
    assert np.allclose(loaded.H, h.H)
    assert loaded.image_points == h.image_points
