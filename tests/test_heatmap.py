from pathlib import Path

from rallylens.analysis.events import RallyStats
from rallylens.analysis.heatmap import render_heatmaps
from rallylens.vision.detect_track import Detection


def test_render_heatmaps_writes_png(tmp_path: Path):
    stats = [
        RallyStats(
            video_id="v",
            rally_index=1,
            duration_s=10.0,
            total_frames=300,
            shot_count=5,
            first_shot_frame=10,
            last_shot_frame=280,
            avg_inter_shot_gap_s=2.0,
            top_side_shots=3,
            bottom_side_shots=2,
        )
    ]
    detections = [
        Detection(
            frame_idx=i,
            bbox_xyxy=(200.0, 200.0, 300.0, 400.0),
            confidence=0.9,
            keypoints_xy=[],
            keypoints_conf=[],
        )
        for i in range(5)
    ]
    out_path = tmp_path / "heatmaps.png"
    result = render_heatmaps(
        out_path, detections, shuttle_track=[], rally_stats=stats
    )
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # sanity — real PNG is much bigger than 1KB
