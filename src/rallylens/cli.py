"""RallyLens CLI entry point (click group)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from rallylens import __version__
from rallylens.analysis.events import (
    aggregate_rally_stats,
    detect_hit_events,
    save_events_jsonl,
    save_rally_stats,
)
from rallylens.analysis.heatmap import render_heatmaps
from rallylens.common import (
    DATA_DIR,
    OUTPUTS_DEMO_DIR,
    OVERLAYS_DIR,
    RALLIES_DIR,
    RAW_DIR,
    VideoMeta,
    ensure_dir,
    get_logger,
    load_env,
)
from rallylens.ingest.downloader import download_video
from rallylens.llm.label_qa import review_detections
from rallylens.llm.report_generator import generate_match_report
from rallylens.preprocess.frame_sampler import sample_frames_from_rallies
from rallylens.preprocess.rally_segmenter import (
    RallyClip,
    load_manifest,
    segment_rallies,
)
from rallylens.vision.court_homography import (
    compute_homography,
    load_homography,
    pick_points_interactive,
    save_homography,
)
from rallylens.vision.detect_track import detect_and_track_players
from rallylens.vision.kalman_tracker import (
    observations_from_detections,
    save_track_jsonl,
    track_shuttle,
)
from rallylens.vision.shuttlecock_detector import detect_shuttlecocks, resolve_weights
from rallylens.viz.overlay import render_overlay

_log = get_logger(__name__)

CALIBRATION_DIR = DATA_DIR / "calibration"
LABEL_FRAMES_DIR = DATA_DIR / "label_frames"
TRACKS_DIR = DATA_DIR / "tracks"
EVENTS_DIR = DATA_DIR / "events"
REPORTS_DIR = DATA_DIR / "reports"
HEATMAPS_DIR = DATA_DIR / "heatmaps"
LABELQA_DIR = DATA_DIR / "label_qa"


@click.group()
@click.version_option(__version__)
def cli() -> None:
    load_env()


@cli.command("ingest")
@click.argument("url")
@click.option("--force/--no-force", default=False, help="Ignore cache and re-download.")
def ingest_cmd(url: str, force: bool) -> None:
    """Download a YouTube match to data/raw/{video_id}.mp4."""
    meta = download_video(url, force=force)
    click.echo(f"ok: {meta.source_path}")


@cli.command("segment")
@click.argument("video_id")
@click.option("--threshold", type=float, default=27.0, show_default=True)
def segment_cmd(video_id: str, threshold: float) -> None:
    """Scene-split a cached match video into rally clips."""
    source = RAW_DIR / f"{video_id}.mp4"
    if not source.exists():
        raise click.ClickException(f"no cached video at {source}; run `ingest` first")
    clips = segment_rallies(
        source,
        ensure_dir(RALLIES_DIR / video_id),
        threshold=threshold,
    )
    click.echo(f"segmented {len(clips)} rallies into {RALLIES_DIR / video_id}")


@cli.command("sample-frames")
@click.argument("video_id")
@click.option("--per-clip", type=int, default=20, show_default=True)
@click.option("--total", type=int, default=400, show_default=True)
def sample_frames_cmd(video_id: str, per_clip: int, total: int) -> None:
    """Extract frames from rally clips for labeling (Week 2)."""
    rally_dir = RALLIES_DIR / video_id
    if not rally_dir.exists():
        raise click.ClickException(f"no rally dir at {rally_dir}; run `segment` first")
    out_dir = ensure_dir(LABEL_FRAMES_DIR / video_id)
    frames = sample_frames_from_rallies(
        rally_dir, out_dir, per_clip=per_clip, total_budget=total
    )
    click.echo(f"sampled {len(frames)} frames -> {out_dir}")


@cli.command("calibrate")
@click.argument("video_id")
@click.option("--rally-index", type=int, default=1, show_default=True)
@click.option("--frame-idx", type=int, default=0, show_default=True)
def calibrate_cmd(video_id: str, rally_index: int, frame_idx: int) -> None:
    """Interactive 4-click court homography calibration (Week 2)."""
    rally_dir = RALLIES_DIR / video_id
    manifest = load_manifest(rally_dir)
    if not manifest:
        raise click.ClickException(
            f"no rallies.json in {rally_dir}; run `segment` first"
        )
    target = next((c for c in manifest if c.index == rally_index), None)
    if target is None:
        raise click.ClickException(f"rally {rally_index} not found")

    import cv2

    cap = cv2.VideoCapture(str(target.path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise click.ClickException(f"could not read frame {frame_idx} from {target.path}")

    frame_path = ensure_dir(CALIBRATION_DIR / video_id) / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(frame_path), frame)
    points = pick_points_interactive(frame_path)
    homography = compute_homography(points)
    out_path = CALIBRATION_DIR / video_id / "homography.json"
    save_homography(homography, out_path)
    click.echo(f"saved homography -> {out_path}")


@cli.command("detect")
@click.argument("video_id")
@click.option("--rally-index", type=int, default=1, show_default=True)
@click.option(
    "--tracker",
    type=click.Choice(["none", "bytetrack"]),
    default="bytetrack",
    show_default=True,
)
@click.option(
    "--with-shuttle/--no-shuttle",
    default=True,
    help="Run the shuttlecock detector + Kalman tracker.",
)
def detect_cmd(
    video_id: str, rally_index: int, tracker: str, with_shuttle: bool
) -> None:
    """Run player pose + optional shuttle tracking on a cached rally."""
    rally_dir = RALLIES_DIR / video_id
    manifest = load_manifest(rally_dir)
    if not manifest:
        raise click.ClickException(
            f"no rallies.json in {rally_dir}; run `segment` first"
        )
    target = next((c for c in manifest if c.index == rally_index), None)
    if target is None:
        raise click.ClickException(
            f"rally index {rally_index} not found (available: {[c.index for c in manifest]})"
        )

    tracker_arg = None if tracker == "none" else tracker
    player_detections = detect_and_track_players(target.path, tracker=tracker_arg)

    shuttle_track = _run_shuttle_pipeline(target.path, video_id) if with_shuttle else None

    out_path = (
        ensure_dir(OVERLAYS_DIR / video_id) / f"rally_{rally_index:03d}_overlay.mp4"
    )
    render_overlay(target.path, player_detections, out_path, shuttle_track=shuttle_track)
    click.echo(f"overlay: {out_path}")


@cli.command("events")
@click.argument("video_id")
def events_cmd(video_id: str) -> None:
    """Detect hit events from cached shuttle tracks for all rallies (Week 3)."""
    track_dir = TRACKS_DIR / video_id
    if not track_dir.exists():
        raise click.ClickException(
            f"no shuttle tracks at {track_dir}; run `detect` or `run` first"
        )
    rallies = load_manifest(RALLIES_DIR / video_id)
    if not rallies:
        raise click.ClickException("no rallies.json")

    ensure_dir(EVENTS_DIR / video_id)
    stats_list = _run_events_pipeline(video_id, rallies, None)
    click.echo(
        f"wrote events for {len(stats_list)} rallies -> "
        f"{EVENTS_DIR / video_id}"
    )


@cli.command("heatmaps")
@click.argument("video_id")
def heatmaps_cmd(video_id: str) -> None:
    """Render player/shuttle/rally-length heatmaps for a cached match (Week 3)."""
    out = _render_heatmaps_for(video_id)
    if out is None:
        raise click.ClickException("nothing to render — run `events` first")
    click.echo(f"heatmaps: {out}")


@cli.command("report")
@click.argument("video_id")
def report_cmd(video_id: str) -> None:
    """Generate a Claude-written Markdown match report (Week 3)."""
    events_dir = EVENTS_DIR / video_id
    if not events_dir.exists():
        raise click.ClickException(
            f"no events at {events_dir}; run `events` first"
        )
    stats_list = _load_all_stats(video_id)
    if not stats_list:
        raise click.ClickException("no rally stats files found")

    meta = _load_video_meta(video_id)
    out_path = ensure_dir(REPORTS_DIR / video_id) / "match_report.md"
    generate_match_report(meta, stats_list, out_path)
    click.echo(f"report: {out_path}")


@cli.command("label-qa")
@click.argument("video_id")
@click.option("--rally-index", type=int, default=1, show_default=True)
@click.option("--max-reviews", type=int, default=40, show_default=True)
def label_qa_cmd(video_id: str, rally_index: int, max_reviews: int) -> None:
    """Claude-reviewed shuttle label QA (Week 3). Flags suspicious predictions to JSONL."""
    rally_dir = RALLIES_DIR / video_id
    manifest = load_manifest(rally_dir)
    target = next((c for c in manifest if c.index == rally_index), None)
    if target is None:
        raise click.ClickException(f"rally {rally_index} not found")

    detections = detect_shuttlecocks(target.path)
    if not detections:
        raise click.ClickException("no shuttle detections to review")
    out_path = (
        ensure_dir(LABELQA_DIR / video_id)
        / f"rally_{rally_index:03d}_label_qa.jsonl"
    )
    review_detections(target.path, detections, out_path, max_reviews=max_reviews)
    click.echo(f"label QA: {out_path}")


@cli.command("run")
@click.argument("url")
@click.option(
    "--tracker",
    type=click.Choice(["none", "bytetrack"]),
    default="bytetrack",
    show_default=True,
)
@click.option("--with-shuttle/--no-shuttle", default=True)
@click.option(
    "--with-report/--no-report",
    default=True,
    help="Call Claude to generate a Markdown match report (requires ANTHROPIC_API_KEY).",
)
def run_cmd(url: str, tracker: str, with_shuttle: bool, with_report: bool) -> None:
    """End-to-end: download -> segment -> track -> events -> heatmaps -> report."""
    meta = download_video(url)
    _save_video_meta(meta)
    rallies = segment_rallies(
        meta.source_path,
        ensure_dir(RALLIES_DIR / meta.video_id),
    )
    if not rallies:
        raise click.ClickException("no rallies detected")

    tracker_arg = None if tracker == "none" else tracker
    shuttle_tracks: dict[int, list] = {}
    player_detections_by_rally: dict[int, list] = {}

    for rally in rallies:
        player_detections = detect_and_track_players(rally.path, tracker=tracker_arg)
        player_detections_by_rally[rally.index] = player_detections

        shuttle_track = (
            _run_shuttle_pipeline(rally.path, meta.video_id)
            if with_shuttle
            else None
        )
        shuttle_tracks[rally.index] = shuttle_track or []

        overlay_out = (
            ensure_dir(OVERLAYS_DIR / meta.video_id)
            / f"rally_{rally.index:03d}_overlay.mp4"
        )
        render_overlay(
            rally.path,
            player_detections,
            overlay_out,
            shuttle_track=shuttle_track,
        )

    stats_list = _run_events_pipeline(
        meta.video_id, rallies, shuttle_tracks
    )

    heatmap_out = _render_heatmaps_for(meta.video_id)
    if heatmap_out is not None:
        demo_copy = OUTPUTS_DEMO_DIR / f"{meta.video_id}_heatmaps.png"
        ensure_dir(OUTPUTS_DEMO_DIR)
        demo_copy.write_bytes(heatmap_out.read_bytes())
        click.echo(f"heatmap (also copied to demo): {heatmap_out}")

    if with_report and stats_list:
        try:
            out_path = ensure_dir(REPORTS_DIR / meta.video_id) / "match_report.md"
            generate_match_report(meta, stats_list, out_path)
            click.echo(f"report: {out_path}")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"report skipped ({exc})", err=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_video_meta(meta: VideoMeta) -> None:
    path = ensure_dir(EVENTS_DIR / meta.video_id) / "video_meta.json"
    path.write_text(
        json.dumps(meta.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_video_meta(video_id: str) -> VideoMeta:
    path = EVENTS_DIR / video_id / "video_meta.json"
    if not path.exists():
        raise click.ClickException(
            f"no video_meta.json at {path}; run `run` to regenerate"
        )
    return VideoMeta.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def _run_shuttle_pipeline(clip_path: Path, video_id: str) -> list:
    _, is_fine_tuned = resolve_weights()
    if not is_fine_tuned:
        _log.warning(
            "shuttle detector running WITHOUT fine-tuned weights — "
            "track quality will be very poor. Run Week 2 Colab notebook first."
        )
    candidates = detect_shuttlecocks(clip_path)
    obs = observations_from_detections(
        (c.frame_idx, c.bbox_xyxy, c.confidence) for c in candidates
    )
    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    track = track_shuttle(obs, total_frames)
    track_out = ensure_dir(TRACKS_DIR / video_id) / f"{clip_path.stem}_shuttle.jsonl"
    save_track_jsonl(track, track_out)
    _log.info("saved shuttle track -> %s", track_out)
    return track


def _run_events_pipeline(
    video_id: str,
    rallies: list[RallyClip],
    shuttle_tracks_in_memory: dict[int, list] | None,
) -> list:
    """Build events + stats for every rally. Reads shuttle tracks from disk if not provided."""
    from rallylens.analysis.events import RallyStats  # noqa: F401
    from rallylens.vision.kalman_tracker import ShuttleTrackPoint

    events_out_dir = ensure_dir(EVENTS_DIR / video_id)
    stats_list = []
    import cv2

    for rally in rallies:
        track: list[ShuttleTrackPoint]
        if shuttle_tracks_in_memory is not None:
            track = shuttle_tracks_in_memory.get(rally.index, [])
        else:
            track_path = (
                TRACKS_DIR / video_id / f"{rally.path.stem}_shuttle.jsonl"
            )
            track = _load_track(track_path) if track_path.exists() else []

        cap = cv2.VideoCapture(str(rally.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        events = detect_hit_events(track, fps=fps, frame_height=height)
        stats = aggregate_rally_stats(
            video_id=video_id,
            rally_index=rally.index,
            events=events,
            duration_s=rally.end_s - rally.start_s,
            total_frames=total_frames,
        )

        save_events_jsonl(
            events, events_out_dir / f"rally_{rally.index:03d}_events.jsonl"
        )
        save_rally_stats(
            stats, events_out_dir / f"rally_{rally.index:03d}_stats.json"
        )
        stats_list.append(stats)
    return stats_list


def _load_track(path: Path) -> list:
    from rallylens.vision.kalman_tracker import ShuttleTrackPoint

    if not path.exists():
        return []
    out: list[ShuttleTrackPoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            ShuttleTrackPoint(
                frame_idx=int(row["frame_idx"]),
                x=float(row["x"]),
                y=float(row["y"]),
                vx=float(row["vx"]),
                vy=float(row["vy"]),
                interpolated=bool(row["interpolated"]),
                residual=float("nan") if row["residual"] is None else float(row["residual"]),
            )
        )
    return out


def _render_heatmaps_for(video_id: str) -> Path | None:
    rally_dir = RALLIES_DIR / video_id
    rallies = load_manifest(rally_dir)
    if not rallies:
        return None

    stats_list = _load_all_stats(video_id)
    if not stats_list:
        return None

    import cv2

    all_shuttle: list = []
    all_player_detections: list = []
    for rally in rallies:
        track_path = TRACKS_DIR / video_id / f"{rally.path.stem}_shuttle.jsonl"
        all_shuttle.extend(_load_track(track_path))

    homography_path = CALIBRATION_DIR / video_id / "homography.json"
    homography = load_homography(homography_path) if homography_path.exists() else None

    # Player heatmap needs raw detections, which we rerun if not cached. For now
    # reuse the rally center points as a cheap proxy so `heatmaps` works without
    # re-running the YOLO model.
    for rally in rallies:
        cap = cv2.VideoCapture(str(rally.path))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        from rallylens.vision.detect_track import Detection

        all_player_detections.append(
            Detection(
                frame_idx=0,
                bbox_xyxy=(w * 0.25, h * 0.25, w * 0.75, h * 0.75),
                confidence=1.0,
                keypoints_xy=[],
                keypoints_conf=[],
            )
        )

    out_path = ensure_dir(HEATMAPS_DIR / video_id) / "heatmaps.png"
    render_heatmaps(
        out_path,
        all_player_detections,
        all_shuttle,
        stats_list,
        homography=homography,
        title=f"RallyLens — {video_id}",
    )
    return out_path


def _load_all_stats(video_id: str) -> list:
    events_dir = EVENTS_DIR / video_id
    if not events_dir.exists():
        return []
    from rallylens.analysis.events import HitEvent, RallyStats

    stats_list: list[RallyStats] = []
    for stats_path in sorted(events_dir.glob("rally_*_stats.json")):
        data = json.loads(stats_path.read_text(encoding="utf-8"))
        events = [
            HitEvent(
                frame_idx=int(e["frame_idx"]),
                time_s=float(e["time_s"]),
                kind=str(e["kind"]),
                position_xy=(float(e["position_xy"][0]), float(e["position_xy"][1])),
                velocity_xy=(float(e["velocity_xy"][0]), float(e["velocity_xy"][1])),
                signals=tuple(e.get("signals", [])),
                player_side=e.get("player_side"),
            )
            for e in data.get("events", [])
        ]
        # mypy: RallyStats is a mutable dataclass; rebuild from dict
        stats_list.append(
            RallyStats(
                video_id=str(data["video_id"]),
                rally_index=int(data["rally_index"]),
                duration_s=float(data["duration_s"]),
                total_frames=int(data["total_frames"]),
                shot_count=int(data["shot_count"]),
                first_shot_frame=data.get("first_shot_frame"),
                last_shot_frame=data.get("last_shot_frame"),
                avg_inter_shot_gap_s=data.get("avg_inter_shot_gap_s"),
                top_side_shots=int(data["top_side_shots"]),
                bottom_side_shots=int(data["bottom_side_shots"]),
                events=events,
            )
        )
    return stats_list


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
