"""RallyLens CLI entry point (click group)."""

from __future__ import annotations

from pathlib import Path

import click

from rallylens import __version__
from rallylens.common import ensure_dir, get_logger, load_env, read_frame_at
from rallylens.config import (
    CALIBRATION_DIR,
    EVENTS_DIR,
    LABEL_FRAMES_DIR,
    LABELQA_DIR,
    RALLIES_DIR,
    RAW_DIR,
    REPORTS_DIR,
    TRACKS_DIR,
)
from rallylens.domain.video import is_likely_youtube_url
from rallylens.ingest.downloader import download_video
from rallylens.llm.label_qa import review_detections
from rallylens.llm.report_generator import generate_match_report
from rallylens.pipeline import (
    load_all_stats,
    load_video_meta,
    render_match_heatmaps,
    run_events_pipeline,
    run_full_pipeline,
)
from rallylens.pipeline.io import homography_path, save_homography, save_player_detections
from rallylens.preprocess.frame_sampler import sample_frames_from_rallies
from rallylens.preprocess.rally_segmenter import find_rally, load_manifest, segment_rallies
from rallylens.vision.court_homography import (
    compute_homography,
    pick_points_interactive,
)
from rallylens.vision.detect_track import coerce_tracker_name, detect_and_track_players
from rallylens.vision.shuttlecock_detector import detect_shuttlecocks

_log = get_logger(__name__)


@click.group()
@click.version_option(__version__)
def cli() -> None:
    load_env()


@cli.command("ingest")
@click.argument("url")
@click.option("--force/--no-force", default=False, help="Ignore cache and re-download.")
def ingest_cmd(url: str, force: bool) -> None:
    """Download a YouTube match to data/raw/{video_id}.mp4."""
    if not is_likely_youtube_url(url):
        raise click.ClickException(
            f"{url!r} does not look like a YouTube URL (watch/shorts/embed/youtu.be)"
        )
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
    manifest = load_manifest(RALLIES_DIR / video_id)
    try:
        target = find_rally(manifest, rally_index)
        frame = read_frame_at(target.path, frame_idx)
    except (LookupError, FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    import cv2  # local import — only needed for imwrite below

    frame_path = ensure_dir(CALIBRATION_DIR / video_id) / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(frame_path), frame)
    points = pick_points_interactive(frame_path)
    homography = compute_homography(points)
    out_path = homography_path(video_id)
    save_homography(homography, out_path)
    click.echo(f"saved homography -> {out_path}")


@cli.command("detect")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--tracker",
    type=click.Choice(["none", "bytetrack"]),
    default="bytetrack",
    show_default=True,
)
def detect_cmd(video_path: Path, tracker: str) -> None:
    """Run player pose tracking on a video file."""
    tracker_arg = coerce_tracker_name(None if tracker == "none" else tracker)
    player_detections = detect_and_track_players(video_path, tracker=tracker_arg)
    out_path = save_player_detections(player_detections, video_path.stem, video_path.stem)
    click.echo(f"detections: {out_path}")


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

    stats_list = run_events_pipeline(video_id, rallies)
    click.echo(
        f"wrote events for {len(stats_list)} rallies -> {EVENTS_DIR / video_id}"
    )


@cli.command("heatmaps")
@click.argument("video_id")
def heatmaps_cmd(video_id: str) -> None:
    """Render player/shuttle/rally-length heatmaps for a cached match (Week 3)."""
    out = render_match_heatmaps(video_id)
    if out is None:
        raise click.ClickException("nothing to render — run `events` first")
    click.echo(f"heatmaps: {out}")


@cli.command("report")
@click.argument("video_id")
def report_cmd(video_id: str) -> None:
    """Generate a Claude-written Markdown match report (Week 3)."""
    events_dir = EVENTS_DIR / video_id
    if not events_dir.exists():
        raise click.ClickException(f"no events at {events_dir}; run `events` first")
    stats_list = load_all_stats(video_id)
    if not stats_list:
        raise click.ClickException("no rally stats files found")

    meta_path = EVENTS_DIR / video_id / "video_meta.json"
    if not meta_path.exists():
        raise click.ClickException(
            f"no video_meta.json at {meta_path}; run `run` to regenerate"
        )
    meta = load_video_meta(video_id)
    out_path = ensure_dir(REPORTS_DIR / video_id) / "match_report.md"
    generate_match_report(meta, stats_list, out_path)
    click.echo(f"report: {out_path}")


@cli.command("label-qa")
@click.argument("video_id")
@click.option("--rally-index", type=int, default=None,
              help="Process a single rally by index. Omit to process all rallies.")
@click.option("--max-reviews", type=int, default=40, show_default=True)
def label_qa_cmd(video_id: str, rally_index: int | None, max_reviews: int) -> None:
    """Claude-reviewed shuttle label QA (Week 3). Flags suspicious predictions to JSONL.

    Processes all rallies by default. Use --rally-index N to process a single one.
    """
    manifest = load_manifest(RALLIES_DIR / video_id)
    if not manifest:
        raise click.ClickException(f"no rallies.json in {RALLIES_DIR / video_id}; run `segment` first")

    if rally_index is not None:
        try:
            targets = [find_rally(manifest, rally_index)]
        except LookupError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        targets = manifest

    for target in targets:
        click.echo(f"processing rally {target.index}/{len(manifest)} ...")
        detections = detect_shuttlecocks(target.path)
        if not detections:
            click.echo(f"  no shuttle detections in rally {target.index}, skipping")
            continue
        out_path = (
            ensure_dir(LABELQA_DIR / video_id) / f"rally_{target.index:03d}_label_qa.jsonl"
        )
        review_detections(target.path, detections, out_path, max_reviews=max_reviews)
        click.echo(f"label QA: {out_path}")


@cli.command("run")
@click.argument("url_or_path")
@click.option(
    "--tracker",
    type=click.Choice(["none", "bytetrack"]),
    default="bytetrack",
    show_default=True,
)
def run_cmd(url_or_path: str, tracker: str) -> None:
    """Download (or use local file) and run player tracking on a full match video."""
    tracker_arg = coerce_tracker_name(None if tracker == "none" else tracker)
    try:
        result = run_full_pipeline(url_or_path, tracker=tracker_arg)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"detections saved for {result.video_id}: {result.detections_path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
