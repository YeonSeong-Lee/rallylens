"""RallyLens CLI entry point (click group)."""

from __future__ import annotations

from pathlib import Path

import click

from rallylens import __version__
from rallylens.common import get_logger, load_env
from rallylens.domain.video import is_likely_youtube_url
from rallylens.ingest.downloader import download_video
from rallylens.pipeline import run_court_detection, run_court_detection_interactive, run_full_pipeline, run_shuttle_pipeline
from rallylens.pipeline.io import court_corners_path, save_player_detections, shuttle_track_path
from rallylens.vision.detect_track import coerce_tracker_name, detect_and_track_players

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


@cli.command("detect-shuttle")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--weights",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to TrackNet weights (.pth). Defaults to models/shuttle_tracknet.pth.",
)
def detect_shuttle_cmd(video_path: Path, weights: Path | None) -> None:
    """Run TrackNetV3 shuttlecock detection on a video file."""
    track = run_shuttle_pipeline(video_path, video_path.stem, weights)
    out_path = shuttle_track_path(video_path.stem, video_path.stem)
    click.echo(f"shuttle track ({len(track)} points): {out_path}")


@cli.command("calibrate")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--samples",
    type=int,
    default=20,
    show_default=True,
    help="Number of frames to sample for court detection.",
)
@click.option(
    "--interactive",
    is_flag=True,
    default=False,
    help="Open an OpenCV window to manually click or confirm court corners.",
)
def calibrate_cmd(video_path: Path, samples: int, interactive: bool) -> None:
    """Auto-detect badminton court corners and save calibration data."""
    if interactive:
        corners = run_court_detection_interactive(video_path, video_path.stem, sample_count=samples)
        if corners is None:
            raise click.ClickException("interactive calibration cancelled by user")
    else:
        corners = run_court_detection(video_path, video_path.stem, sample_count=samples)
        if corners is None:
            raise click.ClickException(
                "court corner detection failed — try a different video, increase --samples,"
                " or use --interactive to pick manually"
            )
    out_path = court_corners_path(video_path.stem)
    click.echo(f"court corners: {out_path}")


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
