"""RallyLens CLI entry point (click group)."""

from __future__ import annotations

from pathlib import Path

import click

from rallylens import __version__
from rallylens.common import get_logger, load_env
from rallylens.domain.video import is_likely_youtube_url
from rallylens.ingest.downloader import download_video
from rallylens.pipeline import run_full_pipeline
from rallylens.pipeline.io import save_player_detections
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
