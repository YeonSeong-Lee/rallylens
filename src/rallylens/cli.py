"""RallyLens CLI entry point (click group)."""

from __future__ import annotations

from pathlib import Path

import click

from rallylens import __version__
from rallylens.common import get_logger, load_env
from rallylens.domain.video import is_likely_youtube_url
from rallylens.ingest.downloader import download_video, parse_time
from rallylens.pipeline import (
    run_court_detection,
    run_court_detection_interactive,
    run_full_pipeline,
    run_shuttle_pipeline,
)
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
@click.option(
    "--start",
    "start_time",
    default=None,
    help="Start time: seconds (90) or MM:SS / HH:MM:SS (1:30).",
)
@click.option(
    "--end",
    "end_time",
    default=None,
    help="End time: seconds (120) or MM:SS / HH:MM:SS (2:00).",
)
def ingest_cmd(url: str, force: bool, start_time: str | None, end_time: str | None) -> None:
    """Download a YouTube match to data/raw/{video_id}.mp4.

    Use --start / --end to download only a specific time range.
    The clip is saved as {video_id}_{start}s_{end}s.mp4 so full and clipped
    downloads can coexist in the cache.
    """
    if not is_likely_youtube_url(url):
        raise click.ClickException(
            f"{url!r} does not look like a YouTube URL (watch/shorts/embed/youtu.be)"
        )
    start_s = parse_time(start_time) if start_time is not None else None
    end_s = parse_time(end_time) if end_time is not None else None
    if start_s is not None and end_s is not None and end_s <= start_s:
        raise click.ClickException("--end must be greater than --start")
    meta = download_video(url, force=force, start_s=start_s, end_s=end_s)
    click.echo(f"ok: {meta.source_path}")


@cli.command("detect")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--tracker",
    type=click.Choice(["none", "bytetrack"]),
    default="bytetrack",
    show_default=True,
)
@click.option(
    "--singles/--no-singles",
    default=True,
    show_default=True,
    help="Keep only the 2 most-stable track IDs (1v1 singles matches).",
)
@click.option(
    "--imgsz",
    type=int,
    default=1280,
    show_default=True,
    help="YOLO inference image size (larger = better small-object recall).",
)
def detect_cmd(video_path: Path, tracker: str, singles: bool, imgsz: int) -> None:
    """Run player pose tracking on a video file."""
    tracker_arg = coerce_tracker_name(None if tracker == "none" else tracker)
    player_detections = detect_and_track_players(
        video_path, tracker=tracker_arg, singles=singles, imgsz=imgsz
    )
    out_path = save_player_detections(player_detections, video_path.stem)
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
    out_path = shuttle_track_path(video_path.stem)
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
@click.option(
    "--singles/--no-singles",
    default=True,
    show_default=True,
    help="Keep only the 2 most-stable track IDs (1v1 singles matches).",
)
@click.option(
    "--imgsz",
    type=int,
    default=1280,
    show_default=True,
    help="YOLO inference image size.",
)
def run_cmd(url_or_path: str, tracker: str, singles: bool, imgsz: int) -> None:
    """Download (or use local file) and run player tracking on a full match video."""
    tracker_arg = coerce_tracker_name(None if tracker == "none" else tracker)
    try:
        result = run_full_pipeline(url_or_path, tracker=tracker_arg, singles=singles, imgsz=imgsz)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"detections saved for {result.video_id}: {result.detections_path}")


@cli.command("report")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--model",
    default="gemini-2.5-flash",
    show_default=True,
    help="Vertex AI Gemini model ID.",
)
@click.option("--temperature", type=float, default=0.4, show_default=True)
@click.option(
    "--metrics-only",
    is_flag=True,
    default=False,
    help="Skip the LLM call and only produce metrics.json (no Vertex AI creds needed).",
)
@click.option(
    "--skip-viz",
    is_flag=True,
    default=False,
    help="Skip auto-rendering of court diagram GIF and heatmap PNG.",
)
def report_cmd(
    video_path: Path,
    model: str,
    temperature: float,
    metrics_only: bool,
    skip_viz: bool,
) -> None:
    """Generate a Korean rally analysis report via Vertex AI Gemini.

    Requires existing detections and court corners artifacts — run
    `rallylens detect` and `rallylens calibrate` first. Shuttle tracking
    (`rallylens detect-shuttle`) is optional but strongly recommended.
    """
    from rallylens.pipeline import run_report_pipeline

    try:
        result = run_report_pipeline(
            video_path,
            model=model,
            temperature=temperature,
            metrics_only=metrics_only,
            skip_viz=skip_viz,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"metrics: {result.metrics_path}")
    if result.court_diagram_path is not None:
        click.echo(f"gif:     {result.court_diagram_path}")
    if result.heatmap_path is not None:
        click.echo(f"heatmap: {result.heatmap_path}")
    if result.report_md_path is not None:
        click.echo(f"report:  {result.report_md_path}")


@cli.command("viz")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--overlay/--no-overlay", default=True, show_default=True, help="Render video overlay MP4.")
@click.option("--heatmap/--no-heatmap", default=True, show_default=True, help="Render position heatmap PNG.")
@click.option("--court/--no-court", default=True, show_default=True, help="Render court trajectory diagram GIF.")
@click.option("--trail-len", type=int, default=30, show_default=True, help="Shuttle trail length in frames.")
@click.option("--court-stride", type=int, default=5, show_default=True, help="Emit every Nth source frame to the court GIF.")
@click.option("--court-scale", type=float, default=0.5, show_default=True, help="Downscale factor for the court GIF.")
def viz_cmd(
    video_path: Path,
    overlay: bool,
    heatmap: bool,
    court: bool,
    trail_len: int,
    court_stride: int,
    court_scale: float,
) -> None:
    """Render visualization outputs for a processed video.

    Loads existing detections, shuttle track, and court corners from the data
    directory and generates up to three outputs: a video overlay (MP4), a
    position heatmap (PNG), and an animated court trajectory diagram (GIF).
    """
    from rallylens.common import read_video_properties
    from rallylens.pipeline.io import (
        load_court_corners,
        load_player_detections,
        load_shuttle_track,
        viz_court_diagram_path,
        viz_heatmap_path,
        viz_overlay_path,
    )
    from rallylens.viz import render_court_diagram, render_heatmap, render_overlay_video

    video_id = video_path.stem

    detections = load_player_detections(video_id)
    shuttle_track = load_shuttle_track(video_id)

    if overlay:
        out = render_overlay_video(
            video_path,
            detections,
            shuttle_track,
            viz_overlay_path(video_id),
            trail_len=trail_len,
        )
        click.echo(f"overlay:  {out}")

    if heatmap or court:
        try:
            corners = load_court_corners(video_id)
        except FileNotFoundError as exc:
            raise click.ClickException(
                f"court calibration not found for {video_id!r} — run `rallylens calibrate` first"
            ) from exc

        if heatmap:
            out = render_heatmap(
                detections,
                shuttle_track,
                corners,
                viz_heatmap_path(video_id),
            )
            click.echo(f"heatmap:  {out}")

        if court:
            props = read_video_properties(video_path)
            out = render_court_diagram(
                detections,
                shuttle_track,
                corners,
                viz_court_diagram_path(video_id),
                fps=props.fps,
                stride=court_stride,
                scale=court_scale,
            )
            click.echo(f"court:    {out}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
