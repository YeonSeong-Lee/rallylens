"""yt-dlp wrapper: downloads YouTube badminton videos at <=720p/30fps mp4."""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

from rallylens.common import ensure_dir, get_logger
from rallylens.config import RAW_DIR
from rallylens.domain.video import VideoMeta, video_id_from_url
from rallylens.serialization import load_json, save_json

_log = get_logger(__name__)

_FORMAT_SELECTOR = (
    "bv*[height<=720][fps<=30][ext=mp4]+ba[ext=m4a]/"
    "b[height<=720][ext=mp4]/"
    "b[height<=720]"
)


def parse_time(value: str | float | int) -> float:
    """Parse a time value to seconds.

    Accepts:
    - A plain number (int or float) as seconds: ``90``, ``90.5``
    - ``"MM:SS"`` or ``"HH:MM:SS"`` string: ``"1:30"``, ``"0:01:30"``
    """
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(value)


def _clip_suffix(start_s: float, end_s: float) -> str:
    return f"_{int(start_s)}s_{int(end_s)}s"


def _sidecar_path(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}.meta.json"


def _video_path(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}.mp4"


def _read_cached_meta(out_dir: Path, video_id: str) -> VideoMeta | None:
    sidecar = _sidecar_path(out_dir, video_id)
    video = _video_path(out_dir, video_id)
    if not (sidecar.exists() and video.exists()):
        return None
    try:
        return load_json(sidecar, VideoMeta)
    except (OSError, ValueError):
        return None


def _write_sidecar(meta: VideoMeta) -> None:
    save_json(meta, _sidecar_path(meta.source_path.parent, meta.video_id))


def download_video(
    url: str,
    out_dir: Path = RAW_DIR,
    force: bool = False,
    start_s: float | None = None,
    end_s: float | None = None,
) -> VideoMeta:
    """Download a YouTube video at <=720p/30fps mp4 and return metadata.

    When *start_s* and/or *end_s* are given (seconds), only that time range
    is downloaded and stored as ``{video_id}_{start}s_{end}s.mp4``.  This lets
    multiple clips from the same video coexist in the cache.

    Cache: if the target ``.mp4`` and ``.meta.json`` files both exist and
    ``force=False``, the sidecar is read and no network call is made.
    """
    ensure_dir(out_dir)
    base_id = video_id_from_url(url)

    clipping = start_s is not None or end_s is not None
    clip_start = start_s if start_s is not None else 0.0
    clip_end = end_s  # None means "to end of video" for yt-dlp

    # Unique ID for this clip so it doesn't collide with the full download.
    if clipping:
        end_label = int(clip_end) if clip_end is not None else "end"
        clip_id = f"{base_id}_{int(clip_start)}s_{end_label}s"
    else:
        clip_id = base_id

    if not force:
        cached = _read_cached_meta(out_dir, clip_id)
        if cached is not None:
            _log.info("cache hit for %s (%s)", clip_id, cached.title)
            return cached

    ydl_opts: dict = {
        "format": _FORMAT_SELECTOR,
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{clip_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": force,
    }

    if clipping:
        range_end = clip_end if clip_end is not None else float("inf")
        ydl_opts["download_ranges"] = download_range_func(None, [(clip_start, range_end)])
        ydl_opts["force_keyframes_at_cuts"] = True
        _log.info(
            "downloading %s [%ss → %s] with format=%s",
            url,
            clip_start,
            f"{clip_end}s" if clip_end is not None else "end",
            _FORMAT_SELECTOR,
        )
    else:
        _log.info("downloading %s with format=%s", url, _FORMAT_SELECTOR)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if info is None:
        raise RuntimeError(f"yt-dlp returned no info for {url}")

    video_path = _video_path(out_dir, clip_id)
    if not video_path.exists():
        raise RuntimeError(
            f"expected {video_path} to exist after download — no compatible format?"
        )

    clip_duration = (
        (clip_end - clip_start) if clip_end is not None
        else float(info.get("duration") or 0.0) - clip_start
    )
    meta = VideoMeta(
        video_id=clip_id,
        title=str(info.get("title") or ""),
        upload_date=(str(info["upload_date"]) if info.get("upload_date") else None),
        duration_s=clip_duration if clipping else float(info.get("duration") or 0.0),
        url=url,
        source_path=video_path,
    )
    _write_sidecar(meta)
    _log.info("saved %s (%.1fs)", video_path.name, meta.duration_s)
    return meta
