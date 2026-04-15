"""Claude-powered label QA: flags suspicious shuttlecock predictions for review.

Use this to triage YOLO shuttle predictions before manual relabeling. Claude
receives a cropped frame around each bbox and returns a confidence + reason in
a structured JSON schema (strict). Output is JSONL so the results can feed
straight back into Label Studio or a relabeling script.

This module addresses two of the job description's four areas simultaneously:
(1) data cleanup and (4) LLM automation — one reason the plan called it out as
a high-signal deliverable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic

from rallylens.common import ensure_dir, get_logger, read_frame_at
from rallylens.image_utils import crop_around_bbox, encode_jpeg_base64
from rallylens.vision.shuttlecock_detector import ShuttleDetection

_log = get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048
CROP_PADDING_PX = 80

LABEL_QA_SYSTEM_PROMPT = """You are reviewing automatically generated shuttlecock bounding box annotations in badminton match video frames. Your job is to flag suspicious annotations for human re-review.

A shuttlecock is a small feathered projectile — it appears as a tiny white or pale blob, typically 10-30 pixels wide, usually blurred due to high speed, and often located high in the frame on its way between players. It has a distinctive cone or teardrop silhouette when not heavily blurred.

For each cropped frame you see, evaluate whether the red bounding box center actually contains a shuttlecock. Common YOLO failure modes to flag:
- Racket head (strings make a small bright patch)
- Player hand or forearm
- Crowd member's bright clothing
- Scoreboard or court line artifacts
- Empty sky / ceiling (false positive)
- The bbox is on a real shuttlecock but the box is much larger or smaller than the actual object

You must reply ONLY with valid JSON matching the provided schema. No prose outside the JSON."""


def _label_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["looks_correct", "suspicious", "clearly_wrong"],
            },
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
            "likely_object": {"type": "string"},
        },
        "required": ["verdict", "confidence", "reason", "likely_object"],
        "additionalProperties": False,
    }


def review_detection(
    clip_path: Path,
    detection: ShuttleDetection,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Ask Claude to review a single detection. Returns a dict matching _label_schema."""
    client = client or anthropic.Anthropic()

    frame = read_frame_at(clip_path, detection.frame_idx)
    crop = crop_around_bbox(frame, detection.bbox_xyxy, padding_px=CROP_PADDING_PX)
    image_b64 = encode_jpeg_base64(crop)

    prompt = (
        f"Frame {detection.frame_idx} from {clip_path.name}. "
        f"YOLO reported a shuttlecock with confidence {detection.confidence:.2f}. "
        "The red rectangle shows the predicted bounding box. "
        "Is this actually a shuttlecock? Reply in the JSON schema provided."
    )

    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=LABEL_QA_SYSTEM_PROMPT,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _label_schema(),
            }
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError("label_qa: empty response from claude")
    try:
        parsed: dict[str, Any] = json.loads(text_blocks[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"label_qa: claude returned non-JSON: {text_blocks[0]!r}") from exc
    return parsed


def review_detections(
    clip_path: Path,
    detections: list[ShuttleDetection],
    out_path: Path,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_reviews: int = 40,
) -> Path:
    """Review up to `max_reviews` detections and write flagged ones to JSONL."""
    ensure_dir(out_path.parent)
    client = client or anthropic.Anthropic()

    ranked = sorted(detections, key=lambda d: d.confidence)[:max_reviews]
    _log.info("reviewing %d detections (lowest-confidence first)", len(ranked))

    with out_path.open("w", encoding="utf-8") as f:
        for det in ranked:
            try:
                result = review_detection(clip_path, det, client=client, model=model)
            except (anthropic.APIStatusError, RuntimeError) as exc:
                _log.warning("review failed for frame %d: %s", det.frame_idx, exc)
                continue
            row = {
                "clip": clip_path.name,
                "frame_idx": det.frame_idx,
                "bbox_xyxy": list(det.bbox_xyxy),
                "yolo_confidence": det.confidence,
                **result,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _log.info("wrote label QA results -> %s", out_path)
    return out_path
