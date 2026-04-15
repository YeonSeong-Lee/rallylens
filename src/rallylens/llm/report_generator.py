"""Claude Sonnet 4.6 match report generator with prompt caching.

Flow:
    events.jsonl + rally stats  -->  JSON payload (per-match, uncached)
                                            |
             (large stable system prompt — cached)
                                            |
                                            v
                                    Markdown match report

The stable system prompt (schema + writing rules + example) is placed first so
top-level `cache_control={"type": "ephemeral"}` caches it across matches. Each
per-match user message only pays uncached cost for the tiny event payload —
~90% savings on the repeated system prefix. Cost target: $0.05-0.10 per match.

Model: `claude-sonnet-4-6` — the plan's explicit choice (cost-sensitive, strong
long-context behavior, same price as 4.5). Sonnet 4.6 requires a >=2048 token
cacheable prefix, which our schema+rules+example comfortably exceeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic

from rallylens.analysis.events import RallyStats
from rallylens.common import ensure_dir, get_logger
from rallylens.domain.video import VideoMeta

_log = get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8000


SYSTEM_PROMPT = """You are an expert badminton match analyst writing structured scouting reports for coaches and players from automatically extracted match data. Your output is a Markdown document a coach can read in under 5 minutes.

## Input format

You will receive a single JSON payload inside <match_data> tags with this structure:

```
{
  "video_meta": {
    "video_id": "<youtube id>",
    "title": "<video title>",
    "url": "<youtube url>",
    "duration_s": <float seconds>,
    "upload_date": "<YYYYMMDD or null>"
  },
  "rallies": [
    {
      "rally_index": <int 1-based>,
      "duration_s": <float>,
      "total_frames": <int>,
      "shot_count": <int>,
      "first_shot_frame": <int or null>,
      "last_shot_frame": <int or null>,
      "avg_inter_shot_gap_s": <float or null>,
      "top_side_shots": <int>,
      "bottom_side_shots": <int>,
      "events": [
        {
          "frame_idx": <int>,
          "time_s": <float>,
          "kind": "hit",
          "position_xy": [<float x pixel>, <float y pixel>],
          "velocity_xy": [<float vx>, <float vy>],
          "signals": ["velocity_reversal" | "residual_spike", ...],
          "player_side": "top" | "bottom" | null
        },
        ...
      ]
    },
    ...
  ]
}
```

## Critical data caveats (NEVER ignore these)

1. **Positions are PIXEL coordinates**, not meters. Do NOT convert to court meters, to kilometers per hour, or to any real-world unit. Court homography is not applied in this payload.
2. **Hit events are heuristic.** A "hit" is detected when the shuttle's velocity vector reverses direction AND/OR the Kalman-filter residual spikes. Missed hits (under-count) and spurious hits (over-count) are both possible, especially for fast exchanges and net shots. Rally shot counts under 3 may reflect detector misses more than actual play. Never call a 1-shot rally an "ace" — the detector likely missed the defender's return.
3. **`player_side` is not player identity.** It is the half of the frame (top vs bottom) that the shuttle was in at the moment the hit was detected. Two rallies where player_side="top" may be entirely different players; you CANNOT track a specific player's behavior across rallies.
4. **No player skill, no tactical intent, no score.** You do NOT have access to player names, handedness, match score, set number, tournament, or winner. Never invent any of these.
5. **No pose data in this payload.** You CANNOT say "backhand clear", "forehand smash", or "net drop" — you only know hit timing and shuttle position, not racket face or swing type.
6. **No court meters in this payload.** You CANNOT say "near the baseline" with specific distance — only "top half" or "bottom half" of the video frame.

If the data is thin (fewer than 3 rallies, or shot_count < 2 in the majority of rallies), say so explicitly in the Limitations section and do not extrapolate trends.

## Output format

Output ONLY valid GitHub-flavored Markdown. No ```markdown fence around the whole document, no preamble like "Here is the report". The first line must be the H1 title.

Use this exact section structure and keep these exact section headers (English headers, Korean body):

```
# RallyLens Match Report — <video title, truncated to 70 chars>

_Source: [<video_id>](<url>) · Duration: <MM:SS> · Rallies analyzed: <N>_

## 1. 경기 개요 / Match Summary
<one or two paragraphs in Korean. Total rallies, avg rally duration, total detected hits, longest rally by duration, longest rally by shot count. Use bullet points for the stats if clearer.>

## 2. 랠리별 분석 / Per-rally Breakdown
| Rally | Duration (s) | Shots | Top / Bottom |
|------:|-------------:|------:|--------------|
|   1   |     12.4     |   9   |    5 / 4     |
|  ...  |     ...      |  ...  |    ... / ... |

Include every rally as a row. Round duration to 1 decimal place.

## 3. 플레이 패턴 / Play Patterns
<2-4 bullets. Observations ONLY from the data: shot-count histogram (short/medium/long buckets by your own choice but justified), top vs bottom split, longest gap between shots, shortest gap. Call out asymmetries but never assign blame to a player.>

## 4. 주목할 랠리 / Notable Rallies
<up to 3 short items, each with rally index + one-sentence observation. Pick from: highest shot count, longest duration, most lopsided top/bottom split, fastest avg shot cadence. For each, mention the specific number that made it stand out.>

## 5. 한계 / Limitations
<bullets. Always include at least the 6 caveats above that applied to this analysis. Be honest when data was thin.>
```

## Style rules

- Write the body in Korean. Section headers stay bilingual exactly as shown above.
- Use digits for numbers. Round to 1 decimal unless the source is already an integer.
- Never hallucinate player names, scores, handedness, shot types, or tournament names.
- Never use phrases like "Player A dominated" — you do not know which player is which.
- Do not recommend drills or coaching actions. This is a descriptive scouting report, not a prescriptive one.
- Keep the entire document under 600 words of body text. Tables and headers don't count toward that budget.
- End with the Limitations section. Do not append a conclusion, call to action, or signature.

## Example shape

Here is the kind of output shape I expect (filler content, not real analysis):

```
# RallyLens Match Report — BWF 2024 All England Men's Singles F...

_Source: [dQw4w9WgXcQ](https://youtu.be/dQw4w9WgXcQ) · Duration: 48:12 · Rallies analyzed: 34_

## 1. 경기 개요 / Match Summary
총 34개의 랠리가 검출되었고 평균 랠리 길이는 8.7초였습니다. 최장 랠리는 #18 (21.4초, 17샷), 최다 샷 랠리는 #7 (18샷, 14.2초)입니다.

## 2. 랠리별 분석 / Per-rally Breakdown
| Rally | Duration (s) | Shots | Top / Bottom |
|------:|-------------:|------:|--------------|
|   1   |     5.2      |   4   |    2 / 2     |
|   2   |     9.1      |   7   |    4 / 3     |
...

## 3. 플레이 패턴 / Play Patterns
- 샷 분포: 짧은 랠리(4샷 이하) 12개, 중간(5-9) 15개, 긴 랠리(10+) 7개 — 중간 길이가 가장 흔함.
- 상단/하단 샷 합계는 74 / 69로 거의 대칭. 특정 사이드의 일방적 지배는 관찰되지 않음.
- 가장 빠른 평균 샷 간격은 랠리 #22 (0.6초), 가장 느린 것은 #4 (1.4초).

## 4. 주목할 랠리 / Notable Rallies
- **#18** — 21.4초 지속, 최장 랠리. 17샷으로 양측이 끈질기게 공방.
- **#7** — 18샷, 이벤트 밀도 가장 높음. 랠리 후반부에 샷 간격이 짧아짐.
- **#12** — 상단 8샷 / 하단 1샷. 한쪽 사이드에 이벤트가 집중된 이례적 랠리.

## 5. 한계 / Limitations
- 선수 식별 없음: "top"/"bottom"은 프레임 내 위치이며 특정 선수를 지칭하지 않습니다.
- 힛 검출은 heuristic이며 빠른 랠리는 누락 가능성이 있습니다.
- 픽셀 좌표만 사용 — 코트 meters 변환 및 speed(km/h)는 이 리포트에 포함되지 않습니다.
- 샷 타입 (클리어/스매시/드롭)은 별도 분류기 없이는 식별 불가능합니다.
- 경기 스코어, 선수명, 세트 정보는 입력되지 않았습니다.
- 총 3개 미만의 랠리가 검출되었을 경우 트렌드 분석은 신뢰할 수 없습니다.
```

Now process the actual input payload below and produce the report."""


def _build_user_payload(
    video_meta: VideoMeta, rally_stats: list[RallyStats]
) -> dict[str, Any]:
    return {
        "video_meta": {
            "video_id": video_meta.video_id,
            "title": video_meta.title,
            "url": video_meta.url,
            "duration_s": video_meta.duration_s,
            "upload_date": video_meta.upload_date,
        },
        "rallies": [r.model_dump(mode="json") for r in rally_stats],
    }


def generate_match_report(
    video_meta: VideoMeta,
    rally_stats: list[RallyStats],
    out_path: Path,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Path:
    """Generate a Markdown match report from events + stats. Writes to out_path."""
    ensure_dir(out_path.parent)
    client = client or anthropic.Anthropic()

    payload = _build_user_payload(video_meta, rally_stats)
    user_text = (
        "<match_data>\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n</match_data>\n\nGenerate the match report now."
    )

    _log.info(
        "generating report with %s (rallies=%d, total events=%d)",
        model,
        len(rally_stats),
        sum(r.shot_count for r in rally_stats),
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            cache_control={"type": "ephemeral"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )
    except anthropic.APIStatusError as e:
        _log.error("anthropic API error %s: %s", e.status_code, e.message)
        raise
    except anthropic.APIConnectionError:
        _log.error("anthropic connection error")
        raise

    usage = response.usage
    _log.info(
        "usage: input=%d cache_write=%d cache_read=%d output=%d",
        usage.input_tokens,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        usage.output_tokens,
    )
    if response.stop_reason == "max_tokens":
        _log.warning(
            "response hit max_tokens=%d — output may be truncated", max_tokens
        )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(
            f"anthropic returned no text blocks (stop_reason={response.stop_reason})"
        )
    report_md = "\n".join(text_blocks).strip() + "\n"
    out_path.write_text(report_md, encoding="utf-8")
    _log.info("wrote report -> %s (%d chars)", out_path, len(report_md))
    return out_path
