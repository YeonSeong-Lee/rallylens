from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from rallylens.analysis.events import HitEvent, RallyStats
from rallylens.common import VideoMeta
from rallylens.llm.report_generator import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    generate_match_report,
)


def _fake_response(text: str = "# RallyLens Match Report\n\nmocked") -> SimpleNamespace:
    text_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(
        input_tokens=123,
        output_tokens=456,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(content=[text_block], usage=usage, stop_reason="end_turn")


def test_generate_match_report_uses_sonnet_and_caches(tmp_path: Path):
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response()

    meta = VideoMeta(
        video_id="abcdef12345",
        title="Test Match",
        upload_date="20240101",
        duration_s=300.0,
        url="https://www.youtube.com/watch?v=abcdef12345",
        source_path=tmp_path / "video.mp4",
    )
    stats = [
        RallyStats(
            video_id="abcdef12345",
            rally_index=1,
            duration_s=10.0,
            total_frames=300,
            shot_count=5,
            first_shot_frame=10,
            last_shot_frame=280,
            avg_inter_shot_gap_s=2.0,
            top_side_shots=3,
            bottom_side_shots=2,
            events=[
                HitEvent(
                    frame_idx=10,
                    time_s=0.33,
                    kind="hit",
                    position_xy=(100.0, 200.0),
                    velocity_xy=(5.0, 0.0),
                    signals=("velocity_reversal",),
                    player_side="top",
                )
            ],
        )
    ]

    out_path = tmp_path / "report.md"
    result = generate_match_report(meta, stats, out_path, client=fake_client)

    fake_client.messages.create.assert_called_once()
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"] == SYSTEM_PROMPT
    user_content = kwargs["messages"][0]["content"]
    assert "<match_data>" in user_content
    assert "abcdef12345" in user_content
    assert result == out_path
    assert out_path.exists()
    assert "RallyLens" in out_path.read_text(encoding="utf-8")


def test_system_prompt_exceeds_sonnet_cache_minimum():
    # Sonnet 4.6 min cache prefix is 2048 tokens (~ chars/4). A 2048-token prompt
    # is roughly 7000-8000 characters. Verify the prompt is at least that long.
    assert len(SYSTEM_PROMPT) > 6500, (
        f"system prompt is only {len(SYSTEM_PROMPT)} chars — likely under sonnet-4.6 "
        "cache minimum of 2048 tokens. Bulk up the schema/rules/example to ensure "
        "cache_control actually caches."
    )
