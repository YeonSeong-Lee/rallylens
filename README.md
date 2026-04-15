# RallyLens

> **YouTube 배드민턴 경기 URL → 자동 랠리 분할 · 추적 · Claude 경기 리포트.**
> *Automated badminton match analysis pipeline: rally segmentation, player & shuttlecock tracking, and Claude-powered match reports — all from a YouTube URL.*

Single-command end-to-end CLI for badminton match video analysis, built with YOLO11, Kalman tracking, and Claude.

> **Status**: Active development (Phase 3 / 4 — Event detection + heatmaps + Claude match report + label QA agent). See [`TODO.md`](TODO.md) for the full roadmap.

## Quickstart (placeholder)

```bash
git clone https://github.com/YeonSeong-Lee/rallylens
cd rallylens
brew install ffmpeg    # PySceneDetect splitter + GIF export prereq
uv sync
cp .env.example .env   # ANTHROPIC_API_KEY only needed for Week 3+
uv run rallylens run <youtube-url>
```

## License

MIT. YOLO11 weights are AGPL-3.0; this project is for portfolio / educational use only.
