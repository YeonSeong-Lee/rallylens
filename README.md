# RallyLens

> **YouTube 배드민턴 경기 URL → 자동 랠리 분할 · 추적 · Claude 경기 리포트.**
> *Automated badminton match analysis pipeline: rally segmentation, player & shuttlecock tracking, and Claude-powered match reports — all from a YouTube URL.*

Single-command end-to-end CLI for badminton match video analysis, built with YOLO11, Kalman tracking, and Claude.

> **Status**: Active development (Phase 3 / 4 — Event detection + heatmaps + Claude match report + label QA agent). See [`TODO.md`](TODO.md) for the full roadmap.

---

## 설치

```bash
git clone https://github.com/YeonSeong-Lee/rallylens
cd rallylens
brew install ffmpeg        # macOS; Linux: sudo apt install ffmpeg
uv sync
cp .env.example .env       # ANTHROPIC_API_KEY 입력 (report 커맨드 사용 시 필요)
```

---

## 실행

### 전체 파이프라인 (one-shot)

```bash
uv run rallylens run <youtube-url>
```

다운로드 → 랠리 분할 → 선수/셔틀콕 추적 → 이벤트 탐지 → 히트맵 → Claude 리포트를 순서대로 실행합니다.

옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--tracker [none\|bytetrack]` | `bytetrack` | 선수 추적기 선택 |
| `--no-shuttle` | — | 셔틀콕 추적 생략 |
| `--no-report` | — | Claude 리포트 생략 (API 키 불필요) |

---

### 단계별 실행

각 단계를 개별로 실행할 수 있습니다. `<video_id>`는 `ingest` 완료 후 출력되는 ID입니다.

#### 1. 영상 다운로드

```bash
uv run rallylens ingest <youtube-url>
# 결과: data/raw/<video_id>.mp4
```

#### 2. 랠리 분할

```bash
uv run rallylens segment <video_id>
# 결과: data/rallies/<video_id>/rally_*.mp4 + rallies.json
```

씬 감지 임계값 조정: `--threshold 27.0` (기본값, 낮을수록 세밀하게 분할)

#### 3. 코트 캘리브레이션 (호모그래피)

```bash
uv run rallylens calibrate <video_id>
# 화면에서 코트 4개 꼭짓점 클릭 → data/calibration/<video_id>/homography.json 저장
```

옵션: `--rally-index 1` `--frame-idx 0` (기준 프레임 지정)

#### 4. 선수 + 셔틀콕 추적

```bash
uv run rallylens detect <video_id> --rally-index 1
# 결과: data/overlays/<video_id>/rally_001_overlay.mp4
```

#### 5. 히트 이벤트 탐지

```bash
uv run rallylens events <video_id>
# 결과: data/events/<video_id>/rally_*_events.jsonl
```

#### 6. 히트맵 렌더링

```bash
uv run rallylens heatmaps <video_id>
# 결과: data/heatmaps/<video_id>/heatmaps.png
```

#### 7. Claude 경기 리포트 생성

```bash
uv run rallylens report <video_id>
# 결과: data/reports/<video_id>/match_report.md
```

`ANTHROPIC_API_KEY` 필요.

#### 8. 라벨링용 프레임 추출 (Week 2 파인튜닝)

```bash
uv run rallylens sample-frames <video_id> --total 400
# 결과: data/label_frames/<video_id>/
```

#### 9. 라벨 QA (Claude 비전 검수)

```bash
uv run rallylens label-qa <video_id> --rally-index 1 --max-reviews 40
# 결과: data/label_qa/<video_id>/rally_001_label_qa.jsonl
```

---

## 출력 디렉토리 구조

```
data/
├── raw/                  # 다운로드 원본 영상
├── rallies/              # 분할된 랠리 클립 + rallies.json
├── overlays/             # 추적 오버레이 영상
├── tracks/               # 셔틀콕 추적 데이터
├── events/               # 히트 이벤트 JSONL + 랠리 stats
├── heatmaps/             # 히트맵 PNG
├── reports/              # Claude Markdown 리포트
├── calibration/          # 코트 호모그래피 JSON
├── label_frames/         # 라벨링용 샘플 프레임
└── label_qa/             # Claude 라벨 검수 결과 JSONL
```

---

## 개발

```bash
uv run pytest             # 전체 테스트
uv run ruff check src/    # 린트
uv run ruff format src/   # 포맷
```

---

## License

MIT. YOLO11 weights are AGPL-3.0; this project is for portfolio / educational use only.
