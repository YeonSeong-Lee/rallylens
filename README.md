# RallyLens

> **배드민턴 경기 영상 → 선수 추적 자동화.**
> *Automated badminton player tracking pipeline from a video file or YouTube URL.*

Single-command CLI for badminton player tracking, built with YOLO11 and ByteTrack.

> **Status**: Active development. See [`TODO.md`](TODO.md) for the full roadmap.

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
uv run rallylens run <video-path-or-youtube-url>
```

로컬 파일 또는 YouTube URL을 입력하면 선수 추적 결과를 저장합니다.

옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--tracker [none\|bytetrack]` | `bytetrack` | 선수 추적기 선택 |

---

### 단계별 실행

#### 1. 선수 추적

```bash
uv run rallylens detect <video_path>
# 결과: data/detections/<video_stem>/<video_stem>_players.jsonl
```

옵션: `--tracker [none|bytetrack]` (기본값: `bytetrack`)

#### 2. 영상 다운로드 (YouTube)

```bash
uv run rallylens ingest <youtube-url>
# 결과: data/raw/<video_id>.mp4
```

#### 3. 코트 캘리브레이션 (호모그래피)

```bash
uv run rallylens calibrate <video_id>
# 화면에서 코트 4개 꼭짓점 클릭 → data/calibration/<video_id>/homography.json 저장
```

옵션: `--rally-index 1` `--frame-idx 0` (기준 프레임 지정)

#### 4. 라벨링용 프레임 추출 (파인튜닝)

```bash
uv run rallylens sample-frames <video_id> --total 400
# 결과: data/label_frames/<video_id>/
```

#### 5. 히트 이벤트 탐지 (셔틀콕 트랙 필요)

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

#### 8. 라벨 QA (Claude 비전 검수)

```bash
uv run rallylens label-qa <video_id> --rally-index 1 --max-reviews 40
# 결과: data/label_qa/<video_id>/rally_001_label_qa.jsonl
```

---

## 출력 디렉토리 구조

```
data/
├── raw/                  # 다운로드 원본 영상
├── detections/           # 선수 추적 결과 JSONL
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
