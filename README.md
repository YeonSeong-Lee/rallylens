# RallyLens

> **배드민턴 경기 영상 → 선수 추적 자동화.**
> *Automated badminton player tracking pipeline from a video file or YouTube URL.*

Single-command CLI for badminton player tracking, built with YOLO11, ByteTrack, and TrackNetV3.

> **Status**: Active development. See [`TODO.md`](TODO.md) for the full roadmap.

---

## 설치

```bash
git clone https://github.com/YeonSeong-Lee/rallylens
cd rallylens
brew install ffmpeg        # macOS; Linux: sudo apt install ffmpeg
uv sync
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

#### 1. 영상 다운로드 (YouTube)

```bash
uv run rallylens ingest <youtube-url>
# 결과: data/raw/<video_id>.mp4
```

#### 2. 선수 추적

```bash
uv run rallylens detect <video_path>
# 결과: data/detections/<video_stem>/<video_stem>_players.jsonl
```

옵션: `--tracker [none|bytetrack]` (기본값: `bytetrack`)

#### 3. 셔틀콕 탐지 (TrackNetV3)

```bash
uv run rallylens detect-shuttle <video_path>
# 결과: data/tracks/<video_stem>/<video_stem>_shuttle.jsonl
```

옵션: `--weights <path>` (기본값: `models/shuttle_tracknet.pth`)

#### 4. 코트 캘리브레이션 (호모그래피)

```bash
uv run rallylens calibrate <video_path>
# 결과: data/calibration/<video_stem>/corners.json
```

옵션: `--samples 20` (코트 탐지에 사용할 프레임 수, 기본값: `20`)

#### 5. 시각화

```bash
uv run rallylens viz <video_path>
# 결과: data/viz/<video_stem>_overlay.mp4
#       data/viz/<video_stem>_heatmap.png
#       data/viz/<video_stem>_court.png
```

사전에 `detect`, `detect-shuttle`, `calibrate` 결과가 있어야 합니다.

옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--overlay / --no-overlay` | `True` | 선수·셔틀콕 오버레이 영상(MP4) 생성 |
| `--heatmap / --no-heatmap` | `True` | 선수·셔틀콕 위치 히트맵(PNG) 생성 |
| `--court / --no-court` | `True` | 코트 탑뷰 궤적 다이어그램(PNG) 생성 |
| `--trail-len <int>` | `30` | 셔틀콕 잔상 길이 (프레임 수) |

---

## 출력 디렉토리 구조

```
data/
├── raw/           # 다운로드 원본 영상
├── detections/    # 선수 추적 결과 JSONL
├── tracks/        # 셔틀콕 트랙 JSONL
├── calibration/   # 코트 코너 JSON
└── viz/           # 시각화 출력 (MP4, PNG)
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
