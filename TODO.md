# RallyLens TODO

현재 구현 상태와 남은 작업을 추적합니다.

---

## 완료된 항목

### 인프라
- [x] 프로젝트 셋업 (`uv`, `pyproject.toml`, `.gitignore`)
- [x] `uv.lock` 커밋 (재현성)

### Ingest
- [x] `src/rallylens/ingest/downloader.py` — yt-dlp 래퍼, 캐시 로직 포함
- [x] `tests/test_downloader.py`

### 선수 추적
- [x] `src/rallylens/vision/detect_track.py` — YOLO11 + ByteTrack
- [x] `rallylens detect <video_path>` CLI
- [x] 결과: `data/detections/<video_stem>/<video_stem>_players.jsonl`

### 셔틀콕 탐지
- [x] `src/rallylens/vision/tracknet.py` — TrackNetV3 모델 구현
- [x] `src/rallylens/vision/shuttle_tracker.py` — 슬라이딩 윈도우 추론 래퍼
- [x] `src/rallylens/pipeline/shuttle.py` — 전체 영상 탐지 파이프라인
- [x] `rallylens detect-shuttle <video_path>` CLI
- [x] 결과: `data/tracks/<video_stem>/<video_stem>_shuttle.jsonl`

### 코트 캘리브레이션
- [x] `src/rallylens/vision/court_detector.py` — Hough 변환 기반 자동 코너 탐지
- [x] `src/rallylens/pipeline/court.py`
- [x] `rallylens calibrate <video_path>` CLI
- [x] 결과: `data/calibration/<video_stem>/corners.json`

### 전체 파이프라인
- [x] `rallylens run <url-or-path>` — 다운로드 + 선수 추적 one-shot

### 공통
- [x] `src/rallylens/pipeline/io.py` — 아티팩트 경로 단일 관리
- [x] `src/rallylens/serialization.py`
- [x] `tests/test_serialization.py`, `tests/test_common.py`, `tests/test_cli_integration.py`

---

## 남은 작업

### 필수

- [ ] TrackNetV3 pretrained weights 확보 → `models/shuttle_tracknet.pth` 저장
- [ ] `rallylens run <sample_url>` end-to-end 검증 (실 URL)
- [ ] GitHub 원격 레포 생성 + push

### CI / 품질

- [ ] `.github/workflows/test.yml` — pytest CI (Python 3.11, uv 사용)
- [ ] `ruff check` 통과 확인

### README / 데모

- [ ] 히어로 GIF — 선수 추적 오버레이 (`scripts/make_gif.sh` 활용)
- [ ] `outputs/demo/` 에 데모 GIF 커밋
- [ ] 아키텍처 다이어그램 (ASCII)

---

## 스트레치

- [ ] **CourtMapper 활용** — `src/rallylens/vision/court_mapping.py` 복구 후 셔틀 궤적을 코트 좌표계로 투영
- [ ] **히트맵** — 선수 포지션 / 셔틀 낙하 지점 시각화
- [ ] **이벤트 탐지** — 셔틀 속도 반전으로 hit 이벤트 검출
- [ ] **Gradio 데모** — 클립 업로드 → 오버레이 반환
