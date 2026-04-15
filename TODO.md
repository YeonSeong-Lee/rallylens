# RallyLens 구현 TODO

> 플랜 원본: `~/.claude/plans/resilient-zooming-anchor.md`
> 전체 일정: 3-4주, 주차별 산출물 중심

---

## Phase 0 — 초기 셋업 (Day 1, ~1시간)

- [ ] `uv init --package rallylens` 실행
- [ ] 의존성 설치: `uv add ultralytics opencv-python yt-dlp scenedetect anthropic click pandas pyarrow python-dotenv`
- [ ] 개발 의존성: `uv add --dev pytest ruff`
- [ ] 디렉토리 구조 생성 (`src/rallylens/{ingest,preprocess,vision,analysis,llm,viz}`, `data`, `models`, `outputs/demo`, `configs`, `notebooks`, `scripts`, `tests`)
- [ ] `.gitignore` 작성 (`data/`, `models/`, `.env`, `__pycache__/`, `.venv/`, `*.mp4` 예외 포함)
- [ ] `.env.example` 작성 (`ANTHROPIC_API_KEY=`)
- [ ] `pyproject.toml` 메타데이터 정리 (이름, 설명, 라이선스 = MIT)
- [ ] `git init` + 초기 커밋
- [ ] GitHub 원격 레포 생성 + push
- [ ] README 플레이스홀더 작성 (한 줄 피치만)

---

## Week 1 — Ingest + Baseline 탐지

**목표:** 1개 샘플 클립에서 end-to-end 동작 + 선수 오버레이 데모 GIF

### Ingest 모듈
- [ ] `src/rallylens/ingest/downloader.py` — yt-dlp 래퍼 (가장 먼저 구현, ML 없는 최소 단위)
- [ ] 720p/30fps 자동 정규화 로직 (`ffmpeg` via opencv 또는 yt-dlp format)
- [ ] 메타데이터 스크레이퍼 (제목, 업로드 날짜, 길이, URL)
- [ ] `src/rallylens/ingest/__init__.py`
- [ ] 다운로드 캐시 로직 (이미 있으면 재사용)
- [ ] `tests/test_downloader.py` — mocking 기반 단위 테스트

### Preprocess 모듈
- [ ] `src/rallylens/preprocess/rally_segmenter.py` — PySceneDetect `ContentDetector` 래퍼
- [ ] 임계값을 3개 경기로 튜닝 (노트북: `notebooks/01_explore_yolo_baseline.ipynb`)
- [ ] 모션 에너지 fallback (프레임 간 diff로 저동작 구간 제외)
- [ ] 랠리 클립 저장 (`rallies/rally_{idx:04d}.mp4`)
- [ ] 클립 길이 필터 (너무 짧은 것 drop, 예: 3초 미만)

### Vision 기초
- [ ] `src/rallylens/vision/detect_track.py` — `YOLO("yolo11n-pose.pt")` 로딩
- [ ] 선수 추론 + 박스 + 키포인트 추출
- [ ] `src/rallylens/viz/overlay.py` — OpenCV 박스/키포인트 렌더러
- [ ] 오버레이 클립 저장
- [ ] `scripts/make_gif.sh` — mp4 → GIF 변환 (`ffmpeg`)

### Week 1 산출물 체크
- [ ] **데모 GIF #1** — 선수 박스 + 포즈 키포인트 오버레이
- [ ] `rallylens ingest <url>` CLI 동작
- [ ] 1개 샘플 클립 end-to-end 통과

---

## Week 2 — 셔틀콕 파인튜닝 + 추적

**목표:** fine-tuned 셔틀콕 디텍터 + Kalman 추적 + 궤적 오버레이 GIF

### 데이터 라벨링
- [ ] 라벨링용 프레임 샘플러 구현 (`src/rallylens/preprocess/frame_sampler.py`)
- [ ] 랠리에서 ~400 프레임 자동 추출
- [ ] Label Studio 로컬 설치 (`uv tool install label-studio` 또는 Docker)
- [ ] 셔틀콕 bbox 라벨링 작업 (~2.5시간, 단일 클래스)
- [ ] Label Studio → YOLO format export
- [ ] train/val 8:2 split

### 파인튜닝 (Colab T4)
- [ ] `configs/yolo_shuttle.yaml` 작성 (Mosaic **비활성화**, `mosaic: 0.0`)
- [ ] `notebooks/02_shuttlecock_finetune.ipynb` 작성
- [ ] W&B 프로젝트 연동 (`wandb.init(project="rallylens-shuttle")`)
- [ ] `random_seed=42` 고정
- [ ] `yolo11n.pt` 50 epoch 학습, `save_period=1` (epoch 체크포인트)
- [ ] pre-trained baseline mAP 측정 (학습 전)
- [ ] fine-tuned mAP 측정 (학습 후)
- [ ] **현실적 목표: mAP@0.5 ≥ 0.30** (baseline 대비 개선폭 강조)
- [ ] W&B 학습 곡선 스크린샷 저장

### 추적 모듈
- [ ] `src/rallylens/vision/shuttlecock_detector.py` — fine-tuned weight 래퍼
- [ ] Kalman 필터 구현 (상태: [x, y, vx, vy], 측정: [x, y])
- [ ] nearest-neighbor 연관 (max-distance 임계값)
- [ ] missed 프레임 보간 로직
- [ ] `src/rallylens/vision/detect_track.py`에 **ByteTrack(선수 전용)** 추가: `model.track(tracker="bytetrack.yaml")`
- [ ] 싱글스 2명 선수 ID 유지 검증

### 코트 호모그래피
- [ ] `src/rallylens/vision/court_homography.py` — 4점 수동 클릭 UI (OpenCV `setMouseCallback`)
- [ ] `cv2.findHomography()`로 H 행렬 계산
- [ ] 경기 시작 시 1회 캘리브레이션 후 JSON에 저장
- [ ] 이미지 좌표 → 코트 좌표 변환 유틸 함수

### Week 2 산출물 체크
- [ ] **데모 GIF #2** — 셔틀콕 궤적 오버레이 (Kalman 보간 포함)
- [ ] 셔틀콕 탐지 평가 표 (baseline / fine-tuned)
- [ ] 선수 추적 ID 유지 확인

---

## Week 3 — 분석 + LLM 리포트

**목표:** 이벤트 탐지 + 히트맵 + Claude 리포트 생성기 + 전체 CLI

### 이벤트 탐지
- [ ] `src/rallylens/analysis/events.py` 구현
- [ ] **속도 벡터 방향 반전** 검출 (연속 프레임 내적 부호 변화)
- [ ] **Kalman 잔차 스파이크** 검출 (평균 + N×σ 임계값)
- [ ] 두 신호 AND/OR 조합 → hit 이벤트
- [ ] `events.jsonl` 스키마 정의 (frame, time, type, position, velocity, player_side)
- [ ] 랠리 단위 집계 (샷 수, 지속 시간, 승자 사이드 추정)

### 분석 시각화
- [ ] `src/rallylens/analysis/heatmap.py` — matplotlib 시각화
- [ ] 선수 포지션 히트맵 (코트 좌표계, 2D hist)
- [ ] 셔틀 궤적 밀도 히트맵
- [ ] 랠리 길이 히스토그램
- [ ] `outputs/heatmaps.png` 저장

### LLM 리포트 생성기
- [ ] `src/rallylens/llm/report_generator.py` 구현
- [ ] anthropic SDK 연동, 모델: `claude-sonnet-4-5`
- [ ] system 프롬프트 설계 (이벤트 스키마 + 리포트 템플릿, **1024 토큰 이상**)
- [ ] `cache_control: {"type": "ephemeral"}` 적용 (prompt caching)
- [ ] events.jsonl + rally stats → Markdown 리포트 생성
- [ ] 리포트 구조: 경기 요약 / 세트별 분석 / 선수별 통계 / 주목할 랠리
- [ ] 에러 핸들링 (API 실패, 토큰 초과)

### 라벨 QA 보조 에이전트
- [ ] `src/rallylens/llm/label_qa.py` — YOLO 예측 샘플을 Claude에 리뷰 요청
- [ ] 의심 라벨 플래그 출력 (JSONL)
- [ ] 공고 (1) 데이터 정리 + (4) 자동화 로직 동시 충족 컴포넌트

### CLI 완성
- [ ] `src/rallylens/cli.py` — click 기반 진입점
- [ ] `rallylens ingest <url>` / `rallylens run <url>` / `rallylens report <events.jsonl>` 명령
- [ ] `pyproject.toml`의 `[project.scripts]`에 등록
- [ ] `uv run rallylens run <sample_url>` end-to-end 검증

### Week 3 산출물 체크
- [ ] **데모 GIF #3** — 히트맵/리포트 스크린샷 또는 풀 파이프라인 요약
- [ ] `outputs/sample_match_report.md` 저장
- [ ] `rallylens run <url>` 단일 명령으로 전체 파이프라인 실행 성공

---

## Week 4 — 폴리싱 + README + 재현성

**목표:** GitHub 레포 제출 가능한 최종 상태

### 스크립트 & 재현성
- [ ] `scripts/run_full_pipeline.sh` — 샘플 URL 내장, 5분 내 완료 목표
- [ ] `scripts/download_match.sh` — 여러 경기 배치 다운로드
- [ ] `uv.lock` 커밋 (재현성)
- [ ] `.env.example` 최종 검수

### 테스트 & CI
- [ ] `tests/test_ingest.py` — yt-dlp 래퍼 단위 테스트 (mocking)
- [ ] `tests/test_events.py` — 이벤트 검출 로직 테스트 (합성 데이터)
- [ ] `tests/test_kalman.py` — Kalman 필터 sanity check
- [ ] `.github/workflows/test.yml` — pytest CI (Python 3.11, uv 사용)
- [ ] `ruff check` 통과 (포맷/린트)

### README 최종판
- [ ] **히어로 GIF** — 스크롤 없이 보이는 최상단 (추적 오버레이)
- [ ] 한 줄 피치 (한국어 + 영어)
- [ ] **공고 매핑 표** (✓ + 파일 링크) ← 가장 중요한 섹션
- [ ] 데모 GIF 3개 임베드
- [ ] 퀵스타트 (`uv sync` → `cp .env.example .env` → `rallylens run <url>`)
- [ ] 아키텍처 다이어그램 (ASCII 또는 이미지)
- [ ] 파인튜닝 결과 표 (baseline vs fine-tuned) + W&B 스크린샷
- [ ] LLM 리포트 샘플 임베드
- [ ] 데이터 & 라이선스 섹션:
  - [ ] 비디오는 gitignore, URL만 포함, 연구/교육 목적 명시
  - [ ] 셔틀콕 어노테이션 MIT
  - [ ] **Ultralytics YOLO11 AGPL-3.0 주의**
- [ ] Roadmap / Limitations (정직하게):
  - [ ] 단일 프레임 YOLO의 셔틀콕 한계 + TrackNetV3 인용
  - [ ] 싱글스 전용, 복식 미지원
  - [ ] 고정 카메라 가정, 리플레이 미처리
  - [ ] 수동 코트 캘리브레이션
- [ ] 참고 문헌 (ShuttleSet KDD 2023, TrackNetV3)

### 데모 자산
- [ ] `outputs/demo/` 에 최종 GIF 3개 커밋
- [ ] `outputs/demo/sample_report.md` 커밋
- [ ] 30초 스크린캐스트 녹화 + 호스팅 링크 (또는 GIF로 변환)

### 최종 검증 체크리스트
- [ ] `git clone` + `uv sync` + `rallylens run <sample>` 클린 환경에서 5분 내 완료
- [ ] `pytest` 전체 통과
- [ ] README 4개 공고 업무 매핑 ✓ 체크 — 각 파일 링크 클릭 테스트
- [ ] `outputs/demo/` GIF 3개 재생 가능
- [ ] match_report.md 할루시네이션 수동 검토
- [ ] fine-tuned mAP@0.5 ≥ 0.30 달성

---

## 스트레치 (시간 여유 시, 우선순위 순)

- [ ] **샷 타입 분류기** — smash/clear/drop 3클래스, ~200개 hit-frame 윈도우 라벨링, 셔틀 속도 + 선수 포즈 feature로 소형 MLP 학습 (공고의 "**분류**" 항목을 추가로 충족)
- [ ] **Gradio 데모** — 클립 업로드 → 오버레이 + 리포트 반환 (단일 파일 ~80줄)
- [ ] **코트 키포인트 자동 탐지** — 수동 4점 클릭 대체

---

## 리스크 알림 (작업 중 수시 확인)

- 셔틀콕 mAP가 0.30에 못 미쳐도 **Kalman 보간으로 궤적은 복구 가능** — README에 한계 솔직히 기록
- 스코프 크립 주의: 복식/자동 코트/실시간은 **절대 코어 일정에 포함 금지**
- Colab 세션 드롭 → `save_period=1`로 epoch 체크포인트 필수
- LLM 비용 경기당 $0.05-0.10, 예산 $5 이하 유지 (Sonnet + prompt caching)
