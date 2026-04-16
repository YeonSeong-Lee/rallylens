"""Pipeline orchestration for RallyLens."""

from rallylens.pipeline.court import run_court_detection, run_court_detection_interactive
from rallylens.pipeline.orchestrator import run_full_pipeline
from rallylens.pipeline.report import ReportResult, run_report_pipeline
from rallylens.pipeline.shuttle import run_shuttle_pipeline

__all__ = [
    "ReportResult",
    "run_court_detection",
    "run_court_detection_interactive",
    "run_full_pipeline",
    "run_report_pipeline",
    "run_shuttle_pipeline",
]
