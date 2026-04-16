"""Vertex AI Gemini wrapper for rally analysis report generation.

The Vertex SDK (`google-genai`) is an optional dependency (`uv sync --extra
report`). To keep `rallylens detect` and other non-report commands free of
the SDK, this package's `__init__.py` only re-exports the pure-pydantic
schema types. The `generate_report` / `render_report_markdown` functions
live in `rallylens.llm.report` and must be imported directly — loading that
submodule triggers `google.genai`, so callers should do it lazily (see
`rallylens.pipeline.report._load_llm`).
"""

from rallylens.llm.report_schema import PlayerInsight, ReportOutput

__all__ = ["PlayerInsight", "ReportOutput"]
