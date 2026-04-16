"""Factory for a Vertex AI Gemini client.

Reads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` from environment
variables (populated by `load_env()` at CLI startup). Authentication is
handled by Application Default Credentials — run
`gcloud auth application-default login` once locally, or rely on the
metadata server when deployed to GCP.
"""

from __future__ import annotations

import os

from google import genai

from rallylens.common import get_logger

_log = get_logger(__name__)

_DEFAULT_LOCATION = "us-central1"


def create_vertex_client() -> genai.Client:
    """Construct a Vertex AI `genai.Client` from environment variables.

    Raises:
        RuntimeError: if `GOOGLE_CLOUD_PROJECT` is unset.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set. Add it to .env (and set "
            "GOOGLE_CLOUD_LOCATION), then ensure Application Default "
            "Credentials are configured: "
            "`gcloud auth application-default login`."
        )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", _DEFAULT_LOCATION)
    _log.info("creating Vertex AI client (project=%s location=%s)", project, location)
    return genai.Client(vertexai=True, project=project, location=location)
