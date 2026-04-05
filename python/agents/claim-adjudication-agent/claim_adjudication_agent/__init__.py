"""Health claim advisor: facilitate health insurance claim processing."""

import os

import google.auth

from . import agent

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
os.environ.setdefault("GEMINI_FLASH", "gemini-2.5-flash")
os.environ.setdefault(
    "CLAIM_DOCUMENTS_BUCKET", "agentspace-demo-ds-bucket-proj-genai-1729"
)
os.environ.setdefault("CLAIM_DOCUMENTS_BUCKET_FOLDER", "health_claim_documents")
