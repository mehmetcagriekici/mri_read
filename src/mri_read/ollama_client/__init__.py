"""Minimal shared HTTP client for a local Ollama server (stdlib urllib only).

Both OllamaVisionEngine (image analysis) and the agent's text-synthesis pass
talk to the same local Ollama server, so the connect/pull/POST plumbing lives
here once instead of being copied into each:
  - http.py       : post(), generic JSON-over-HTTP transport.
  - resolve.py    : resolve_model(), matching a requested model name to the
                    exact tag Ollama has pulled (Ollama-specific business
                    logic, not generic transport).
  - models.py     : model_present() / ensure_model() (pull-if-missing),
                    built on resolve_model().
  - json_reply.py : parse_json_reply(), tolerant JSON extraction from a
                    chat model's free-form reply.
"""

from mri_read.ollama_client.http import post
from mri_read.ollama_client.json_reply import parse_json_reply
from mri_read.ollama_client.models import ensure_model, model_present

__all__ = ["post", "parse_json_reply", "model_present", "ensure_model"]
