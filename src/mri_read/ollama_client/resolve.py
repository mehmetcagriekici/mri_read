"""Resolving a requested model name to the exact tag Ollama has pulled."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def resolve_model(host: str, model: str) -> str | None:
    """Find the exact locally-present tag matching `model`, or None.

    GET /api/tags lists local models by their exact tag (e.g. "qwen2.5:0.5b").
    We first try an exact match, then fall back to matching on the base name
    (before any ":tag") so a short name like "qwen2.5" resolves to whatever
    tag is actually pulled — important because /api/chat and /api/generate,
    unlike this lookup, require an EXACT name match (a bare base name only
    matches a ":latest" tag, not an arbitrary one).

    Also doubles as the connectivity check — if Ollama is unreachable we raise
    a clear error here rather than failing cryptically later.
    """
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=15) as r:
            tags = json.loads(r.read().decode()).get("models", [])
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {host} ({e}). Is the ollama server running?"
        ) from e
    names = [m.get("name", "") for m in tags]
    if model in names:
        return model
    base = model.split(":")[0]
    for name in names:
        if name.split(":")[0] == base:
            return name
    return None
