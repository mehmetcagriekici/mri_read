"""Generic HTTP transport (stdlib urllib only) to a local Ollama server."""

from __future__ import annotations

import json
import urllib.request


def post(host: str, path: str, payload: dict, timeout: int) -> dict:
    """POST JSON to the Ollama server and return the parsed JSON reply."""
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
