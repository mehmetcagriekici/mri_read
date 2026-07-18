"""
Minimal shared HTTP client for a local Ollama server (stdlib urllib only).

Both OllamaVisionEngine (image analysis) and the Step 5 agent loop (tool-calling
orchestration) talk to the same local Ollama server, so the connect/pull/POST
plumbing lives here once instead of being copied into each.
"""

from __future__ import annotations

import json
import urllib.error
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


def model_present(host: str, model: str) -> bool:
    """Is `model` already downloaded? (GET /api/tags lists local models.)

    Also doubles as the connectivity check — if Ollama is unreachable we raise
    a clear error here rather than failing cryptically later.
    """
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=15) as r:
            tags = json.loads(r.read().decode()).get("models", [])
        # Compare on the base name (before any ":tag") so "llama3.2-vision"
        # matches "llama3.2-vision:latest".
        names = {m.get("name", "").split(":")[0] for m in tags}
        return model.split(":")[0] in names
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {host} ({e}). Is the ollama server running?"
        ) from e


def ensure_model(host: str, model: str) -> None:
    """Pull `model` into the local Ollama store if it's not there yet.

    This is why the Docker image stays small: weights are pulled at runtime
    into a persistent volume, not baked into the image.
    """
    if model_present(host, model):
        return
    print(f"Pulling local model '{model}' (one-time)...")
    # /api/pull streams progress lines; read to completion.
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/pull",
        data=json.dumps({"name": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=None) as r:
        for line in r:
            try:
                status = json.loads(line).get("status", "")
            except json.JSONDecodeError:
                continue
            if status:
                print(f"  {status}", end="\r")
    print("\n  done.")
