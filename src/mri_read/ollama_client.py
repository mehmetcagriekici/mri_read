"""
Minimal shared HTTP client for a local Ollama server (stdlib urllib only).

Both OllamaVisionEngine (image analysis) and the Step 5 agent loop (tool-calling
orchestration) talk to the same local Ollama server, so the connect/pull/POST
plumbing lives here once instead of being copied into each.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def post(host: str, path: str, payload: dict, timeout: int) -> dict:
    """POST JSON to the Ollama server and return the parsed JSON reply."""
    req = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_json_reply(text: str) -> dict:
    """Pull a JSON object out of a chat model's reply, tolerating extra text.

    Local models don't reliably return pure JSON — they add prose or wrap it in
    ```json fences. Strip a leading fence, then slice from the first "{" to the
    last "}" and parse that. Shared by every step that asks an LLM for
    structured JSON: the Ollama vision engine, the agent's synthesis pass, and
    (despite the module name) the non-local Claude engine too — the parsing
    problem isn't Ollama-specific, only the HTTP transport below is.
    """
    t = text.strip()
    if t.startswith("```"):                          # drop a ```json ... ``` fence
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")            # outermost braces
    return json.loads(t[start:end + 1])


def _resolve_model(host: str, model: str) -> str | None:
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


def model_present(host: str, model: str) -> bool:
    """Is `model` (or some tag of it) already downloaded?"""
    return _resolve_model(host, model) is not None


def ensure_model(host: str, model: str) -> str:
    """Pull `model` into the local Ollama store if it's not there yet.

    Returns the exact tag to use for subsequent API calls: if `model` (or a
    same-base-name tag of it) is already present, that exact tag is returned
    unchanged; if a pull is triggered, `model` is returned as-is since Ollama
    pulls an untagged name to ":latest", which then matches exactly.

    This is why the Docker image stays small: weights are pulled at runtime
    into a persistent volume, not baked into the image.
    """
    resolved = _resolve_model(host, model)
    if resolved is not None:
        return resolved
    logger.info("Pulling local model '%s' (one-time)...", model)
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
                # A live-updating single-line progress meter, not a discrete
                # log event — print() (with \r) is the right tool here, not
                # logging, which would emit one line per update instead of
                # overwriting in place.
                print(f"  {status}", end="\r")
    print()
    logger.info("done.")
    return model
