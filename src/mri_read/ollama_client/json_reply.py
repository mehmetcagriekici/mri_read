"""Pulling a JSON object out of a chat model's free-form reply."""

from __future__ import annotations

import json


def parse_json_reply(text: str) -> dict:
    """Pull a JSON object out of a chat model's reply, tolerating extra text.

    Local models don't reliably return pure JSON — they add prose or wrap it in
    ```json fences. Strip a leading fence, then slice from the first "{" to the
    last "}" and parse that. Shared by every step that asks an LLM for
    structured JSON: the Ollama vision engine, the agent's synthesis pass, and
    (despite the package name) the non-local Claude engine too — the parsing
    problem isn't Ollama-specific, only the HTTP transport is.
    """
    t = text.strip()
    if t.startswith("```"):                          # drop a ```json ... ``` fence
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")            # outermost braces
    return json.loads(t[start:end + 1])
