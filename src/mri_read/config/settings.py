"""Centralized runtime configuration.

Env-var defaults were previously read independently in each module that talks
to a model server (ollama_vision, agent both had their own
`os.environ.get("OLLAMA_HOST", ...)`). Collecting them here means there's one
place to look, and one place to add a new setting.

This does NOT replace CLI overrides: src/cmd/*.py flags (--host, --model, ...)
still win, same as before -- this only centralizes the fallback default that
kicks in when no flag is given. Each module keeps its own DEFAULT_HOST/
DEFAULT_MODEL constants (sourced from here) so existing imports of those names
keep working unchanged.

ANTHROPIC_API_KEY is deliberately NOT read here: claude_vision loads a
.env file at ClaudeVisionEngine construction time (a runtime side effect), so
that key has to be read live via os.environ at that point, not captured once
at import time -- capturing it here would read it before .env has a chance to
set it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    vision_model: str = os.environ.get("OLLAMA_MODEL", "llava:13b")
    agent_model: str = os.environ.get("OLLAMA_AGENT_MODEL", "meditron:7b")


DEFAULT = Config()
