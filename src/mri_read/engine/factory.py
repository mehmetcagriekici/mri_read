"""Maps an engine name string to a concrete AnalysisEngine instance."""

from __future__ import annotations

from mri_read.engine.base import AnalysisEngine


def get_engine(name: str, **kwargs) -> AnalysisEngine:
    """Factory so analyze.py can pick an engine by name from the CLI.

    Default is 'ollama' — fully local, nothing leaves the machine. The Claude
    engine is a NON-LOCAL option kept only to demonstrate the swappable
    interface; do not use it on sensitive data.
    """
    # Imports are LAZY (inside each branch) so choosing the local engine never
    # imports the Claude SDK, and vice versa — you only need the deps you use.
    if name in ("ollama", "local"):
        from mri_read.ollama_vision import OllamaVisionEngine
        return OllamaVisionEngine(**kwargs)
    if name in ("claude", "claude-vision"):
        from mri_read.claude_vision import ClaudeVisionEngine  # 3rd-party / non-local
        return ClaudeVisionEngine(**kwargs)
    # To add a specialized model later, add a branch here pointing at a new
    # subpackage that implements AnalysisEngine — nothing else in the project
    # changes:
    #   if name == "medmodel": from mri_read.med_model import MedModelEngine; return ...
    raise ValueError(f"Unknown engine: {name!r}")
