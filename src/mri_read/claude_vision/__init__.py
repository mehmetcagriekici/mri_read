"""
Claude vision engine — the first AnalysisEngine implementation.

Sends the selected slices (labeled by sequence) to Claude with a structured,
research-oriented prompt and parses the JSON it returns.

Requires:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-...        (or put it in a .env file)

Layout:
  dotenv.py      : load_dotenv(), minimal ANTHROPIC_API_KEY .env loader.
  prompts.py     : SYSTEM, the vision-model prompt.
  engine_impl.py : ClaudeVisionEngine, the AnalysisEngine implementation.

NOT a diagnostic tool. Output is a structured description for engineering /
research purposes only.
"""

from mri_read.claude_vision.engine_impl import ClaudeVisionEngine

__all__ = ["ClaudeVisionEngine"]
