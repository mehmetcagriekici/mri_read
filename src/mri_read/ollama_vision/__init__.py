"""
Ollama vision engine — fully LOCAL AnalysisEngine implementation.

No data leaves the machine: slices are sent to a local Ollama server running a
local vision model (e.g. llava:13b, qwen2.5vl). This is the default engine
for this project because the imaging data is sensitive.

Config (env or CLI):
  OLLAMA_HOST   default http://localhost:11434
  OLLAMA_MODEL  default llava:13b

Only the standard library is used (urllib) so the container stays small.

Layout:
  prompts.py     : SYSTEM, the vision-model prompt.
  sanitize.py    : filter_hallucinated_observations, catching well-formed-JSON
                   hallucinations (prompt-echo/placeholder observations) that
                   parse_json_reply's syntax check can't catch.
  engine_impl.py : OllamaVisionEngine, the AnalysisEngine implementation.

NOT a diagnostic tool. Research/engineering prototype only.
"""

from mri_read.ollama_vision.engine_impl import OllamaVisionEngine

__all__ = ["OllamaVisionEngine"]
