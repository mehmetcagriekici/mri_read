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
  engine_impl.py : OllamaVisionEngine, the AnalysisEngine implementation.

NOT a diagnostic tool. Research/engineering prototype only.
"""

from mri_read.ollama_vision.engine_impl import OllamaVisionEngine

__all__ = ["OllamaVisionEngine"]
