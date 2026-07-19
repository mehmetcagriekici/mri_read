"""
Ollama vision engine — fully LOCAL AnalysisEngine implementation.

No data leaves the machine: slices are sent to a local Ollama server running a
local vision model (default qwen2.5vl:7b; llava:13b also supported). This is
the default engine for this project because the imaging data is sensitive.

Config (env or CLI):
  OLLAMA_HOST   default http://localhost:11434
  OLLAMA_MODEL  default qwen2.5vl:7b -- switched from llava:13b (still
                supported via --vision-model llava:13b) because llava's
                CLIP-ViT-L/14-336 vision tower has a fixed 336x336 input
                resolution, so every image this project sends (often
                1024x1024) was being silently downsampled with zero benefit
                -- pure wasted decode/transfer cost. qwen2.5vl's vision
                tower uses native dynamic resolution instead of a fixed
                crop, so it can actually use more of what's sent. See
                CLAUDE.md's Performance section for the measured details.

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
