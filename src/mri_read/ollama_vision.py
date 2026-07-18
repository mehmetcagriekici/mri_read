"""
Ollama vision engine — fully LOCAL AnalysisEngine implementation.

No data leaves the machine: slices are sent to a local Ollama server running a
local vision model (e.g. llama3.2-vision, qwen2.5vl). This is the default engine
for this project because the imaging data is sensitive.

Config (env or CLI):
  OLLAMA_HOST   default http://localhost:11434
  OLLAMA_MODEL  default llama3.2-vision

Only the standard library is used (urllib) so the container stays small.

NOT a diagnostic tool. Research/engineering prototype only.
"""

from __future__ import annotations

import base64
import json
import os

from mri_read.engine import AnalysisEngine, AnalysisResult, SeriesImages
from mri_read.ollama_client import ensure_model, post

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")

SYSTEM = """You are assisting a software engineer prototyping a LOCAL MRI-reading
pipeline. You are shown selected slices from a brain MRI, grouped and labeled by
sequence (T1, T2, FLAIR, DWI, 3D T1). Describe what is visible in a structured,
neutral, radiology-style way.

Rules:
- Research/engineering prototype, NOT clinical care. Frame everything as
  "visible on these images". Do not give patient instructions.
- Only comment on what the provided slices actually show. If slices are too few
  to judge, say so.
- Note asymmetries, signal abnormalities, mass effect, midline shift, restricted
  diffusion (DWI), or fluid-suppressed lesions (FLAIR) if visible.
- Be explicit about uncertainty.

Return ONLY valid JSON in this exact shape:
{
  "sequences_reviewed": ["T2 FLAIR", "DWI"],
  "observations": [
    {"sequence": "T2 FLAIR", "finding": "...", "location": "...", "confidence": "low|moderate|high"}
  ],
  "impression": "1-3 sentence overall summary",
  "flags": ["short phrases for anything notable or limiting"]
}"""


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model's reply, tolerating extra text.

    Local models don't reliably return pure JSON — they add prose or wrap it in
    ```json fences. We strip a leading fence, then slice from the first "{" to
    the last "}" and parse that.
    """
    t = text.strip()
    if t.startswith("```"):                          # drop a ```json ... ``` fence
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")           # outermost braces
    return json.loads(t[start:end + 1])


class OllamaVisionEngine(AnalysisEngine):
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: int = 900, auto_pull: bool = True):
        # host normalized (no trailing slash); timeout is generous because local
        # vision inference on CPU can take minutes.
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        if auto_pull:
            self.model = ensure_model(self.host, self.model)  # resolve to exact pulled tag

    # --- engine interface (implements AnalysisEngine.analyze) ---
    def analyze(self, study_meta: dict,
                series: list[SeriesImages]) -> AnalysisResult:
        # Build the chat: system rules, a study intro, then ONE message per
        # series carrying that sequence's label + its slice images. Keeping each
        # sequence's images in its own labeled message is how the model knows
        # which pictures are FLAIR vs DWI (Ollama attaches images per-message).
        messages = [{"role": "system", "content": SYSTEM}]
        messages.append({
            "role": "user",
            "content": (f"Study: {study_meta.get('body_part')} MRI on "
                        f"{study_meta.get('model')} at "
                        f"{study_meta.get('field_T')}T. "
                        f"{len(series)} sequences follow."),
        })
        for s in series:
            messages.append({
                "role": "user",
                "content": f"=== {s.label} ({s.plane}, {s.series}) — "
                           f"slice indices {s.slice_indices} ===",
                # Ollama wants base64 strings (no data-URI prefix) in "images".
                "images": [base64.b64encode(p).decode() for p in s.slice_pngs],
            })
        messages.append({
            "role": "user",
            "content": "Now return ONLY the JSON described in the system prompt.",
        })

        # stream=False -> one complete reply; low temperature -> steadier JSON.
        resp = post(self.host, "/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }, self.timeout)
        text = resp.get("message", {}).get("content", "")

        # If the model didn't return parseable JSON, keep its raw text as the
        # impression and flag it, rather than crashing.
        try:
            data = _extract_json(text)
        except Exception:                            # noqa: BLE001
            data = {"impression": text, "observations": [], "flags": ["unparsed"]}

        return AnalysisResult(
            engine=f"{self.name}:{self.model}",
            sequences_reviewed=data.get("sequences_reviewed",
                                        [s.label for s in series]),
            observations=data.get("observations", []),
            impression=data.get("impression", ""),
            flags=data.get("flags", []),
            disclaimer=("Local research/engineering prototype. NOT a medical "
                        "diagnosis. Not validated for clinical use."),
            raw=data,
        )
