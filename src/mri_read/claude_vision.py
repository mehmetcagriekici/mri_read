"""
Claude vision engine — the first AnalysisEngine implementation.

Sends the selected slices (labeled by sequence) to Claude with a structured,
research-oriented prompt and parses the JSON it returns.

Requires:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-...        (or put it in a .env file)

NOT a diagnostic tool. Output is a structured description for engineering /
research purposes only.
"""

from __future__ import annotations

import base64
import json
import os

from mri_read.engine import AnalysisEngine, AnalysisResult, SeriesImages
from mri_read.paths import ROOT

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM = """You are assisting a software engineer who is prototyping an MRI-reading
pipeline. You are shown selected slices from a brain MRI, grouped and labeled by
sequence (T1, T2, FLAIR, DWI, 3D T1). Describe what is visible in a structured,
neutral, radiology-style way.

Rules:
- This is a research/engineering prototype, NOT clinical care. Do not tell a
  patient what to do. Frame everything as "visible on these images".
- Only comment on what the provided slices actually show. If a sequence is
  missing or slices are too few to judge, say so.
- Note asymmetries, signal abnormalities, mass effect, midline shift,
  restricted diffusion (DWI), or fluid-suppressed lesions (FLAIR) if visible.
- Be explicit about uncertainty.

Return ONLY valid JSON, no prose outside it, in this exact shape:
{
  "sequences_reviewed": ["T2 FLAIR", "DWI", ...],
  "observations": [
    {"sequence": "T2 FLAIR", "finding": "...", "location": "...", "confidence": "low|moderate|high"}
  ],
  "impression": "1-3 sentence overall summary of what the images show",
  "flags": ["short phrases for anything notable or that limits the read"]
}"""


def _load_dotenv() -> None:
    """Minimal .env loader so ANTHROPIC_API_KEY can live in a file."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _extract_json(text: str) -> dict:
    """Tolerate ```json fences / stray prose around the JSON object."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    return json.loads(t[start:end + 1])


class ClaudeVisionEngine(AnalysisEngine):
    name = "claude-vision"

    def __init__(self, model: str = DEFAULT_MODEL):
        # Fail fast with a clear message if the key is missing, before any call.
        _load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or add it to a .env file."
            )
        import anthropic  # imported lazily so the rest of the app needs no SDK
        self.client = anthropic.Anthropic()          # reads the key from env
        self.model = model

    def analyze(self, study_meta: dict,
                series: list[SeriesImages]) -> AnalysisResult:
        # Claude takes a single message whose `content` is an ORDERED list mixing
        # text and image blocks. We interleave: a text label, then that series'
        # images, so each picture sits right after the sequence name it belongs to.
        content: list[dict] = [{
            "type": "text",
            "text": (f"Study: {study_meta.get('body_part')} MRI on "
                     f"{study_meta.get('model')} at {study_meta.get('field_T')}T.\n"
                     f"Below are labeled slices from {len(series)} sequences."),
        }]
        for s in series:
            content.append({                         # label for this sequence
                "type": "text",
                "text": f"\n=== {s.label} ({s.plane}, {s.series}) — "
                        f"slice indices {s.slice_indices} ===",
            })
            for png in s.slice_pngs:                 # then its slice images
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(png).decode(),
                    },
                })

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")

        try:
            data = _extract_json(text)
        except Exception:                            # noqa: BLE001
            data = {"impression": text, "observations": [], "flags": ["unparsed"]}

        return AnalysisResult(
            engine=self.name,
            sequences_reviewed=data.get("sequences_reviewed",
                                        [s.label for s in series]),
            observations=data.get("observations", []),
            impression=data.get("impression", ""),
            flags=data.get("flags", []),
            disclaimer=("Research/engineering prototype. NOT a medical diagnosis. "
                        "Not validated for clinical use."),
            raw=data,
        )
