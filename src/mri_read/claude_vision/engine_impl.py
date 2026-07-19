"""ClaudeVisionEngine — the first AnalysisEngine implementation."""

from __future__ import annotations

import base64
import os

from mri_read.claude_vision.dotenv import load_dotenv
from mri_read.claude_vision.prompts import SYSTEM
from mri_read.engine import AnalysisEngine, AnalysisResult, SeriesImages
from mri_read.ollama_client import parse_json_reply

DEFAULT_MODEL = "claude-sonnet-5"


def _format_acq(acq: dict) -> str:
    """Render TE/TR/TI as a short "Acquisition: ..." line, or "" if none of
    them are present. See ollama_vision.engine_impl's identical helper.
    """
    parts = []
    for key in ("TE", "TR", "TI"):
        value = acq.get(key)
        if value is not None:
            parts.append(f"{key}={value}ms")
    return f" Acquisition: {', '.join(parts)}." if parts else ""


class ClaudeVisionEngine(AnalysisEngine):
    name = "claude-vision"

    def __init__(self, model: str = DEFAULT_MODEL):
        # Fail fast with a clear message if the key is missing, before any call.
        # Reads os.environ live (not via mri_read.config) because load_dotenv()
        # is a runtime side effect that must run first -- a module-level Config
        # snapshot would capture the env before .env ever gets a chance to set it.
        load_dotenv()
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
                        f"slice indices {s.slice_indices} ==={_format_acq(s.acq)}",
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
            data = parse_json_reply(text)
        except (ValueError, IndexError):             # malformed/non-JSON model reply
            data = {"impression": text, "observations": [], "flags": ["unparsed"]}

        return AnalysisResult(
            engine=self.name,
            sequences_reviewed=data.get("sequences_reviewed",
                                        [s.label for s in series]),
            observations=data.get("observations", []),
            impression=data.get("impression", ""),
            flags=data.get("flags") or [],
            disclaimer=("Research/engineering prototype. NOT a medical diagnosis. "
                        "Not validated for clinical use."),
            raw=data,
        )
