"""OllamaVisionEngine — fully LOCAL AnalysisEngine implementation."""

from __future__ import annotations

import base64
import logging

from mri_read.config import DEFAULT as CFG
from mri_read.engine import AnalysisEngine, AnalysisResult, SeriesImages
from mri_read.ollama_client import ensure_model, parse_json_reply, post
from mri_read.ollama_vision.prompts import SYSTEM

logger = logging.getLogger(__name__)

DEFAULT_HOST = CFG.ollama_host
DEFAULT_MODEL = CFG.vision_model


class OllamaVisionEngine(AnalysisEngine):
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: int = 600, auto_pull: bool = True):
        # host normalized (no trailing slash). timeout is PER SERIES CALL (see
        # analyze()) — one call per sequence type, not one call for the whole
        # study, so this only needs to cover a handful of images on CPU, not
        # every selected series' slices at once.
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        if auto_pull:
            self.model = ensure_model(self.host, self.model)  # resolve to exact pulled tag

    def _analyze_one(self, study_meta: dict, s: SeriesImages) -> dict:
        """One /api/chat call scoped to a single series' slices.

        Splitting per series (rather than bundling every sequence type into one
        request, as this used to do) keeps each call's image count small. On a
        CPU-only Ollama host, one request carrying every selected series at
        once (5+ sequence types x several slices = 20+ images) routinely blew
        past even a generous timeout with zero progress feedback (stream=False)
        — indistinguishable from a hang. Per-series calls are each fast enough
        to finish reliably, and a slow/failed sequence no longer sinks the
        whole report.
        """
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                (f"Study: {study_meta.get('body_part')} MRI on "
                 f"{study_meta.get('model')} at {study_meta.get('field_T')}T.\n"
                 f"=== {s.label} ({s.plane}, {s.series}) — "
                 f"slice indices {s.slice_indices} ===\n"
                 "Now return ONLY the JSON described in the system prompt, "
                 "scoped to this one sequence."),
             # Ollama wants base64 strings (no data-URI prefix) in "images".
             "images": [base64.b64encode(p).decode() for p in s.slice_pngs]},
        ]
        # stream=False -> one complete reply; low temperature -> steadier JSON.
        resp = post(self.host, "/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }, self.timeout)
        text = resp.get("message", {}).get("content", "")
        try:
            return parse_json_reply(text)
        except (ValueError, IndexError):             # malformed/non-JSON model reply
            return {"impression": text, "observations": [], "flags": ["unparsed"]}

    # --- engine interface (implements AnalysisEngine.analyze) ---
    def analyze(self, study_meta: dict,
                series: list[SeriesImages]) -> AnalysisResult:
        sequences_reviewed: list[str] = []
        observations: list[dict] = []
        impressions: list[str] = []
        flags: list[str] = []
        raw_per_series: dict[str, dict] = {}
        errors: list[str] = []

        for s in series:
            logger.info("vision: reading %s (%s, %d slices)...",
                       s.label, s.series, len(s.slice_pngs))
            try:
                data = self._analyze_one(study_meta, s)
            except Exception as e:
                # Intentionally broad: this is the resilience boundary the
                # per-series split exists for — one sequence's HTTP/timeout/
                # model failure gets recorded as a flag, not allowed to sink
                # every other sequence's results. Nothing is swallowed
                # silently: the error text is kept in `flags` and shows up in
                # the final report.
                errors.append(f"{s.label}: {e}")
                flags.append(f"{s.label}: analysis failed ({e})")
                continue
            sequences_reviewed.extend(data.get("sequences_reviewed", [s.label]))
            observations.extend(data.get("observations", []))
            if data.get("impression"):
                impressions.append(f"{s.label}: {data['impression']}")
            flags.extend(data.get("flags") or [])
            raw_per_series[s.series] = data

        if not raw_per_series and errors:
            raise RuntimeError(f"all series failed vision analysis: {'; '.join(errors)}")

        return AnalysisResult(
            engine=f"{self.name}:{self.model}",
            sequences_reviewed=list(dict.fromkeys(sequences_reviewed)),
            observations=observations,
            impression=" ".join(impressions),
            flags=list(dict.fromkeys(flags)),
            disclaimer=("Local research/engineering prototype. NOT a medical "
                        "diagnosis. Not validated for clinical use."),
            raw=raw_per_series,
        )
