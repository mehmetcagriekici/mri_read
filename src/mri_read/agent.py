"""
Step 5 — the local pipeline: deterministic analyze, then an LLM synthesis pass.

Previously an orchestrator LLM decided (via tool calls) which series to
inspect, QC, and analyze — a small tool-calling model turned out to be
unreliable about that, silently skipping whole sequence types some runs.
Coverage now comes entirely from deterministic code, same as analyze.py; the
two LLM calls are narrowed to what LLMs are actually needed for here: reading
images, and turning structured findings into a concise write-up. Neither
model decides what data exists to look at.

Flow, every step always runs (no model can skip one):
  1. build_manifest()               -- classify every series (rule-based).
  2. run_qc() on every use_for_analysis series -- deterministic quality flags,
                                        folded into the manifest (mirrors qc.py).
     Reformats/localizers are skipped since they're never analyzed anyway.
  3. select_series() + one vision-engine .analyze() call over every primary
     series (one per sequence type, picking the BEST candidate per type via
     analyze._rank_key -- e.g. the DWI folder with more b-values, the
     thinnest 3D T1, the cleanest QC/SNR for everything else) -- the same
     building blocks analyze.py uses, so slice selection/windowing isn't
     duplicated. The vision model (OLLAMA_MODEL, default llava:13b) reads the
     images and returns structured per-sequence observations.
  4. A separate TEXT-reasoning model (OLLAMA_AGENT_MODEL, default a local
     medical-domain fine-tune) reads the manifest + QC + step-3 findings —
     never the images themselves — and writes the final concise impression.
     This is one-shot: no tool-calling support is required for this model.

CLI entry point: src/cmd/agent.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mri_read.analyze import select_series, write_report
from mri_read.config import DEFAULT as CFG
from mri_read.engine import AnalysisResult, get_engine
from mri_read.manifest import build_manifest
from mri_read.ollama_client import ensure_model, parse_json_reply, post
from mri_read.qc import run_qc

DEFAULT_HOST = CFG.ollama_host
# Text-reasoning model that synthesizes the final report from the deterministic
# manifest/QC data + the vision engine's findings. Deliberately a different
# model from OLLAMA_MODEL (the vision engine) — a medical-domain fine-tune
# rather than a vision model, and no tool-calling support is required since
# this is a single one-shot completion over data that's already assembled.
DEFAULT_MODEL = CFG.agent_model

SYNTH_SYSTEM = """You are a medical-domain assistant writing the FINAL summary of a \
local, research-prototype brain MRI read. You do not see images yourself. You are \
given:
- the study manifest: every series, its inferred sequence type, and deterministic \
QC flags (missing slices, uneven spacing, low contrast, low SNR, mostly-empty).
- structured findings a vision model already reported after reading the images per \
sequence (observations, a draft impression, flags/caveats).

Synthesize these into ONE concise, coherent impression: reconcile findings across \
sequences, lower your confidence in (or explicitly caveat) any finding whose series \
has a QC flag, and note any primary sequence type that has QC issues.

This is a research/engineering prototype, NOT clinical care — never give patient \
instructions or a diagnosis.

Return ONLY valid JSON in this exact shape:
{
  "impression": "2-4 sentence concise overall summary",
  "flags": ["short phrases for anything notable, low-confidence, or QC-limited"]
}"""


@dataclass
class AgentContext:
    """State returned to the CLI after one pipeline run."""
    manifest: dict | None = None
    last_result: AnalysisResult | None = None


class PipelineError(RuntimeError):
    """The vision engine or text-synthesis call failed.

    Carries the AgentContext built so far (manifest + QC already ran) so the
    caller can still persist output/manifest.json for debugging instead of
    losing that work to an uncaught traceback.
    """
    def __init__(self, message: str, ctx: AgentContext):
        super().__init__(message)
        self.ctx = ctx


def _synthesize(model: str, host: str, study_meta: dict, manifest: dict,
                vision_result: AnalysisResult, timeout: int) -> dict:
    """One-shot text-reasoning pass over the deterministic data + vision findings.

    No images, no tool calls, no decisions about what to look at — just the
    manifest/QC/vision-observations JSON in, a concise {"impression", "flags"}
    JSON out.
    """
    payload = {
        "study": study_meta,
        "series": [
            {"series": r["series"], "label": r["label"],
             "use_for_analysis": r["use_for_analysis"], "qc": r.get("qc")}
            for r in manifest["series"]
        ],
        "vision_findings": {
            "engine": vision_result.engine,
            "sequences_reviewed": vision_result.sequences_reviewed,
            "observations": vision_result.observations,
            "draft_impression": vision_result.impression,
            "flags": vision_result.flags,
        },
    }
    messages = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]
    resp = post(host, "/api/chat", {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }, timeout)
    text = resp.get("message", {}).get("content", "")
    try:
        synth = parse_json_reply(text)
    except (ValueError, IndexError):                    # malformed/non-JSON model reply
        return {"impression": text, "flags": ["unparsed"]}
    synth["flags"] = synth.get("flags") or []           # tolerate an explicit null
    return synth


def run_agent(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
             engine_name: str = "ollama", engine_kwargs: dict | None = None,
             vision_slices: int = 4, skip_qc_warn: bool = False,
             timeout: int = 900) -> tuple[str, AgentContext]:
    """Run the full local pipeline: deterministic analyze, then LLM synthesis.

    Coverage is guaranteed — manifest, QC, and vision analysis all run over
    the whole study, the same way analyze.py does it, regardless of what
    either model says. Returns (final_impression_text, context); context.manifest
    carries the qc-augmented manifest, context.last_result the final
    AnalysisResult (already written to output/report.md + report.json).
    """
    ctx = AgentContext()
    ctx.manifest = build_manifest()
    for row in ctx.manifest["series"]:
        if row.get("use_for_analysis"):               # skip reformats/localizers — never analyzed
            row["qc"] = run_qc(row["series"])

    study_meta = ctx.manifest.get("study", {})
    series_images = select_series(ctx.manifest, vision_slices,
                                  one_per_label=True, skip_qc_warn=skip_qc_warn)

    # engine_kwargs only ever carries CLI overrides (model/timeout); inject
    # `host` here too so --host reaches the vision engine, not just the text
    # model below. The Claude engine doesn't accept a host kwarg, so this is
    # ollama/local-only, same as the tool-calling agent this replaced did.
    vision_kwargs = dict(engine_kwargs or {})
    if engine_name in ("ollama", "local") and "host" not in vision_kwargs:
        vision_kwargs["host"] = host
    try:
        vision_engine = get_engine(engine_name, **vision_kwargs)
        vision_result = vision_engine.analyze(study_meta, series_images)
    except Exception as e:
        # Intentionally broad: this boundary sits behind the pluggable engine
        # abstraction (get_engine), so the failure could be a network error
        # from the ollama engine, an anthropic SDK error from the claude
        # engine, or something else from a future engine — the point of the
        # abstraction is that this call site doesn't know which. Nothing is
        # swallowed: it's converted to a PipelineError carrying the already-
        # built manifest/QC and re-raised (`from e` keeps the original
        # traceback) so the caller can still persist that partial work.
        raise PipelineError(f"vision engine failed: {e}", ctx) from e

    try:
        text_model = ensure_model(host, model)          # resolve to exact pulled tag
        synth = _synthesize(text_model, host, study_meta, ctx.manifest, vision_result, timeout)
    except Exception as e:
        # Same rationale as above: ensure_model/_synthesize talk HTTP to a
        # local Ollama server, whose failure modes (connection refused, model
        # not found, bad JSON reply) aren't worth enumerating here — re-raised
        # as PipelineError, not swallowed.
        raise PipelineError(f"text-synthesis model failed: {e}", ctx) from e

    ctx.last_result = AnalysisResult(
        engine=f"{vision_result.engine} + text:{text_model}",
        sequences_reviewed=vision_result.sequences_reviewed,
        observations=vision_result.observations,
        impression=synth.get("impression") or vision_result.impression,
        flags=list(dict.fromkeys(vision_result.flags + synth.get("flags", []))),
        disclaimer=vision_result.disclaimer,
        raw={"vision": vision_result.raw, "synthesis": synth},
    )
    write_report(ctx.last_result, study_meta)
    return ctx.last_result.impression, ctx
