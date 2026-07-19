"""run_agent() — the full local pipeline: deterministic analyze, then LLM synthesis."""

from __future__ import annotations

from mri_read.agent.context import AgentContext, PipelineError
from mri_read.agent.synthesis import _synthesize
from mri_read.analyze import select_series, write_report
from mri_read.config import DEFAULT as CFG
from mri_read.engine import AnalysisResult, get_engine
from mri_read.manifest import build_manifest
from mri_read.ollama_client import ensure_model
from mri_read.qc import run_qc

DEFAULT_HOST = CFG.ollama_host
# Text-reasoning model that synthesizes the final report from the deterministic
# manifest/QC data + the vision engine's findings. Deliberately a different
# model from OLLAMA_MODEL (the vision engine) — a medical-domain fine-tune
# rather than a vision model, and no tool-calling support is required since
# this is a single one-shot completion over data that's already assembled.
DEFAULT_MODEL = CFG.agent_model


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
