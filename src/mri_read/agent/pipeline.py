"""run_agent() — the full local pipeline: deterministic analyze, then LLM synthesis."""

from __future__ import annotations

import logging
import time

from mri_read.agent.context import AgentContext, PipelineError
from mri_read.agent.guard import apply_correlation_guard, guard_final_impression
from mri_read.agent.synthesis import _synthesize
from mri_read.analyze import select_series, write_report
from mri_read.config import DEFAULT as CFG
from mri_read.engine import AnalysisResult, get_engine
from mri_read.manifest import build_manifest
from mri_read.ollama_client import ensure_model
from mri_read.qc import run_qc

logger = logging.getLogger(__name__)

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
    run_start = time.monotonic()
    logger.info("agent: starting run")

    ctx = AgentContext()
    t0 = time.monotonic()
    ctx.manifest = build_manifest()
    for row in ctx.manifest["series"]:
        if row.get("use_for_analysis"):               # skip reformats/localizers — never analyzed
            row["qc"] = run_qc(row["series"])
    logger.info("agent: manifest + QC done in %.1fs (%d series)",
               time.monotonic() - t0, len(ctx.manifest["series"]))

    study_meta = ctx.manifest.get("study", {})
    series_images = select_series(ctx.manifest, vision_slices,
                                  one_per_label=True, skip_qc_warn=skip_qc_warn)
    logger.info("agent: selected %d series-images for vision analysis", len(series_images))

    # engine_kwargs only ever carries CLI overrides (model/timeout); inject
    # `host` here too so --host reaches the vision engine, not just the text
    # model below. The Claude engine doesn't accept a host kwarg, so this is
    # ollama/local-only, same as the tool-calling agent this replaced did.
    vision_kwargs = dict(engine_kwargs or {})
    if engine_name in ("ollama", "local") and "host" not in vision_kwargs:
        vision_kwargs["host"] = host
    t0 = time.monotonic()
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
    logger.info("agent: vision phase (model load + all series) done in %.1fs",
               time.monotonic() - t0)

    # Deterministic guard, pass 1 (no model call): suppress any vision
    # observation asserting a diagnostic-sounding claim that no OTHER
    # sequence corroborates, before it ever reaches the text-synthesis
    # model. See agent.guard's module docstring for the full rationale.
    vision_result.observations, correlation_flags = apply_correlation_guard(
        vision_result.observations)
    vision_result.flags = list(dict.fromkeys(vision_result.flags + correlation_flags))
    if correlation_flags:
        logger.warning("agent: correlation guard suppressed %d uncorroborated claim(s)",
                       len(correlation_flags))

    t0 = time.monotonic()
    try:
        text_model = ensure_model(host, model)          # resolve to exact pulled tag
        synth = _synthesize(text_model, host, study_meta, ctx.manifest, vision_result, timeout)
    except Exception as e:
        # Same rationale as above: ensure_model/_synthesize talk HTTP to a
        # local Ollama server, whose failure modes (connection refused, model
        # not found, bad JSON reply) aren't worth enumerating here — re-raised
        # as PipelineError, not swallowed.
        raise PipelineError(f"text-synthesis model failed: {e}", ctx) from e
    logger.info("agent: synthesis phase (model load + call) done in %.1fs",
               time.monotonic() - t0)

    # Deterministic guard, pass 2: the same check against the SYNTHESIZED
    # prose (a different model than the one that produced the observations,
    # so it needs its own pass), plus a manifest-consistency check and the
    # overall confidence -- computed here, never trusted from either model's
    # self-report.
    raw_impression = synth.get("impression") or vision_result.impression
    final_impression, confidence, impression_flags = guard_final_impression(
        raw_impression, vision_result.observations, ctx.manifest)

    ctx.last_result = AnalysisResult(
        engine=f"{vision_result.engine} + text:{text_model}",
        sequences_reviewed=vision_result.sequences_reviewed,
        observations=vision_result.observations,
        impression=final_impression,
        confidence=confidence,
        flags=list(dict.fromkeys(
            vision_result.flags + synth.get("flags", []) + impression_flags)),
        disclaimer=vision_result.disclaimer,
        raw={"vision": vision_result.raw, "synthesis": synth},
    )
    write_report(ctx.last_result, study_meta)
    logger.info("agent: run complete in %.1fs total (confidence: %s)",
               time.monotonic() - run_start, ctx.last_result.confidence)
    return ctx.last_result.impression, ctx
