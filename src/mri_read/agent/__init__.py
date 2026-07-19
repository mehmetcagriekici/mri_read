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
     duplicated. The vision model (OLLAMA_MODEL, default qwen2.5vl:7b) reads
     the images and returns structured per-sequence observations.
  4. A separate TEXT-reasoning model (OLLAMA_AGENT_MODEL, default a local
     medical-domain fine-tune) reads the manifest + QC + step-3 findings —
     never the images themselves — and writes the final concise impression.
     This is one-shot: no tool-calling support is required for this model.
  5. guard.py's two deterministic passes -- no model call, so this step
     can't itself hallucinate. Between steps 3 and 4, apply_correlation_guard()
     suppresses any vision observation asserting a diagnostic-sounding claim
     (tumor, malignancy, ...) that no OTHER sequence corroborates, replacing
     the claim text outright rather than just flagging it -- a caveat next to
     "tumor" is too easy to miss. After step 4, guard_final_impression() runs
     the same check against the synthesized prose, and cross-checks it
     doesn't discuss a sequence that was never actually analyzed. Confidence
     is never trusted from a model self-report; it's computed by this guard
     from what actually survived both passes.

Layout:
  context.py   : AgentContext/PipelineError, state handed back to the CLI.
  prompts.py   : SYNTH_SYSTEM, the text-reasoning model's system prompt.
  synthesis.py : _synthesize(), the one-shot text-reasoning call.
  guard.py     : apply_correlation_guard()/guard_final_impression(), the
                 deterministic hallucination guard (see step 5 above).
  pipeline.py  : run_agent(), wiring the whole flow together.

CLI entry point: src/cmd/agent.py
"""

from mri_read.agent.context import AgentContext, PipelineError
from mri_read.agent.pipeline import DEFAULT_HOST, DEFAULT_MODEL, run_agent

__all__ = ["AgentContext", "PipelineError", "run_agent",
          "DEFAULT_HOST", "DEFAULT_MODEL"]
