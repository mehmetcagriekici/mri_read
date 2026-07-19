"""The one-shot text-reasoning call that writes the final concise impression."""

from __future__ import annotations

import json
import logging
import time

from mri_read.agent.prompts import SYNTH_SYSTEM
from mri_read.engine import AnalysisResult
from mri_read.ollama_client import parse_json_reply, post

logger = logging.getLogger(__name__)

# See ollama_vision.engine_impl.MAX_REPLY_TOKENS for the full rationale --
# same fix, same real incident. A synthesized impression is "2-4 sentences";
# 512 tokens is far above any legitimate reply but bounds a stuck/looping
# generation's worst case.
MAX_REPLY_TOKENS = 512


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
    logger.info("synthesis: calling %s...", model)
    t0 = time.monotonic()
    try:
        resp = post(host, "/api/chat", {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": MAX_REPLY_TOKENS},
        }, timeout)
    except Exception:
        # Not swallowed -- re-raised as-is; the caller (agent.pipeline)
        # wraps this in a PipelineError. Logged here only so the failure's
        # duration is on record before that happens.
        logger.warning("synthesis: %s FAILED after %.1fs", model, time.monotonic() - t0)
        raise
    logger.info("synthesis: %s done in %.1fs", model, time.monotonic() - t0)
    text = resp.get("message", {}).get("content", "")
    try:
        synth = parse_json_reply(text)
    except (ValueError, IndexError):                    # malformed/non-JSON model reply
        # Don't surface the raw reply as the "impression" -- a model that
        # fails to follow the JSON contract has been observed dumping
        # unrelated hallucinated content here (e.g. a fake patient-record
        # schema), which would otherwise land verbatim in the final report
        # looking like real analysis. "unknown" makes the failure visible
        # instead of silently passing garbage through.
        return {"impression": "unknown", "flags": ["unparsed"]}
    synth["flags"] = synth.get("flags") or []           # tolerate an explicit null
    return synth
