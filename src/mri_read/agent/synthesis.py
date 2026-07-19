"""The one-shot text-reasoning call that writes the final concise impression."""

from __future__ import annotations

import json

from mri_read.agent.prompts import SYNTH_SYSTEM
from mri_read.engine import AnalysisResult
from mri_read.ollama_client import parse_json_reply, post


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
