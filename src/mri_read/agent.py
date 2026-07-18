"""
Step 5 — the agent loop.

Instead of the fixed manifest -> qc -> analyze pipeline (src/cmd/manifest.py,
src/cmd/qc.py, src/cmd/analyze.py), an orchestrator LLM decides which series
are worth inspecting, QC'ing, and analyzing by calling tools, then writes the
report itself. Everything still runs against a local Ollama server — the
orchestrator model just needs tool-calling support (e.g. qwen2.5, llama3.1),
which is a different (smaller, text-only) model than the vision model used for
the actual slice analysis.

Flow the orchestrator is nudged toward: get_manifest -> optionally run_qc on
series it's unsure about -> analyze_series on the ones worth a detailed read
-> write_report.

CLI entry point: src/cmd/agent.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from mri_read.analyze import select_named_series, write_report
from mri_read.engine import get_engine
from mri_read.manifest import build_manifest
from mri_read.mri import list_series
from mri_read.ollama_client import ensure_model, post
from mri_read.paths import OUT
from mri_read.qc import run_qc

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# A small text/tool-calling model — deliberately NOT the vision model used for
# analyze_series, which is chosen separately (see get_engine).
DEFAULT_MODEL = os.environ.get("OLLAMA_AGENT_MODEL", "qwen2.5")

SYSTEM_PROMPT = """You are orchestrating a local, research-prototype MRI-reading \
pipeline for a brain MRI study. You do not see the images yourself — you decide \
which series are worth a detailed read and drive the pipeline via tools.

Typical flow:
1. get_manifest — see what sequences are in the study (T1, T2, FLAIR, DWI, 3D T1, \
   reformats, localizers...) and their use_for_analysis flag.
2. Optionally run_qc on any series you want more confidence in before including it.
3. analyze_series with the series names worth a detailed read — usually one per \
   sequence type, skipping reformats/localizers/unknowns.
4. write_report to persist the result.

This is a research/engineering prototype, NOT clinical care. When you are done, \
reply with a short plain-text summary of what you found (no further tool calls)."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_series",
            "description": "List all series folder names found in the MRI study.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_manifest",
            "description": ("Classify every series (sequence type, plane, confidence, "
                            "use_for_analysis) and return the study manifest. "
                            "Call this first to see what's in the study."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_qc",
            "description": ("Run deterministic quality-control checks on one series "
                            "(missing slices, uneven spacing, contrast, SNR, empty slices)."),
            "parameters": {
                "type": "object",
                "properties": {
                    "series": {"type": "string",
                              "description": "Series folder name, e.g. 'Seri6'"},
                },
                "required": ["series"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_series",
            "description": ("Send the named series' slices to the local vision engine "
                            "for a structured radiology-style read. Call once you've "
                            "picked which series are worth analyzing."),
            "parameters": {
                "type": "object",
                "properties": {
                    "series": {"type": "array", "items": {"type": "string"},
                              "description": "Series folder names, e.g. ['Seri6', 'Seri1']"},
                    "slices_per_series": {"type": "integer",
                                         "description": "Representative slices per series (default 4)"},
                },
                "required": ["series"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": ("Persist the most recent analyze_series result to "
                            "output/report.json and output/report.md. Call this last."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


@dataclass
class AgentContext:
    """State the tool calls read/write across one agent run."""
    engine_name: str = "ollama"
    engine_kwargs: dict | None = None
    host: str = DEFAULT_HOST     # --host, threaded into the vision engine too
    manifest: dict | None = None
    last_result: object = None   # AnalysisResult, once analyze_series has run
    engine: object = None        # AnalysisEngine, built lazily and reused


def _tool_list_series(_args: dict, _ctx: AgentContext) -> dict:
    return {"series": list_series()}


def _tool_get_manifest(_args: dict, ctx: AgentContext) -> dict:
    ctx.manifest = build_manifest()
    return ctx.manifest


def _tool_run_qc(args: dict, _ctx: AgentContext) -> dict:
    name = args["series"]
    return run_qc(name)


def _engine_kwargs(ctx: AgentContext) -> dict:
    """CLI-provided engine kwargs, plus --host for engines that take one.

    The Ollama engines are the ones that talk to a host; the Claude engine
    doesn't accept a host kwarg, so we only inject it for ollama/local.
    """
    kwargs = dict(ctx.engine_kwargs or {})
    if ctx.engine_name in ("ollama", "local") and "host" not in kwargs:
        kwargs["host"] = ctx.host
    return kwargs


def _get_engine(ctx: AgentContext):
    """Build the vision engine once per agent run and reuse it.

    Avoids re-running ensure_model()'s connectivity/pull check on every
    analyze_series call when the orchestrator invokes it more than once.
    """
    if ctx.engine is None:
        ctx.engine = get_engine(ctx.engine_name, **_engine_kwargs(ctx))
    return ctx.engine


def _tool_analyze_series(args: dict, ctx: AgentContext) -> dict:
    if ctx.manifest is None:
        ctx.manifest = build_manifest()
    names = args.get("series") or []
    k = int(args.get("slices_per_series", 4))
    series_images = select_named_series(ctx.manifest, names, k)
    if not series_images:
        return {"error": f"none of {names} matched a series in the manifest"}

    engine = _get_engine(ctx)
    result = engine.analyze(ctx.manifest.get("study", {}), series_images)
    ctx.last_result = result
    return {
        "engine": result.engine,
        "sequences_reviewed": result.sequences_reviewed,
        "observations": result.observations,
        "impression": result.impression,
        "flags": result.flags,
    }


def _tool_write_report(_args: dict, ctx: AgentContext) -> dict:
    if ctx.last_result is None:
        return {"error": "no analysis result yet; call analyze_series first"}
    write_report(ctx.last_result, ctx.manifest.get("study", {}))
    return {"report_md": str(OUT / "report.md"), "report_json": str(OUT / "report.json")}


_TOOL_IMPLS = {
    "list_series": _tool_list_series,
    "get_manifest": _tool_get_manifest,
    "run_qc": _tool_run_qc,
    "analyze_series": _tool_analyze_series,
    "write_report": _tool_write_report,
}


def _dispatch(name: str, args: dict, ctx: AgentContext) -> dict:
    impl = _TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return impl(args, ctx)
    except Exception as e:                           # noqa: BLE001 - report, don't crash the loop
        return {"error": str(e)}


def run_agent(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
             engine_name: str = "ollama", engine_kwargs: dict | None = None,
             max_steps: int = 12, timeout: int = 900) -> tuple[str, AgentContext]:
    """Run the tool-calling agent loop to completion (or until max_steps).

    Returns (final_text_summary, context) — context.last_result holds the
    AnalysisResult if analyze_series was called.
    """
    ensure_model(host, model)                        # auto-pull, like the vision engine
    ctx = AgentContext(engine_name=engine_name, engine_kwargs=engine_kwargs, host=host)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Read this MRI study and produce a report."},
    ]

    for _ in range(max_steps):
        resp = post(host, "/api/chat", {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "options": {"temperature": 0.2},
        }, timeout)
        msg = resp.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content", ""), ctx

        messages.append(msg)
        for call in tool_calls:
            fn = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            print(f"  -> {fn}({args})")
            result = _dispatch(fn, args, ctx)
            messages.append({"role": "tool", "content": json.dumps(result, default=str)})

    return "(stopped after max_steps without a final answer)", ctx
