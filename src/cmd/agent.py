"""
CLI entry point for Step 5 — the LLM-orchestrated agent loop.

Instead of the fixed manifest -> qc -> analyze pipeline, an orchestrator model
decides which series to inspect, QC, and analyze via tool calls, then writes
the report. Everything runs against a local Ollama server.

Usage:
  python src/cmd/agent.py
  python src/cmd/agent.py --model qwen2.5 --engine ollama --vision-model llama3.2-vision
"""

from __future__ import annotations

import argparse

from mri_read.agent import DEFAULT_HOST, DEFAULT_MODEL, run_agent
from mri_read.paths import OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="orchestrator model; must support tool calling (default qwen2.5)")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--engine", default="ollama",
                    help="vision engine analyze_series calls: ollama (local, default) | claude (non-local)")
    ap.add_argument("--vision-model", default=None,
                    help="override the vision engine's model")
    ap.add_argument("--max-steps", type=int, default=12,
                    help="cap on tool-calling rounds before giving up (default 12)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    engine_kwargs = {"model": args.vision_model} if args.vision_model else {}

    summary, ctx = run_agent(args.model, args.host, args.engine, engine_kwargs,
                             max_steps=args.max_steps)

    print("\n=== Agent summary ===")
    print(summary)
    if ctx.last_result is not None:
        print(f"\nImpression: {ctx.last_result.impression}")
    print(f"\n(if write_report was called: {OUT / 'report.md'}, {OUT / 'report.json'})")


if __name__ == "__main__":
    main()
