"""
CLI entry point for Step 5 — the local pipeline (deterministic analyze + LLM synthesis).

This is the primary entry point for the project: run this alone and it builds
the manifest, QCs and analyzes every primary series, then has a text-reasoning
model synthesize the final report. Everything runs against a local Ollama
server. (manifest.py/qc.py/analyze.py still exist individually for
development/debugging, but this is the one command to run for a report.)

Usage:
  python src/cmd/agent.py
  python src/cmd/agent.py --model meditron:7b --engine ollama --vision-model llava:13b
"""

from __future__ import annotations

import argparse
import json

from mri_read.agent import DEFAULT_HOST, DEFAULT_MODEL, PipelineError, run_agent
from mri_read.logging_setup import configure_logging
from mri_read.paths import OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="text-reasoning model that synthesizes the final report "
                         "from the manifest/QC/vision findings (default meditron:7b); "
                         "no tool-calling support needed")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--engine", default="ollama",
                    help="vision engine used to read series images: ollama (local, default) | claude (non-local)")
    ap.add_argument("--vision-model", default=None,
                    help="override the vision engine's model")
    ap.add_argument("--vision-timeout", type=int, default=None,
                    help="seconds to wait for EACH per-series vision call (one "
                         "call per sequence type, default 600; raise this on "
                         "slow CPU-only setups if you see a socket TimeoutError)")
    ap.add_argument("--slices", type=int, default=4,
                    help="slices per series handed to the vision engine (default 4)")
    ap.add_argument("--skip-qc-warn", action="store_true",
                    help="skip series QC flagged as warn from the vision analysis")
    ap.add_argument("--synth-timeout", type=int, default=None,
                    help="seconds to wait for the text-reasoning synthesis call "
                         "(default 900; raise this on slow CPU-only setups if you "
                         "see a socket TimeoutError)")
    args = ap.parse_args()

    # mri_read/ modules log their own progress (model pulls, per-series vision
    # calls with duration, QC skips, stage timing) via the stdlib logging
    # module rather than print(). Console output keeps the plain, unprefixed
    # look print() used to produce; output/agent.log additionally gets a
    # timestamped, appended copy -- a run that takes 30+ minutes on CPU-only
    # hardware is often watched from a different terminal or backgrounded,
    # so its timing needs to survive past whatever scrollback is still on
    # screen. Set up after arg parsing so `--help` stays clean.
    OUT.mkdir(exist_ok=True)
    log_path = OUT / "agent.log"
    configure_logging(log_path)
    print(f"Logging to console and {log_path}")

    engine_kwargs = {}
    if args.vision_model:
        engine_kwargs["model"] = args.vision_model
    if args.vision_timeout:
        engine_kwargs["timeout"] = args.vision_timeout
    run_kwargs = {"vision_slices": args.slices, "skip_qc_warn": args.skip_qc_warn}
    if args.synth_timeout:
        run_kwargs["timeout"] = args.synth_timeout

    try:
        summary, ctx = run_agent(args.model, args.host, args.engine, engine_kwargs,
                                 **run_kwargs)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    except PipelineError as e:
        # manifest + QC already ran even though the vision/text call failed —
        # persist it so it's still inspectable instead of losing that work.
        (OUT / "manifest.json").write_text(json.dumps(e.ctx.manifest, indent=2))
        raise SystemExit(f"{e}\n(partial manifest persisted to {OUT / 'manifest.json'} "
                         "for debugging)") from None

    # Persist the qc-augmented manifest (in-memory only inside run_agent) so
    # it's inspectable afterward, same as manifest.py/qc.py's standalone output.
    (OUT / "manifest.json").write_text(json.dumps(ctx.manifest, indent=2))

    print("\n=== Report summary ===")
    print(summary)
    print(f"\nReport written to {OUT / 'report.md'} and {OUT / 'report.json'}")
    print(f"Full timing log: {log_path}")


if __name__ == "__main__":
    main()
