"""
CLI entry point for Step 4 — the fixed (non-agent) analysis orchestrator.

Development/debugging tool: runs the deterministic manifest -> qc -> analyze
pipeline standalone (builds manifest.json/QC itself if missing) for testing
the vision engine or slice selection in isolation, without an LLM
orchestrator in the loop. The primary entry point for running the project is
`src/cmd/agent.py`.

Usage:
  python src/cmd/analyze.py                         # local ollama, one series per sequence
  python src/cmd/analyze.py --slices 5              # more slices per series
  python src/cmd/analyze.py --model qwen2.5vl       # pick the local vision model
"""

from __future__ import annotations

import argparse
import json

from mri_read.analyze import select_series, write_report
from mri_read.engine import get_engine
from mri_read.logging_setup import configure_logging
from mri_read.manifest import build_manifest
from mri_read.paths import OUT
from mri_read.qc import run_qc

MANIFEST = OUT / "manifest.json"


def _load_or_build_manifest() -> dict:
    """Load manifest.json, building it (and QC) first if it's missing/stale.

    Makes this script a standalone "final product" command: running it alone
    produces a report without first requiring manifest.py/qc.py runs.
    """
    OUT.mkdir(exist_ok=True)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    else:
        print("No manifest.json yet — building it now "
              "(equivalent to  python src/cmd/manifest.py)...")
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(manifest, indent=2))

    if not any("qc" in r for r in manifest.get("series", [])):
        print("Manifest has no QC yet — running it now "
              "(equivalent to  python src/cmd/qc.py)...\n")
        for row in manifest["series"]:
            row["qc"] = run_qc(row["series"])
        MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="ollama",
                    help="ollama (local, default) | claude (non-local)")
    ap.add_argument("--model", default=None, help="engine model override")
    ap.add_argument("--vision-timeout", type=int, default=None,
                    help="seconds to wait for EACH per-series vision call (default "
                         "600; raise this on slow CPU-only setups if you see a "
                         "socket TimeoutError)")
    ap.add_argument("--slices", type=int, default=4,
                    help="slices per series (default 4)")
    ap.add_argument("--all-series", action="store_true",
                    help="include every primary series, not one per sequence type")
    ap.add_argument("--skip-qc-warn", action="store_true",
                    help="skip series that QC flagged (needs qc in manifest)")
    args = ap.parse_args()

    # See src/cmd/agent.py's logging_setup use for the full rationale. A
    # separate log file (not agent.log) since this is a different tool --
    # keeps a fixed-pipeline debug run's timing from interleaving with an
    # agent run's. Set up after arg parsing so `--help` stays clean.
    OUT.mkdir(exist_ok=True)
    configure_logging(OUT / "analyze.log")

    try:
        manifest = _load_or_build_manifest()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    study_meta = manifest.get("study", {})

    series = select_series(manifest, args.slices,
                           one_per_label=not args.all_series,
                           skip_qc_warn=args.skip_qc_warn)
    n_imgs = sum(len(s.slice_pngs) for s in series)
    print(f"Selected {len(series)} series, {n_imgs} slice images. "
          f"Calling engine '{args.engine}'...")

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.vision_timeout:
        kwargs["timeout"] = args.vision_timeout
    engine = get_engine(args.engine, **kwargs)
    result = engine.analyze(study_meta, series)

    write_report(result, study_meta)
    print(f"\nImpression: {result.impression}\n")
    print(f"Report written to {OUT/'report.md'} and {OUT/'report.json'}")


if __name__ == "__main__":
    main()
