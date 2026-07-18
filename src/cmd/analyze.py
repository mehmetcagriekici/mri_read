"""
CLI entry point for Step 4 — the analysis orchestrator.

All the real logic lives in mri_read.analyze; this just wires up argparse and
the manifest.json / report.json / report.md file handling.

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
from mri_read.paths import OUT

MANIFEST = OUT / "manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="ollama",
                    help="ollama (local, default) | claude (non-local)")
    ap.add_argument("--model", default=None, help="engine model override")
    ap.add_argument("--slices", type=int, default=4,
                    help="slices per series (default 4)")
    ap.add_argument("--all-series", action="store_true",
                    help="include every primary series, not one per sequence type")
    ap.add_argument("--skip-qc-warn", action="store_true",
                    help="skip series that QC flagged (needs qc in manifest)")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit("Run  python src/cmd/manifest.py  first (need manifest.json).")
    manifest = json.loads(MANIFEST.read_text())
    study_meta = manifest.get("study", {})
    if not any("qc" in r for r in manifest.get("series", [])):
        print("Note: manifest has no QC yet — run  python src/cmd/qc.py  to add it.\n")

    series = select_series(manifest, args.slices,
                           one_per_label=not args.all_series,
                           skip_qc_warn=args.skip_qc_warn)
    n_imgs = sum(len(s.slice_pngs) for s in series)
    print(f"Selected {len(series)} series, {n_imgs} slice images. "
          f"Calling engine '{args.engine}'...")

    kwargs = {"model": args.model} if args.model else {}
    engine = get_engine(args.engine, **kwargs)
    result = engine.analyze(study_meta, series)

    write_report(result, study_meta)
    print(f"\nImpression: {result.impression}\n")
    print(f"Report written to {OUT/'report.md'} and {OUT/'report.json'}")


if __name__ == "__main__":
    main()
