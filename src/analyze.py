"""
Step 4 — Orchestrator.

Reads output/manifest.json, selects representative slices from the primary
series, hands them to an engine, and writes a report. The engine is chosen by
name (--engine) so nothing here is tied to Claude specifically.

Usage:
  python src/analyze.py                         # local ollama, one series per sequence
  python src/analyze.py --slices 5              # more slices per series
  python src/analyze.py --model qwen2.5vl       # pick the local vision model
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dwi import diffusion_views
from engine import SeriesImages, get_engine
from mri import (apply_window, foreground_fraction, load_series,
                 volume_window_bounds)

OUT = Path(__file__).resolve().parent.parent / "output"
MANIFEST = OUT / "manifest.json"


def content_indices(volume: np.ndarray, k: int, min_fg: float = 0.05) -> list[int]:
    """Pick k slices with the most tissue content, spread across the volume.

    Beats naive even-spacing: near-empty end slices are dropped, and we sample
    within the slices that actually contain brain so a lesion is less likely to
    fall between samples.
    """
    fg = np.array([foreground_fraction(volume[i]) for i in range(volume.shape[0])])
    keep = np.where(fg >= min_fg)[0]
    if keep.size == 0:
        keep = np.arange(volume.shape[0])
    lo, hi = int(keep[0]), int(keep[-1])
    idx = np.linspace(lo, hi, min(k, hi - lo + 1)).astype(int)
    return sorted(set(idx.tolist()))


def volume_to_pngs(volume: np.ndarray, idx: list[int]) -> list[bytes]:
    """Window the whole volume once (consistent contrast), encode chosen slices."""
    lo, hi = volume_window_bounds(volume)
    pngs = []
    for i in idx:
        buf = io.BytesIO()
        Image.fromarray(apply_window(volume[i], lo, hi)).save(buf, format="PNG")
        pngs.append(buf.getvalue())
    return pngs


def _dwi_images(row: dict, slices: int) -> list[SeriesImages]:
    """DWI gets special treatment: high-b stack (+ ADC map when computable)."""
    v = diffusion_views(row["series"])
    out = []
    if v["high_b"] is not None:
        vol = v["high_b"]
        idx = content_indices(vol, slices)
        blabel = f"DWI (b={v['b_value']})" if v["b_value"] else "DWI"
        out.append(SeriesImages(row["series"], blabel, row.get("plane", "?"),
                                idx, volume_to_pngs(vol, idx)))
    if v["adc"] is not None:
        vol = v["adc"]
        idx = content_indices(vol, slices)
        out.append(SeriesImages(row["series"], "DWI ADC map",
                                row.get("plane", "?"), idx,
                                volume_to_pngs(vol, idx)))
    return out


def select_series(manifest: dict, slices: int, one_per_label: bool,
                  skip_qc_warn: bool):
    """Build SeriesImages for the primary series in the manifest."""
    seen = set()
    out = []
    for row in manifest["series"]:
        if not row.get("use_for_analysis"):
            continue
        label = row["label"]
        if one_per_label and label in seen:
            continue
        qc = row.get("qc", {})
        if skip_qc_warn and qc.get("status") == "warn":
            print(f"  skipping {row['series']} ({label}) — QC: "
                  f"{', '.join(qc.get('flags', []))}")
            continue
        seen.add(label)

        if label == "DWI":
            out.extend(_dwi_images(row, slices))
            continue

        s = load_series(row["series"])
        idx = content_indices(s.volume, slices)
        out.append(SeriesImages(
            series=row["series"],
            label=label,
            plane=row.get("plane", "?"),
            slice_indices=idx,
            slice_pngs=volume_to_pngs(s.volume, idx),
        ))
    return out


def write_report(result, study_meta: dict, series) -> None:
    (OUT / "report.json").write_text(json.dumps({
        "engine": result.engine,
        "study": study_meta,
        "sequences_reviewed": result.sequences_reviewed,
        "observations": result.observations,
        "impression": result.impression,
        "flags": result.flags,
        "disclaimer": result.disclaimer,
    }, indent=2))

    lines = [
        "# MRI Analysis Report (prototype)",
        "",
        f"> {result.disclaimer}",
        "",
        f"**Study:** {study_meta.get('body_part')} — {study_meta.get('model')} "
        f"@ {study_meta.get('field_T')}T",
        f"**Engine:** {result.engine}",
        f"**Sequences reviewed:** {', '.join(result.sequences_reviewed)}",
        "",
        "## Impression",
        "",
        result.impression or "_(none)_",
        "",
        "## Observations",
        "",
    ]
    if result.observations:
        for o in result.observations:
            lines.append(
                f"- **{o.get('sequence','?')}** — {o.get('finding','?')} "
                f"({o.get('location','?')}; confidence: {o.get('confidence','?')})"
            )
    else:
        lines.append("_(none reported)_")
    if result.flags:
        lines += ["", "## Flags", ""] + [f"- {f}" for f in result.flags]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


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
        raise SystemExit("Run  python src/manifest.py  first (need manifest.json).")
    manifest = json.loads(MANIFEST.read_text())
    study_meta = manifest.get("study", {})
    if not any("qc" in r for r in manifest.get("series", [])):
        print("Note: manifest has no QC yet — run  python src/qc.py  to add it.\n")

    series = select_series(manifest, args.slices,
                           one_per_label=not args.all_series,
                           skip_qc_warn=args.skip_qc_warn)
    n_imgs = sum(len(s.slice_pngs) for s in series)
    print(f"Selected {len(series)} series, {n_imgs} slice images. "
          f"Calling engine '{args.engine}'...")

    kwargs = {"model": args.model} if args.model else {}
    engine = get_engine(args.engine, **kwargs)
    result = engine.analyze(study_meta, series)

    write_report(result, study_meta, series)
    print(f"\nImpression: {result.impression}\n")
    print(f"Report written to {OUT/'report.md'} and {OUT/'report.json'}")


if __name__ == "__main__":
    main()
