"""
Step 4 — Orchestrator.

Reads output/manifest.json, selects representative slices from the primary
series, hands them to an engine, and writes a report. The engine is chosen by
name (--engine) so nothing here is tied to Claude specifically.

CLI entry point: src/cmd/analyze.py
"""

from __future__ import annotations

import io
import json
import logging

import numpy as np
from PIL import Image

from mri_read.dwi import count_bvalue_buckets, diffusion_views
from mri_read.engine import SeriesImages
from mri_read.mri import (apply_window, foreground_fraction, load_series,
                          volume_window_bounds)
from mri_read.paths import OUT

logger = logging.getLogger(__name__)


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


def build_series_images(row: dict, slices: int) -> list[SeriesImages]:
    """Build SeriesImages for one manifest row, DWI handled specially."""
    if row["label"] == "DWI":
        return _dwi_images(row, slices)
    s = load_series(row["series"])
    idx = content_indices(s.volume, slices)
    return [SeriesImages(
        series=row["series"],
        label=row["label"],
        plane=row.get("plane", "?"),
        slice_indices=idx,
        slice_pngs=volume_to_pngs(s.volume, idx),
    )]


def _rank_key(row: dict) -> tuple:
    """Higher is better. QC status always leads, then a label-specific tie-break.

    - status first, always: a series whose pixel load already failed QC
      ("error") or that's flagged ("warn") never outranks a clean one, no
      matter how it scores on the label-specific criteria below.
    - DWI: more distinct b-value buckets (needed for an ADC map — reused from
      QC's own header pass when available, see count_bvalue_buckets_from_values
      in dwi.py, falling back to a fresh header scan otherwise), then more slices.
    - 3D T1: thinner slices (more volumetric detail).
    - Everything else: higher SNR: an unmeasurable SNR (None — e.g. a flat,
      hard-masked FOV; see qc.py's _background_snr) is treated as a borderline
      PASS (the qc.py low-snr threshold) rather than as the worst possible
      score, since "unmeasured" isn't evidence of a noisy scan.
    Used to pick the single BEST row per label when several series share one
    sequence type (e.g. Seri1/Seri8/Seri9 all classified DWI).
    """
    qc = row.get("qc") or {}
    status_rank = {"pass": 2, "warn": 1, "error": 0}.get(qc.get("status"), 0)
    metrics = qc.get("metrics") or {}
    snr = metrics.get("snr")
    if snr is None:
        snr = 8.0                                      # qc.py's low-snr pass/warn threshold

    if row["label"] == "DWI":
        buckets = metrics.get("bvalue_buckets")
        if buckets is None:                            # QC didn't run on this row
            buckets = count_bvalue_buckets(row["series"])
        return (status_rank, buckets, row.get("n_slices", 0))
    if row["label"] == "3D T1":
        thickness = (row.get("acq") or {}).get("thickness_mm")
        return (status_rank, -thickness if thickness is not None else float("-inf"))
    return (status_rank, snr)


def select_series(manifest: dict, slices: int, one_per_label: bool,
                  skip_qc_warn: bool):
    """Build SeriesImages for the primary series in the manifest.

    one_per_label picks the single BEST candidate per sequence type via
    _rank_key (not just the first one encountered in manifest order).
    """
    candidates = []
    for row in manifest["series"]:
        if not row.get("use_for_analysis"):
            continue
        qc = row.get("qc", {})
        if skip_qc_warn and qc.get("status") == "warn":
            logger.info("skipping %s (%s) — QC: %s",
                       row["series"], row["label"], ", ".join(qc.get("flags", [])))
            continue
        candidates.append(row)

    if one_per_label:
        best: dict[str, dict] = {}
        best_rank: dict[str, tuple] = {}
        for row in candidates:
            label = row["label"]
            rank = _rank_key(row)                       # computed once per row, not per comparison
            if label not in best or rank > best_rank[label]:
                best[label] = row
                best_rank[label] = rank
        candidates = list(best.values())

    out = []
    for row in candidates:
        out.extend(build_series_images(row, slices))
    return out


def write_report(result, study_meta: dict) -> None:
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
