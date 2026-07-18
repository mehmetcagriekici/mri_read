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

import numpy as np
from PIL import Image

from mri_read.dwi import diffusion_views
from mri_read.engine import SeriesImages
from mri_read.mri import (apply_window, foreground_fraction, load_series,
                          volume_window_bounds)
from mri_read.paths import OUT


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
    """Build SeriesImages for one manifest row, DWI handled specially.

    Shared by select_series (the fixed pipeline) and select_named_series (an
    agent picking specific series by name) so both go through one code path.
    """
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
        out.extend(build_series_images(row, slices))
    return out


def select_named_series(manifest: dict, names: list[str],
                        slices: int) -> list[SeriesImages]:
    """Build SeriesImages for specific series named explicitly (e.g. by an agent).

    Unlike select_series, this doesn't filter by use_for_analysis/QC — the
    caller already decided which series it wants.
    """
    rows = {r["series"]: r for r in manifest["series"]}
    out = []
    for name in names:
        row = rows.get(name)
        if row is not None:
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
