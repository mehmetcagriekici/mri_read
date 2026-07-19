"""Building SeriesImages (engine input) for one manifest row."""

from __future__ import annotations

from mri_read.analyze.png_encoding import volume_to_pngs
from mri_read.analyze.slice_selection import content_indices
from mri_read.dwi import diffusion_views
from mri_read.engine import SeriesImages
from mri_read.mri import load_series


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
        # s.window_bounds is cached on the Series -- if qc.run_qc() already
        # ran on this series (the normal agent/analyze flow), this reuses
        # its computation instead of recomputing the same percentile pass.
        slice_pngs=volume_to_pngs(s.volume, idx, bounds=s.window_bounds),
    )]
