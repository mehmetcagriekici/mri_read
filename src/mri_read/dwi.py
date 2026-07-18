"""
Diffusion (DWI) handling.

A DWI series often stacks multiple b-values together (e.g. b0 + b1000). The
diffusion signal lives in the CONTRAST between them, not in any single stack:

  - high-b images show restricted diffusion as bright (e.g. acute stroke),
  - the ADC map, computed from two b-values, separates true restriction from
    "T2 shine-through".

    ADC = -1/(b_high - b_low) * ln(S_high / S_low)

This module splits a DWI series by b-value and computes an ADC map when two
b-values are available. Falls back gracefully when b-values aren't tagged.

Usage (standalone check):
  python src/cmd/dwi.py Seri1
"""

from __future__ import annotations

import numpy as np
import pydicom

from mri_read.mri import DATA_DIR, _slice_position, read_bvalue


def _bucket_key(b: float | None) -> float | None:
    """Round a b-value to the nearest 10 for bucketing; untagged stays None.

    Rounding tolerates the small scanner-to-scanner noise in a nominal
    b-value (e.g. 999.7 vs 1000.2) without collapsing genuinely different
    b-values together.
    """
    return round(b, -1) if b is not None else None


def count_bvalue_buckets_from_values(bvalues: list) -> int:
    """Count distinct b-value groupings from already-read b-values.

    Same bucketing rule as count_bvalue_buckets, but takes b-values a caller
    already pulled from the headers (e.g. qc.py's per-series header pass) so
    a second full disk scan isn't needed just to count buckets.
    """
    return len({_bucket_key(b) for b in bvalues})


def count_bvalue_buckets(name: str) -> int:
    """Header-only count of distinct b-value groupings in a series.

    Untagged slices count as their own bucket (a stack that mixes tagged and
    untagged slices, e.g. b1000 + an untagged b0 reference, still has 2
    buckets). For series SELECTION only: lets select_series() prefer a DWI
    folder that actually stacks multiple diffusion conditions (needed for an
    ADC map) over a single-condition repeat acquisition — cheaper than
    load_by_bvalue since it never decodes pixel data.

    Standalone helper for callers (e.g. select_series) that don't already
    have QC's per-series b-values on hand; prefer
    count_bvalue_buckets_from_values() when they do, to avoid re-reading
    every header from disk.
    """
    files = sorted((DATA_DIR / name).glob("*.dcm"))
    bvalues = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:                            # noqa: BLE001 - skip bad file
            continue
        bvalues.append(read_bvalue(ds))
    return count_bvalue_buckets_from_values(bvalues)


def load_by_bvalue(name: str) -> dict[float, np.ndarray]:
    """Read a DWI series and split it into one 3D volume per b-value.

    Each slice carries its own b-value tag. We read every slice, group slices by
    b-value, sort each group by geometric position, and stack it. Slices with no
    b-value tag are bucketed under -1.0. Returns {b_value: volume}.
    """
    files = sorted((DATA_DIR / name).glob("*.dcm"))
    rows = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)         # need pixels here
            b = read_bvalue(ds)                              # may be None
            arr = ds.pixel_array.astype(np.float32)
            rows.append((b, _slice_position(ds), arr))       # (b, position, image)
        except Exception:                            # noqa: BLE001 - skip bad file
            continue

    # Bucket slices by b-value (None -> -1.0 so it's a valid dict key).
    groups: dict[float, list] = {}
    for b, pos, arr in rows:
        groups.setdefault(b if b is not None else -1.0, []).append((pos, arr))

    # Within each b-value, sort by position and stack into a volume, keeping only
    # the modal shape (same guard as mri.load_series).
    out = {}
    for b, items in groups.items():
        items.sort(key=lambda t: t[0])               # by geometric position
        shapes = [a.shape for _, a in items]
        modal = max(set(shapes), key=shapes.count)
        out[b] = np.stack([a for _, a in items if a.shape == modal], axis=0)
    return out


def compute_adc(by_b: dict[float, np.ndarray]) -> np.ndarray | None:
    """Compute an ADC map from the lowest and highest b-value stacks.

    ADC (apparent diffusion coefficient) removes "T2 shine-through": a lesion
    that looks bright on high-b DWI is only truly restricted if it's also DARK
    on ADC. Formula, applied per voxel:

        ADC = -1/(b_high - b_low) * ln(S_high / S_low)

    Needs at least two b-values; returns None otherwise. Result is scaled to the
    conventional x10^-6 mm^2/s units. eps guards keep log/division finite.
    """
    bs = sorted(b for b in by_b if b is not None and b >= 0)   # real b-values
    if len(bs) < 2:
        return None                                  # can't compute from one b
    b_lo, b_hi = bs[0], bs[-1]                        # e.g. b0 and b1000
    s_lo, s_hi = by_b[b_lo], by_b[b_hi]              # their signal volumes
    if s_lo.shape != s_hi.shape:                     # align if counts differ
        n = min(s_lo.shape[0], s_hi.shape[0])
        s_lo, s_hi = s_lo[:n], s_hi[:n]

    eps = 1e-6
    ratio = np.clip(s_hi + eps, eps, None) / np.clip(s_lo + eps, eps, None)
    adc = -1.0 / (b_hi - b_lo) * np.log(np.clip(ratio, eps, None))
    adc = np.clip(adc, 0, None) * 1e6                # no negatives; to std units
    return adc.astype(np.float32)


def diffusion_views(name: str) -> dict:
    """What analysis should look at for a DWI series.

    Returns {'high_b': volume, 'b_value': b, 'adc': volume|None, 'note': str}.
    Falls back to the whole stack if no b-values are tagged.
    """
    by_b = load_by_bvalue(name)
    real_bs = sorted(b for b in by_b if b is not None and b >= 0)

    if not real_bs:
        whole = np.concatenate(list(by_b.values()), axis=0) if by_b else None
        return {"high_b": whole, "b_value": None, "adc": None,
                "note": "no b-value tags found; using full stack"}

    high_b = real_bs[-1]
    return {
        "high_b": by_b[high_b],
        "b_value": high_b,
        "adc": compute_adc(by_b),
        "note": f"b-values found: {real_bs}",
    }
