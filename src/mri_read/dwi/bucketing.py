"""Counting distinct b-value groupings in a DWI series."""

from __future__ import annotations

import pydicom

from mri_read.mri import read_bvalue, series_dir


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
    files = sorted(series_dir(name).glob("*.dcm"))
    bvalues = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:                            # noqa: BLE001 - skip bad file
            continue
        bvalues.append(read_bvalue(ds))
    return count_bvalue_buckets_from_values(bvalues)
