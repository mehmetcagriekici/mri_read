"""Header-only geometric measurements (no pixel decode)."""

from __future__ import annotations

import pydicom

from mri_read.mri import _slice_position, read_bvalue, series_dir


def _positions_and_instances(name: str):
    """Return (positions, instance_numbers, bvalues) for a series, sorted.

    Header-only read. `positions` are mm along the slice normal (for the
    spacing check); `instances` are the scanner slice indices (for the
    missing-slice check); `bvalues` are each slice's diffusion b-value (or
    None), read from the same headers so a DWI-candidate ranking check later
    (analyze.py's _rank_key) doesn't have to re-open every file on disk.
    """
    files = sorted(series_dir(name).glob("*.dcm"))
    heads = []
    for f in files:
        try:
            heads.append(pydicom.dcmread(str(f), stop_before_pixels=True,
                                         force=True))
        except Exception:                            # noqa: BLE001 - skip bad file
            continue
    heads.sort(key=_slice_position)                  # anatomical order
    positions = [_slice_position(h) for h in heads]
    instances = [int(getattr(h, "InstanceNumber", 0)) for h in heads]
    bvalues = [read_bvalue(h) for h in heads]
    return positions, instances, bvalues
