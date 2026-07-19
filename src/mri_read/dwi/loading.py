"""Splitting a DWI series folder into one volume per b-value."""

from __future__ import annotations

import numpy as np
import pydicom

from mri_read.mri import DATA_DIR, _slice_position, read_bvalue


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
