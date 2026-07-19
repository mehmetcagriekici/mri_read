"""Reading one series folder from disk: full pixel volume or header-only."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pydicom

from mri_read.mri.geometry import _slice_position
from mri_read.mri.tags import extract_tags
from mri_read.mri.types import Series
from mri_read.paths import series_dir


@lru_cache(maxsize=None)
def load_series(name: str) -> Series:
    """Read every slice in a series folder and return a sorted 3D Series.

    Steps:
      1. Read all .dcm files (force=True tolerates minor header quirks).
      2. Sort by true geometric position so the volume is anatomically ordered.
      3. Convert each slice to real-valued float32 via RescaleSlope/Intercept.
      4. Stack into a single (slices, rows, cols) array, keeping only slices
         that share the most common shape (a stray odd-sized frame — e.g. a
         localizer mixed in — would otherwise break np.stack).

    Cached per process: agent.py's run_qc() and select_series() both load the
    same series in a single run (QC needs pixels for contrast/SNR, analysis
    needs them for slice selection) — without this, a 150+-slice series like
    3D T1 gets decoded from disk twice back to back. Callers must treat the
    returned Series as read-only; the array is shared across callers.
    """
    folder = series_dir(name)
    files = sorted(folder.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files in {folder}")

    # Read full datasets (pixels included). force=True: don't choke on files
    # missing the formal DICOM preamble.
    datasets = [pydicom.dcmread(str(f), force=True) for f in files]
    datasets.sort(key=_slice_position)

    frames = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        # Apply the linear rescale to get real intensities. `or 1`/`or 0` guard
        # against tags present but set to None/empty.
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        frames.append(arr * slope + intercept)

    # Guard against mixed frame sizes: keep only the modal (most common) shape.
    shapes = [f.shape for f in frames]
    modal = max(set(shapes), key=shapes.count)
    frames = [f for f in frames if f.shape == modal]
    volume = np.stack(frames, axis=0)            # -> (slices, rows, cols)

    # Metadata comes from the first (lowest-position) slice; the tags we extract
    # are constant across a series.
    return Series(name=name, volume=volume, tags=extract_tags(datasets[0]))


def inspect_series(name: str) -> dict:
    """Header-only look at a series (no pixel decode) — fast, for the manifest.

    stop_before_pixels=True makes pydicom skip the (large) pixel data, so this
    is much cheaper than load_series. We read files until one parses, take its
    tags, and count the .dcm files for the slice total.

    Returns {"name", "n_slices", "tags"}.
    """
    folder = series_dir(name)
    files = sorted(folder.glob("*.dcm"))
    ds = None
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            break                                    # first readable header wins
        except Exception:                            # noqa: BLE001 - be forgiving
            continue
    if ds is None:
        return {"name": name, "n_slices": 0, "tags": {}}
    return {"name": name, "n_slices": len(files), "tags": extract_tags(ds)}
