"""Mapping float MRI intensities to 8-bit for display / model input."""

from __future__ import annotations

import numpy as np


def window_to_uint8(img: np.ndarray, ds=None) -> np.ndarray:
    """Map one float slice to 0–255 for display (PER-SLICE windowing).

    "Windowing" = choosing an intensity range [lo, hi] to stretch onto 0–255;
    everything below lo is black, above hi is white. Used by visualize.py for
    quick-look montages.

    Priority:
      1. The DICOM-provided WindowCenter/WindowWidth (what the scanner intended).
      2. Otherwise a robust 1st–99th percentile stretch of this slice.

    NOTE: this windows each slice independently, so brightness isn't comparable
    across slices. For model input use volume_window_bounds + apply_window,
    which window a whole series consistently.
    """
    center = width = None
    if ds is not None:
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            # These tags are sometimes multi-valued; take the first entry.
            center = float(wc[0] if hasattr(wc, "__iter__") else wc)
            width = float(ww[0] if hasattr(ww, "__iter__") else ww)

    if center is None or not width:
        lo, hi = np.percentile(img, [1, 99])         # robust auto-window
    else:
        lo, hi = center - width / 2, center + width / 2

    if hi <= lo:                                     # degenerate (flat) slice
        hi = lo + 1
    out = np.clip((img - lo) / (hi - lo), 0, 1)      # normalize to 0..1
    return (out * 255).astype(np.uint8)              # -> 8-bit greyscale


def volume_window_bounds(volume: np.ndarray) -> tuple[float, float]:
    """Compute ONE (lo, hi) window for a whole series so its slices match.

    Why over the whole volume: a model (or a human) comparing slices needs them
    windowed identically — per-slice windowing makes a dark slice and a bright
    slice look the same, hiding real intensity differences.

    Why foreground-only: MR background is ~0. Including all those zeros drags the
    1st percentile down and washes out tissue contrast, so we take percentiles
    over voxels above the volume minimum only.
    """
    fg = volume[volume > volume.min()]               # drop the background floor
    if fg.size == 0:                                 # entirely flat volume
        fg = volume.ravel()
    lo, hi = np.percentile(fg, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def apply_window(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a slice to 8-bit using explicit bounds from volume_window_bounds."""
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)
