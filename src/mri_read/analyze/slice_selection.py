"""Picking which slice indices of a volume to hand to a vision engine."""

from __future__ import annotations

import numpy as np

from mri_read.mri import foreground_fraction


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
