"""Windowing chosen slices and encoding them to PNG bytes for engine input."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from mri_read.mri import apply_window, volume_window_bounds


def volume_to_pngs(volume: np.ndarray, idx: list[int]) -> list[bytes]:
    """Window the whole volume once (consistent contrast), encode chosen slices."""
    lo, hi = volume_window_bounds(volume)
    pngs = []
    for i in idx:
        buf = io.BytesIO()
        Image.fromarray(apply_window(volume[i], lo, hi)).save(buf, format="PNG")
        pngs.append(buf.getvalue())
    return pngs
