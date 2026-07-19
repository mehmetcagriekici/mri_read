"""Windowing chosen slices and encoding them to PNG bytes for engine input."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from mri_read.mri import apply_window, volume_window_bounds


def volume_to_pngs(volume: np.ndarray, idx: list[int],
                   bounds: tuple[float, float] | None = None) -> list[bytes]:
    """Window the whole volume once (consistent contrast), encode chosen slices.

    `bounds`, when given, skips recomputing volume_window_bounds -- callers
    that already have it cached (e.g. mri.types.Series.window_bounds, reused
    from qc's low-contrast check on the same series) should pass it through
    rather than paying for the percentile pass twice. DWI's derived high-b/
    ADC volumes aren't tied to a cached Series, so they always recompute.
    """
    lo, hi = bounds if bounds is not None else volume_window_bounds(volume)
    pngs = []
    for i in idx:
        buf = io.BytesIO()
        Image.fromarray(apply_window(volume[i], lo, hi)).save(buf, format="PNG")
        pngs.append(buf.getvalue())
    return pngs
