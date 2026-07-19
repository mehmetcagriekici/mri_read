"""Estimating how much of a slice is tissue vs. empty background.

Distinct from windowing.py: this is a content-detection metric (used by QC and
by content-aware slice selection), not an intensity-to-8-bit mapping.
"""

from __future__ import annotations

import numpy as np


def foreground_fraction(img: np.ndarray) -> float:
    """Estimate what fraction of a slice is tissue vs. empty background.

    Method: threshold at 10% of the slice's dynamic range and count pixels
    above it. Crude but reliable enough to distinguish an empty end-of-stack
    slice (~0) from one packed with brain (~0.3–0.6). Used by QC (empty-slice
    detection) and by content-aware slice selection in analyze.py.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:                                     # flat slice -> no content
        return 0.0
    thresh = lo + 0.10 * (hi - lo)
    return float((img > thresh).mean())              # mean of a bool array = fraction
