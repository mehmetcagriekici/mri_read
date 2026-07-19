"""Computing an ADC map from two b-value volumes."""

from __future__ import annotations

import numpy as np


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
