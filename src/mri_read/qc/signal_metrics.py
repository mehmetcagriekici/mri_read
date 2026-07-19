"""Pixel-based signal-quality measurement (needs the decoded volume)."""

from __future__ import annotations

import numpy as np


def _background_snr(volume: np.ndarray) -> float | None:
    """Rough signal-to-noise ratio of the middle slice, or None if unmeasurable.

    Trick: the four image CORNERS are (almost) always empty background, so their
    standard deviation estimates the noise floor. Signal = mean intensity of the
    foreground (tissue). SNR = signal / noise. A low value means a grainy scan.

    Returns None when the corners are perfectly flat (std == 0) — common with a
    hard-masked circular FOV reconstruction where the true corners are exact
    zero. There's no noise information in a flat patch, so a signal/~0 ratio
    would be a meaningless, arbitrarily huge number rather than a real
    measurement — better to say "unknown" than report a false one.
    """
    z = volume.shape[0] // 2                          # middle slice = most tissue
    img = volume[z]
    h, w = img.shape
    c = max(4, min(h, w) // 16)                       # corner patch size (>=4 px)
    corners = np.concatenate([                        # four corner patches
        img[:c, :c].ravel(), img[:c, -c:].ravel(),
        img[-c:, :c].ravel(), img[-c:, -c:].ravel(),
    ])
    noise = float(corners.std())
    if noise < 1e-6:                                  # flat corners -- no noise to measure
        return None
    lo, hi = float(img.min()), float(img.max())
    fg = img[img > lo + 0.10 * (hi - lo)]            # foreground (tissue) pixels
    signal = float(fg.mean()) if fg.size else 0.0
    return signal / noise
