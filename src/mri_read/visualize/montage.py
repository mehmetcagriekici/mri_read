"""Building a single grid-image montage from a volume's slices."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mri_read.mri import window_to_uint8


def montage(volume: np.ndarray, cols: int = 6, max_tiles: int = 24) -> Image.Image:
    """Build a single grid image from up to `max_tiles` slices of a volume.

    We sample at most max_tiles slices spread evenly across the stack, window
    each to 8-bit, then paste them into a rows x cols canvas. The result is
    downscaled to <=1400px so the PNGs stay small.
    """
    n = volume.shape[0]
    # Evenly spaced slice indices across the whole stack (0..n-1).
    idx = np.linspace(0, n - 1, min(n, max_tiles)).astype(int)
    tiles = [window_to_uint8(volume[i]) for i in idx]

    h, w = tiles[0].shape                    # all slices in a series share size
    rows = (len(tiles) + cols - 1) // cols   # ceil-divide to fit every tile
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)               # grid position of this tile
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile   # paste into cell

    img = Image.fromarray(canvas)
    img.thumbnail((1400, 1400))              # cap resolution -> reasonable file size
    return img
