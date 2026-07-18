"""
Step 2 — Load and visualize.

Turns series into images we can actually look at — the fastest way to sanity
check what each sequence is and confirm the loader works. Uses mri.load_series
for the heavy lifting and PIL to write PNGs.

Two outputs, written to output/:
  1. Overview: one montage PNG per series (a grid of evenly spaced slices) plus
     a printed tag table (plane, TE/TR, thickness, protocol) to identify every
     sequence at a glance.
  2. Deep-dive: every slice of a series exported as individual PNGs.

CLI entry point: src/cmd/visualize.py

Note: this uses PER-SLICE windowing (window_to_uint8) — fine for eyeballing, but
the analysis path uses volume-level windowing instead (see analyze.py).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from mri_read.mri import list_series, load_series, window_to_uint8
from mri_read.paths import OUT


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


def overview() -> None:
    """Write one montage PNG per series and print a one-line tag summary each."""
    OUT.mkdir(exist_ok=True)
    print(f"{'series':7} {'plane':9} {'seq':10} {'TE':>6} {'TR':>7} "
          f"{'thick':>6} {'slices':>6}  protocol")
    print("-" * 78)
    for name in list_series():
        try:
            s = load_series(name)                    # full 3D volume + tags
        except Exception as e:                       # noqa: BLE001
            print(f"{name:7} !! failed to load: {e}")
            continue
        t = s.tags
        montage(s.volume).save(OUT / f"{name}.png")  # one montage per series
        print(f"{name:7} {t['plane'][:9]:9} {t['scanning_sequence'][:10]:10} "
              f"{str(t['echo_time_TE']):>6} {str(t['repetition_TR']):>7} "
              f"{str(t['thickness_mm']):>6} {s.n_slices:>6}  {t['protocol']}")
    print(f"\nMontages written to: {OUT}")


def deep_dive(name: str) -> None:
    """Export every slice of one series as PNG."""
    folder = OUT / f"{name}_slices"
    folder.mkdir(parents=True, exist_ok=True)
    s = load_series(name)
    for i in range(s.n_slices):
        Image.fromarray(window_to_uint8(s.volume[i])).save(
            folder / f"{i:03d}.png"
        )
    print(f"  {name}: exported {s.n_slices} slices -> {folder.name}/")


def deep_dive_all() -> None:
    """Export every slice of every series (recursive over the data folder)."""
    print("\nDeep export (all series):")
    for name in list_series():
        try:
            deep_dive(name)
        except Exception as e:                       # noqa: BLE001
            print(f"  {name}: skipped ({e})")
