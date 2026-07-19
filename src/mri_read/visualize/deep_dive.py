"""Exporting every slice of one or all series as individual PNGs."""

from __future__ import annotations

from PIL import Image

from mri_read.mri import list_series, load_series, window_to_uint8
from mri_read.paths import OUT


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
