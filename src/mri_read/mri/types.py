"""The Series data shape shared by every downstream module."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from mri_read.mri.windowing import volume_window_bounds


@dataclass
class Series:
    """A fully loaded series: the pixel data as one 3D volume, plus metadata.

    Attributes:
        name:   folder name of the series (e.g. "Seri7").
        volume: numpy array, shape (n_slices, rows, cols), dtype float32,
                slices ordered head-to-foot by geometric position.
        tags:   dict of the acquisition/geometry tags we care about
                (see extract_tags).
    """
    name: str
    volume: np.ndarray                      # (slices, rows, cols), float32
    tags: dict = field(default_factory=dict)

    @property
    def n_slices(self) -> int:
        """Number of slices in the stacked volume (its first axis)."""
        return self.volume.shape[0]

    @cached_property
    def window_bounds(self) -> tuple[float, float]:
        """Volume-level (lo, hi) window bounds, computed once and cached.

        Both qc.checks (the low-contrast check) and analyze.png_encoding
        (windowing slices for the vision engine) need this for the same
        Series -- it's an exact percentile over the full-resolution volume,
        not cheap (profiling: ~3s for a 150-slice 3D T1). Caching it here,
        scoped to this Series instance (itself already process-lifetime
        cached by mri.loading.load_series's lru_cache), means it's computed
        once per series per process instead of once per consumer.
        """
        return volume_window_bounds(self.volume)
