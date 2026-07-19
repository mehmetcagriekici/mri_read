"""The Series data shape shared by every downstream module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
