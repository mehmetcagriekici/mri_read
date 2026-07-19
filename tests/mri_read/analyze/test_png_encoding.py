import io
from unittest.mock import patch

import numpy as np
from PIL import Image

from mri_read.analyze.png_encoding import volume_to_pngs


def test_encodes_chosen_indices_as_valid_pngs():
    vol = np.random.default_rng(0).normal(50, 10, (5, 8, 8)).astype(np.float32)
    pngs = volume_to_pngs(vol, idx=[0, 2, 4])

    assert len(pngs) == 3
    for png_bytes in pngs:
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (8, 8)
        assert img.mode in ("L", "P")


def test_uses_one_consistent_window_across_slices():
    """A dim slice and a bright slice windowed independently (per-slice) would
    each get stretched to fill 0-255 on their own and look equally bright.
    volume_to_pngs must window the whole volume ONCE so the dim slice stays
    visibly dimmer than the bright one in the encoded output too.
    """
    rng = np.random.default_rng(0)
    vol = np.zeros((2, 8, 8), dtype=np.float32)  # background floor = 0
    vol[0, 2:-2, 2:-2] = rng.normal(50, 3, (4, 4))    # dim tissue patch
    vol[1, 2:-2, 2:-2] = rng.normal(150, 3, (4, 4))   # bright tissue patch
    pngs = volume_to_pngs(vol, idx=[0, 1])

    dim_arr = np.array(Image.open(io.BytesIO(pngs[0])))
    bright_arr = np.array(Image.open(io.BytesIO(pngs[1])))
    assert bright_arr.mean() > dim_arr.mean()


def test_explicit_bounds_are_used_instead_of_recomputing():
    vol = np.random.default_rng(0).normal(50, 10, (3, 8, 8)).astype(np.float32)
    with patch("mri_read.analyze.png_encoding.volume_window_bounds") as mock_compute:
        volume_to_pngs(vol, idx=[0], bounds=(0.0, 100.0))
    mock_compute.assert_not_called()


def test_no_bounds_given_falls_back_to_computing_them():
    vol = np.random.default_rng(0).normal(50, 10, (3, 8, 8)).astype(np.float32)
    with patch("mri_read.analyze.png_encoding.volume_window_bounds",
              return_value=(0.0, 100.0)) as mock_compute:
        volume_to_pngs(vol, idx=[0])
    mock_compute.assert_called_once()
