import numpy as np
import pytest

from mri_read.analyze.slice_selection import content_indices


def test_picks_k_indices_spread_across_content():
    # A perfectly flat slice reads as "no content" (foreground_fraction
    # treats zero dynamic range as background) -- content slices need a
    # real foreground/background split within the slice, not just a
    # different uniform value from their neighbors.
    vol = np.zeros((20, 10, 10), dtype=np.float32)
    vol[2:18, 3:7, 3:7] = 100.0  # a tissue patch present in slices 2..17
    idx = content_indices(vol, k=4)
    assert len(idx) <= 4
    assert min(idx) >= 2 and max(idx) <= 17


def test_drops_near_empty_end_slices():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[3:7, 3:7, 3:7] = 100.0  # only middle slices have a tissue patch
    idx = content_indices(vol, k=4)
    assert all(3 <= i <= 6 for i in idx)


def test_falls_back_to_whole_volume_when_nothing_passes_threshold():
    vol = np.zeros((5, 10, 10), dtype=np.float32)  # entirely empty
    idx = content_indices(vol, k=3)
    assert len(idx) > 0
    assert max(idx) <= 4


def test_k_larger_than_available_range_is_capped():
    vol = np.zeros((3, 10, 10), dtype=np.float32)
    vol[:] = 100.0
    idx = content_indices(vol, k=10)
    assert len(idx) <= 3


def test_k_zero_returns_no_slices_without_crashing():
    vol = np.zeros((5, 10, 10), dtype=np.float32)
    vol[:, 3:7, 3:7] = 100.0
    assert content_indices(vol, k=0) == []


def test_single_slice_volume():
    vol = np.full((1, 10, 10), 100.0, dtype=np.float32)
    vol[0, 3:7, 3:7] = 200.0  # give it a foreground/background split
    idx = content_indices(vol, k=4)
    assert idx == [0]


def test_negative_k_raises_rather_than_silently_misbehaving():
    """Not a supported input (k comes from the trusted local --slices CLI
    flag), but document that a typo like --slices -1 fails loudly with a
    clear numpy error rather than doing something silently wrong.
    """
    vol = np.zeros((5, 10, 10), dtype=np.float32)
    vol[:, 3:7, 3:7] = 100.0
    with pytest.raises(ValueError):
        content_indices(vol, k=-1)
