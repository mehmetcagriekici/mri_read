from unittest.mock import patch

import numpy as np

from mri_read.mri.types import Series


def test_n_slices_reflects_first_volume_axis():
    vol = np.zeros((7, 4, 4), dtype=np.float32)
    s = Series(name="Seri1", volume=vol)
    assert s.n_slices == 7


def test_tags_defaults_to_empty_dict():
    s = Series(name="Seri1", volume=np.zeros((1, 2, 2)))
    assert s.tags == {}


def test_tags_are_independent_per_instance():
    a = Series(name="a", volume=np.zeros((1, 2, 2)))
    b = Series(name="b", volume=np.zeros((1, 2, 2)))
    a.tags["x"] = 1
    assert b.tags == {}


def test_window_bounds_matches_the_free_function():
    from mri_read.mri.windowing import volume_window_bounds
    vol = np.random.default_rng(0).normal(50, 10, (3, 8, 8)).astype(np.float32)
    s = Series(name="Seri1", volume=vol)
    assert s.window_bounds == volume_window_bounds(vol)


def test_window_bounds_is_computed_only_once_per_instance():
    vol = np.random.default_rng(0).normal(50, 10, (3, 8, 8)).astype(np.float32)
    s = Series(name="Seri1", volume=vol)
    with patch("mri_read.mri.types.volume_window_bounds",
              return_value=(0.0, 1.0)) as mock_compute:
        s.window_bounds        # first access -- computes
        s.window_bounds        # second access -- must reuse the cached value
    mock_compute.assert_called_once()


def test_window_bounds_is_independent_per_instance():
    vol_a = np.zeros((2, 4, 4), dtype=np.float32)
    vol_b = np.ones((2, 4, 4), dtype=np.float32) * 50
    a = Series(name="a", volume=vol_a)
    b = Series(name="b", volume=vol_b)
    assert a.window_bounds != b.window_bounds
