from types import SimpleNamespace

import numpy as np

from mri_read.mri.windowing import (apply_window, volume_window_bounds,
                                    window_to_uint8)


def test_window_to_uint8_uses_dicom_window_when_present():
    img = np.array([[0, 50], [100, 150]], dtype=np.float32)
    ds = SimpleNamespace(WindowCenter=75, WindowWidth=150)  # -> [0, 150]
    out = window_to_uint8(img, ds)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0      # 0 maps to bottom of window
    assert out[1, 1] == 255    # 150 maps to top of window


def test_window_to_uint8_falls_back_to_percentile_without_dicom_window():
    img = np.linspace(0, 100, 100, dtype=np.float32).reshape(10, 10)
    out = window_to_uint8(img)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_window_to_uint8_handles_flat_slice():
    img = np.full((4, 4), 5.0, dtype=np.float32)
    out = window_to_uint8(img)
    assert out.shape == (4, 4)


def test_volume_window_bounds_ignores_background_floor():
    vol = np.zeros((2, 10, 10), dtype=np.float32)
    vol[0, 0, 0] = 100.0  # a little real signal amid a lot of background
    lo, hi = volume_window_bounds(vol)
    assert lo < hi


def test_volume_window_bounds_handles_entirely_flat_volume():
    vol = np.zeros((2, 4, 4), dtype=np.float32)
    lo, hi = volume_window_bounds(vol)
    assert hi > lo  # degenerate case still returns a usable, non-zero-width window


def test_apply_window_clips_to_0_255():
    img = np.array([[-10, 50], [200, 1000]], dtype=np.float32)
    out = apply_window(img, lo=0, hi=100)
    assert out[0, 0] == 0
    assert out[1, 1] == 255
