import numpy as np

from mri_read.mri.foreground import foreground_fraction


def test_all_background_is_zero_fraction():
    img = np.zeros((10, 10), dtype=np.float32)
    assert foreground_fraction(img) == 0.0


def test_half_bright_half_dark():
    img = np.zeros((10, 10), dtype=np.float32)
    img[:5, :] = 100.0
    frac = foreground_fraction(img)
    assert 0.4 < frac < 0.6


def test_mostly_tissue_is_high_fraction():
    img = np.full((10, 10), 100.0, dtype=np.float32)
    img[0, 0] = 0.0  # one background pixel
    assert foreground_fraction(img) > 0.9
