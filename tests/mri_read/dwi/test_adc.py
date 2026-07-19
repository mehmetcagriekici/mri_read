import numpy as np

from mri_read.dwi.adc import compute_adc


def test_returns_none_with_fewer_than_two_bvalues():
    assert compute_adc({0.0: np.ones((2, 4, 4))}) is None
    assert compute_adc({}) is None


def test_adc_is_nonnegative_and_zero_where_signal_unchanged():
    vol = np.full((2, 4, 4), 100.0, dtype=np.float32)
    by_b = {0.0: vol, 1000.0: vol}
    adc = compute_adc(by_b)
    assert adc is not None
    assert adc.shape == vol.shape
    assert np.all(adc >= 0)
    assert np.allclose(adc, 0, atol=1e-3)  # no signal drop -> ~0 diffusion


def test_adc_is_positive_where_signal_drops_with_higher_b():
    lo = np.full((1, 2, 2), 100.0, dtype=np.float32)
    hi = np.full((1, 2, 2), 30.0, dtype=np.float32)   # signal loss at high b
    adc = compute_adc({0.0: lo, 1000.0: hi})
    assert np.all(adc > 0)


def test_uses_lowest_and_highest_b_when_more_than_two_present():
    lo = np.full((1, 2, 2), 100.0, dtype=np.float32)
    mid = np.full((1, 2, 2), 100.0, dtype=np.float32)
    hi = np.full((1, 2, 2), 30.0, dtype=np.float32)
    adc_two = compute_adc({0.0: lo, 1000.0: hi})
    adc_three = compute_adc({0.0: lo, 500.0: mid, 1000.0: hi})
    assert np.allclose(adc_two, adc_three)


def test_mismatched_slice_counts_are_aligned_to_the_shorter_volume():
    lo = np.full((3, 2, 2), 100.0, dtype=np.float32)
    hi = np.full((2, 2, 2), 30.0, dtype=np.float32)
    adc = compute_adc({0.0: lo, 1000.0: hi})
    assert adc.shape == (2, 2, 2)
