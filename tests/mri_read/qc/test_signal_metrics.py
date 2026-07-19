import numpy as np

from mri_read.qc.signal_metrics import _background_snr


def test_returns_none_when_corners_are_perfectly_flat():
    vol = np.zeros((1, 20, 20), dtype=np.float32)
    vol[0, 5:15, 5:15] = 100.0  # tissue, but corners are exact zero
    assert _background_snr(vol) is None


def test_computes_a_positive_ratio_for_noisy_corners_and_real_signal():
    rng = np.random.default_rng(0)
    vol = np.zeros((1, 20, 20), dtype=np.float32)
    vol[0] = rng.normal(1.0, 0.3, (20, 20))       # background noise everywhere
    vol[0, 6:-6, 6:-6] = rng.normal(100.0, 5.0, (8, 8))  # tissue, away from corners

    snr = _background_snr(vol)
    assert snr is not None
    assert snr > 1  # signal well above the noise floor


def test_uses_the_middle_slice():
    vol = np.zeros((3, 20, 20), dtype=np.float32)
    rng = np.random.default_rng(0)
    vol[1] = rng.normal(1.0, 0.3, (20, 20))  # only the middle slice has signal
    vol[1, 6:-6, 6:-6] = rng.normal(100.0, 5.0, (8, 8))
    # slices 0 and 2 stay exactly zero (flat corners -> would read as unmeasurable)
    assert _background_snr(vol) is not None
