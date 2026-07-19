from unittest.mock import patch

import numpy as np

from mri_read.mri.types import Series
from mri_read.qc.checks import run_qc


def _clean_volume(n_slices=5, size=20):
    """A volume with realistic tissue contrast/noise -- should trip no flags.

    Uniform-intensity foreground would itself read as "low-contrast" (no
    internal variation), so this needs real tissue-like variation, not just
    a foreground/background split.
    """
    rng = np.random.default_rng(0)
    vol = rng.normal(2.0, 0.5, (n_slices, size, size)).astype(np.float32)  # background noise
    # Border must stay wider than _background_snr's corner-patch size
    # (max(4, min(h,w)//16)) or the "corners" sample tissue too, which
    # inflates the apparent noise and drags SNR down artificially.
    vol[:, 6:-6, 6:-6] = rng.normal(100.0, 15.0, (n_slices, size - 12, size - 12)).astype(np.float32)
    return vol


def _patched(positions, instances, bvalues=None, volume=None, load_error=None):
    ctx = [
        patch("mri_read.qc.checks._positions_and_instances",
             return_value=(positions, instances, bvalues or [None] * len(positions))),
    ]
    if load_error is not None:
        ctx.append(patch("mri_read.qc.checks.load_series", side_effect=load_error))
    else:
        # A real Series (not a bare SimpleNamespace) -- run_qc now reads
        # series.window_bounds, a cached_property that only exists on the
        # real dataclass.
        fake_series = Series(name="fake", volume=volume if volume is not None else _clean_volume(len(positions)))
        ctx.append(patch("mri_read.qc.checks.load_series", return_value=fake_series))
    return ctx


def test_clean_series_passes_with_no_flags():
    positions = [0, 5, 10, 15, 20]
    instances = [1, 2, 3, 4, 5]
    p1, p2 = _patched(positions, instances)
    with p1, p2:
        result = run_qc("SeriClean")

    assert result["status"] == "pass"
    assert result["flags"] == []


def test_missing_slices_flagged_from_instance_number_span():
    positions = [0, 5, 10]
    instances = [1, 2, 5]  # span 1..5 = 5 slots but only 3 files
    p1, p2 = _patched(positions, instances)
    with p1, p2:
        result = run_qc("SeriGap")

    assert "missing-slices(2)" in result["flags"]
    assert result["status"] == "warn"


def test_uneven_spacing_flagged():
    positions = [0, 1, 2, 3, 20]  # one huge gap among tiny ones
    instances = [1, 2, 3, 4, 5]
    with _patched(positions, instances)[0], _patched(positions, instances)[1]:
        result = run_qc("SeriUneven")

    assert "uneven-spacing" in result["flags"]


def test_low_contrast_flagged_for_nearly_flat_volume():
    positions, instances = [0, 1, 2], [1, 2, 3]
    flat_vol = np.full((3, 10, 10), 100.0, dtype=np.float32)
    flat_vol += np.random.default_rng(0).normal(0, 0.01, flat_vol.shape).astype(np.float32)
    p1, p2 = _patched(positions, instances, volume=flat_vol)
    with p1, p2:
        result = run_qc("SeriFlat")

    assert "low-contrast" in result["flags"]


def test_low_snr_flagged():
    positions, instances = [0, 1, 2], [1, 2, 3]
    p1, p2 = _patched(positions, instances)
    with p1, p2, patch("mri_read.qc.checks._background_snr", return_value=3.0):
        result = run_qc("SeriNoisy")

    assert "low-snr" in result["flags"]
    assert result["metrics"]["snr"] == 3.0


def test_unmeasurable_snr_is_not_flagged():
    positions, instances = [0, 1, 2], [1, 2, 3]
    p1, p2 = _patched(positions, instances)
    with p1, p2, patch("mri_read.qc.checks._background_snr", return_value=None):
        result = run_qc("SeriFlatCorners")

    assert "low-snr" not in result["flags"]
    assert result["metrics"]["snr"] is None


def test_mostly_empty_flagged():
    positions, instances = [0, 1, 2, 3], [1, 2, 3, 4]
    vol = np.zeros((4, 10, 10), dtype=np.float32)
    vol[0, 2:-2, 2:-2] = 100.0  # only 1 of 4 slices has real content
    p1, p2 = _patched(positions, instances, volume=vol)
    with p1, p2:
        result = run_qc("SeriEmpty")

    assert "mostly-empty" in result["flags"]


def test_load_failure_yields_error_status():
    positions, instances = [0, 1, 2], [1, 2, 3]
    p1, p2 = _patched(positions, instances, load_error=RuntimeError("disk error"))
    with p1, p2:
        result = run_qc("SeriBroken")

    assert result["status"] == "error"
    assert any("load-failed" in f for f in result["flags"])
    # pixel-based checks never ran -- geometry metrics collected first are kept
    assert "contrast" not in result["metrics"]
