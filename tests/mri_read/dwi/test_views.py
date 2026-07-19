from unittest.mock import patch

import numpy as np

from mri_read.dwi.views import diffusion_views


def test_uses_highest_bvalue_and_computes_adc():
    by_b = {
        0.0: np.full((2, 4, 4), 100.0, dtype=np.float32),
        1000.0: np.full((2, 4, 4), 30.0, dtype=np.float32),
    }
    with patch("mri_read.dwi.views.load_by_bvalue", return_value=by_b):
        result = diffusion_views("Seri1")

    assert result["b_value"] == 1000.0
    assert result["high_b"] is by_b[1000.0]
    assert result["adc"] is not None
    assert "1000.0" in result["note"] or "1000" in result["note"]


def test_falls_back_to_full_stack_when_no_bvalues_tagged():
    by_b = {-1.0: np.zeros((2, 4, 4), dtype=np.float32)}
    with patch("mri_read.dwi.views.load_by_bvalue", return_value=by_b):
        result = diffusion_views("SeriNoB")

    assert result["b_value"] is None
    assert result["adc"] is None
    assert result["high_b"].shape == (2, 4, 4)
    assert "no b-value" in result["note"]


def test_single_bvalue_has_no_adc():
    by_b = {1000.0: np.zeros((2, 4, 4), dtype=np.float32)}
    with patch("mri_read.dwi.views.load_by_bvalue", return_value=by_b):
        result = diffusion_views("SeriOneB")

    assert result["b_value"] == 1000.0
    assert result["adc"] is None
