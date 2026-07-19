from unittest.mock import patch

import numpy as np

from mri_read.analyze.images import build_series_images
from mri_read.mri.types import Series


def test_non_dwi_series_loads_and_encodes_one_image_set():
    row = {"series": "Seri6", "label": "T2 FLAIR", "plane": "Axial"}
    vol = np.random.default_rng(0).normal(50, 10, (10, 8, 8)).astype(np.float32)
    fake_series = Series(name="Seri6", volume=vol)
    with patch("mri_read.analyze.images.load_series", return_value=fake_series):
        out = build_series_images(row, slices=3)

    assert len(out) == 1
    assert out[0].series == "Seri6"
    assert out[0].label == "T2 FLAIR"
    assert len(out[0].slice_pngs) == len(out[0].slice_indices)


def test_non_dwi_series_reuses_the_series_cached_window_bounds():
    """build_series_images must pass s.window_bounds through to
    volume_to_pngs rather than letting it recompute the percentile pass --
    that's the whole point of caching it on the Series in the first place.
    """
    row = {"series": "Seri6", "label": "T2 FLAIR", "plane": "Axial"}
    vol = np.random.default_rng(0).normal(50, 10, (10, 8, 8)).astype(np.float32)
    fake_series = Series(name="Seri6", volume=vol)
    with patch("mri_read.analyze.images.load_series", return_value=fake_series), \
         patch("mri_read.analyze.images.volume_to_pngs", return_value=[b"png"]) as mock_encode:
        build_series_images(row, slices=3)

    assert mock_encode.call_args.kwargs["bounds"] == fake_series.window_bounds


def test_dwi_series_routes_through_diffusion_views():
    row = {"series": "Seri1", "label": "DWI", "plane": "Axial"}
    high_b = np.random.default_rng(0).normal(50, 10, (5, 8, 8)).astype(np.float32)
    adc = np.random.default_rng(1).normal(500, 50, (5, 8, 8)).astype(np.float32)
    views = {"high_b": high_b, "b_value": 1000.0, "adc": adc, "note": "b-values found: [0, 1000]"}
    with patch("mri_read.analyze.images.diffusion_views", return_value=views):
        out = build_series_images(row, slices=3)

    labels = [s.label for s in out]
    assert "DWI (b=1000.0)" in labels
    assert "DWI ADC map" in labels


def test_dwi_series_without_adc_only_yields_high_b():
    row = {"series": "Seri1", "label": "DWI", "plane": "Axial"}
    high_b = np.random.default_rng(0).normal(50, 10, (5, 8, 8)).astype(np.float32)
    views = {"high_b": high_b, "b_value": None, "adc": None, "note": "no b-values"}
    with patch("mri_read.analyze.images.diffusion_views", return_value=views):
        out = build_series_images(row, slices=3)

    assert len(out) == 1
    assert out[0].label == "DWI"  # no b-value known -> plain label


def test_non_dwi_series_carries_acq_params_through():
    row = {"series": "Seri6", "label": "T2 FLAIR", "plane": "Axial",
          "acq": {"TE": 96.36, "TR": 8200.0, "TI": 2376.04}}
    vol = np.random.default_rng(0).normal(50, 10, (10, 8, 8)).astype(np.float32)
    fake_series = Series(name="Seri6", volume=vol)
    with patch("mri_read.analyze.images.load_series", return_value=fake_series):
        out = build_series_images(row, slices=3)

    assert out[0].acq == {"TE": 96.36, "TR": 8200.0, "TI": 2376.04}


def test_non_dwi_series_missing_acq_defaults_to_empty_dict():
    row = {"series": "Seri6", "label": "T2 FLAIR", "plane": "Axial"}  # no "acq" key
    vol = np.random.default_rng(0).normal(50, 10, (10, 8, 8)).astype(np.float32)
    fake_series = Series(name="Seri6", volume=vol)
    with patch("mri_read.analyze.images.load_series", return_value=fake_series):
        out = build_series_images(row, slices=3)

    assert out[0].acq == {}


def test_dwi_series_carries_acq_params_through_to_both_images():
    row = {"series": "Seri1", "label": "DWI", "plane": "Axial",
          "acq": {"TE": 85.4, "TR": 4281.0}}
    high_b = np.random.default_rng(0).normal(50, 10, (5, 8, 8)).astype(np.float32)
    adc = np.random.default_rng(1).normal(500, 50, (5, 8, 8)).astype(np.float32)
    views = {"high_b": high_b, "b_value": 1000.0, "adc": adc, "note": "b-values found: [0, 1000]"}
    with patch("mri_read.analyze.images.diffusion_views", return_value=views):
        out = build_series_images(row, slices=3)

    assert all(s.acq == {"TE": 85.4, "TR": 4281.0} for s in out)
