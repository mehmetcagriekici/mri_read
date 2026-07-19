from unittest.mock import patch

from mri_read.analyze.select import select_series
from mri_read.engine import SeriesImages


def _row(series, label, use=True, status="pass", snr=10.0):
    return {"series": series, "label": label, "use_for_analysis": use,
           "qc": {"status": status, "flags": [], "metrics": {"snr": snr}}}


def _fake_build(row, slices):
    return [SeriesImages(row["series"], row["label"], "Axial", [0], [b"png"])]


def test_skips_series_not_marked_for_analysis():
    manifest = {"series": [_row("Seri1", "T2", use=False)]}
    with patch("mri_read.analyze.select.build_series_images", side_effect=_fake_build):
        out = select_series(manifest, slices=4, one_per_label=False, skip_qc_warn=False)
    assert out == []


def test_skip_qc_warn_excludes_warned_series():
    manifest = {"series": [_row("Seri1", "T2", status="warn")]}
    with patch("mri_read.analyze.select.build_series_images", side_effect=_fake_build):
        out = select_series(manifest, slices=4, one_per_label=False, skip_qc_warn=True)
    assert out == []


def test_skip_qc_warn_false_keeps_warned_series():
    manifest = {"series": [_row("Seri1", "T2", status="warn")]}
    with patch("mri_read.analyze.select.build_series_images", side_effect=_fake_build):
        out = select_series(manifest, slices=4, one_per_label=False, skip_qc_warn=False)
    assert len(out) == 1


def test_one_per_label_keeps_only_the_best_ranked_candidate():
    manifest = {"series": [_row("Seri1", "T2", snr=5.0), _row("Seri2", "T2", snr=50.0)]}
    with patch("mri_read.analyze.select.build_series_images", side_effect=_fake_build):
        out = select_series(manifest, slices=4, one_per_label=True, skip_qc_warn=False)
    assert len(out) == 1
    assert out[0].series == "Seri2"  # higher SNR wins


def test_one_per_label_false_keeps_every_candidate():
    manifest = {"series": [_row("Seri1", "T2"), _row("Seri2", "T2")]}
    with patch("mri_read.analyze.select.build_series_images", side_effect=_fake_build):
        out = select_series(manifest, slices=4, one_per_label=False, skip_qc_warn=False)
    assert len(out) == 2
