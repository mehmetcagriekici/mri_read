from types import SimpleNamespace
from unittest.mock import patch

from mri_read.explore.summary import summarize_series


def _ds(**kwargs):
    return SimpleNamespace(**kwargs)


def test_summarize_series_collapses_constant_tags():
    files = [f"f{i}.dcm" for i in range(3)]
    datasets = [
        _ds(Rows=256, Columns=256, SliceThickness=5.0, InstanceNumber=i + 1,
           SeriesDescription="T2", Modality="MR", PixelSpacing="[1,1]",
           MagneticFieldStrength=3.0, Manufacturer="GE")
        for i in range(3)
    ]
    with patch("mri_read.explore.summary.read_header", side_effect=datasets):
        summary = summarize_series("Seri1", files)

    assert summary["slices"] == 3
    assert summary["size"] == "256 x 256"
    assert summary["thickness_mm"] == "5.0"
    assert summary["missing_instances"] == []
    assert summary["unreadable"] == 0


def test_summarize_series_detects_inhomogeneous_size():
    files = ["a.dcm", "b.dcm"]
    datasets = [
        _ds(Rows=256, Columns=256, SliceThickness=5.0, InstanceNumber=1),
        _ds(Rows=512, Columns=512, SliceThickness=5.0, InstanceNumber=2),
    ]
    with patch("mri_read.explore.summary.read_header", side_effect=datasets):
        summary = summarize_series("SeriMixed", files)

    assert "/" in summary["size"]  # both sizes shown, joined


def test_summarize_series_detects_missing_instance_numbers():
    files = ["a.dcm", "b.dcm"]
    datasets = [
        _ds(Rows=256, Columns=256, SliceThickness=5.0, InstanceNumber=1),
        _ds(Rows=256, Columns=256, SliceThickness=5.0, InstanceNumber=4),
    ]
    with patch("mri_read.explore.summary.read_header", side_effect=datasets):
        summary = summarize_series("SeriGap", files)

    assert summary["missing_instances"] == [2, 3]


def test_summarize_series_counts_unreadable_files():
    files = ["a.dcm", "b.dcm"]
    datasets = [_ds(Rows=256, Columns=256, SliceThickness=5.0, InstanceNumber=1), None]
    with patch("mri_read.explore.summary.read_header", side_effect=datasets):
        summary = summarize_series("SeriBad", files)

    assert summary["slices"] == 1
    assert summary["unreadable"] == 1


def test_summarize_series_all_unreadable_returns_minimal_row():
    files = ["a.dcm"]
    with patch("mri_read.explore.summary.read_header", return_value=None):
        summary = summarize_series("SeriDead", files)

    assert summary == {"series": "SeriDead", "slices": 0, "unreadable": 1}
