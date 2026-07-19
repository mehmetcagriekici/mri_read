from types import SimpleNamespace
from unittest.mock import patch

from mri_read.explore.headers import read_header, tag


def test_read_header_returns_dataset_on_success(tmp_path):
    f = tmp_path / "slice.dcm"
    f.write_bytes(b"")
    fake_ds = SimpleNamespace(Modality="MR")
    with patch("mri_read.explore.headers.pydicom.dcmread", return_value=fake_ds):
        assert read_header(f) is fake_ds


def test_read_header_returns_none_on_any_error(tmp_path):
    f = tmp_path / "bad.dcm"
    f.write_bytes(b"")
    with patch("mri_read.explore.headers.pydicom.dcmread", side_effect=ValueError("bad")):
        assert read_header(f) is None


def test_tag_returns_value_when_present():
    ds = SimpleNamespace(Modality="MR")
    assert tag(ds, "Modality") == "MR"


def test_tag_returns_default_when_missing():
    ds = SimpleNamespace()
    assert tag(ds, "Modality") == "—"
    assert tag(ds, "Modality", default="?") == "?"
