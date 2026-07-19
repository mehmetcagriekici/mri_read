from unittest.mock import patch

import numpy as np
import pytest

import mri_read.dwi.loading as loading_module
from mri_read.dwi.loading import load_by_bvalue


class _FakeDataset:
    def __init__(self, pixels, position, bvalue):
        self._pixels = pixels
        self._bvalue = bvalue
        self.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        self.ImagePositionPatient = [0, 0, position]
        self.InstanceNumber = position

    @property
    def pixel_array(self):
        return self._pixels


def _make_dir(tmp_path, name, n_files):
    folder = tmp_path / name
    folder.mkdir()
    for i in range(n_files):
        (folder / f"s{i}.dcm").write_bytes(b"")
    return folder


def test_splits_slices_into_one_volume_per_bvalue(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_dir(tmp_path, "SeriDWI", 4)
    datasets = [
        _FakeDataset(np.zeros((4, 4), dtype=np.float32), position=0, bvalue=0.0),
        _FakeDataset(np.ones((4, 4), dtype=np.float32), position=1, bvalue=1000.0),
        _FakeDataset(np.zeros((4, 4), dtype=np.float32), position=1, bvalue=0.0),
        _FakeDataset(np.ones((4, 4), dtype=np.float32), position=0, bvalue=1000.0),
    ]
    with patch.object(loading_module.pydicom, "dcmread", side_effect=datasets), \
         patch.object(loading_module, "read_bvalue", side_effect=lambda ds: ds._bvalue):
        by_b = load_by_bvalue("SeriDWI")

    assert set(by_b.keys()) == {0.0, 1000.0}
    assert by_b[0.0].shape == (2, 4, 4)
    assert by_b[1000.0].shape == (2, 4, 4)


def test_untagged_slices_bucket_under_minus_one(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_dir(tmp_path, "SeriNoB", 1)
    ds = _FakeDataset(np.zeros((4, 4), dtype=np.float32), position=0, bvalue=None)
    with patch.object(loading_module.pydicom, "dcmread", return_value=ds), \
         patch.object(loading_module, "read_bvalue", return_value=None):
        by_b = load_by_bvalue("SeriNoB")

    assert list(by_b.keys()) == [-1.0]


def test_unreadable_files_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_dir(tmp_path, "SeriBad", 2)
    good = _FakeDataset(np.zeros((4, 4), dtype=np.float32), position=0, bvalue=0.0)
    with patch.object(loading_module.pydicom, "dcmread", side_effect=[good, ValueError("bad")]), \
         patch.object(loading_module, "read_bvalue", return_value=0.0):
        by_b = load_by_bvalue("SeriBad")

    assert by_b[0.0].shape == (1, 4, 4)


def test_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="escapes the data directory"):
        load_by_bvalue("../../../../etc")
