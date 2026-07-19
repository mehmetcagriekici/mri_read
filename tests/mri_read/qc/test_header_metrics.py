from unittest.mock import patch

import pytest

import mri_read.qc.header_metrics as header_metrics_module
from mri_read.qc.header_metrics import _positions_and_instances


class _FakeDataset:
    def __init__(self, position, instance_number, bvalue):
        self.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        self.ImagePositionPatient = [0, 0, position]
        self.InstanceNumber = instance_number
        self._bvalue = bvalue


def _make_dir(tmp_path, name, n_files):
    folder = tmp_path / name
    folder.mkdir()
    for i in range(n_files):
        (folder / f"s{i}.dcm").write_bytes(b"")
    return folder


def test_returns_sorted_positions_instances_and_bvalues(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_dir(tmp_path, "Seri1", 3)
    # returned out of geometric order on purpose
    datasets = [
        _FakeDataset(position=20, instance_number=3, bvalue=1000.0),
        _FakeDataset(position=0, instance_number=1, bvalue=0.0),
        _FakeDataset(position=10, instance_number=2, bvalue=None),
    ]
    with patch.object(header_metrics_module.pydicom, "dcmread", side_effect=datasets), \
         patch.object(header_metrics_module, "read_bvalue", side_effect=lambda ds: ds._bvalue):
        positions, instances, bvalues = _positions_and_instances("Seri1")

    assert positions == [0.0, 10.0, 20.0]
    assert instances == [1, 2, 3]
    assert bvalues == [0.0, None, 1000.0]


def test_skips_unreadable_files(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_dir(tmp_path, "SeriBad", 2)
    good = _FakeDataset(position=0, instance_number=1, bvalue=0.0)
    with patch.object(header_metrics_module.pydicom, "dcmread",
                      side_effect=[good, ValueError("bad")]), \
         patch.object(header_metrics_module, "read_bvalue", return_value=0.0):
        positions, instances, bvalues = _positions_and_instances("SeriBad")

    assert positions == [0.0]
    assert instances == [1]


def test_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="escapes the data directory"):
        _positions_and_instances("../../../../etc")
