from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import mri_read.mri.loading as loading_module
from mri_read.mri.loading import inspect_series, load_series


class _FakeDataset:
    def __init__(self, pixels, position, rescale_slope=1, rescale_intercept=0):
        self._pixels = pixels
        self.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # axial
        self.ImagePositionPatient = [0, 0, position]
        self.InstanceNumber = position
        self.RescaleSlope = rescale_slope
        self.RescaleIntercept = rescale_intercept
        self.Modality = "MR"

    @property
    def pixel_array(self):
        return self._pixels


@pytest.fixture(autouse=True)
def _clear_cache():
    load_series.cache_clear()
    yield
    load_series.cache_clear()


def _make_series_dir(tmp_path, name, n_files):
    folder = tmp_path / name
    folder.mkdir()
    for i in range(n_files):
        (folder / f"slice{i}.dcm").write_bytes(b"")  # placeholder for glob()
    return folder


def test_load_series_sorts_by_geometric_position_and_stacks(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_series_dir(tmp_path, "SeriX", 3)

    # returned out of order on purpose -- load_series must re-sort by position
    datasets = [
        _FakeDataset(np.full((4, 4), 20, dtype=np.float32), position=20),
        _FakeDataset(np.full((4, 4), 0, dtype=np.float32), position=0),
        _FakeDataset(np.full((4, 4), 10, dtype=np.float32), position=10),
    ]
    with patch.object(loading_module.pydicom, "dcmread", side_effect=datasets):
        s = load_series("SeriX")

    assert s.n_slices == 3
    assert s.volume[0, 0, 0] == 0
    assert s.volume[1, 0, 0] == 10
    assert s.volume[2, 0, 0] == 20


def test_load_series_applies_rescale_slope_and_intercept(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_series_dir(tmp_path, "SeriY", 1)
    ds = _FakeDataset(np.full((2, 2), 100, dtype=np.float32), position=0,
                      rescale_slope=2, rescale_intercept=-50)
    with patch.object(loading_module.pydicom, "dcmread", return_value=ds):
        s = load_series("SeriY")

    assert np.all(s.volume == 100 * 2 - 50)


def test_load_series_drops_stray_odd_sized_frames(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_series_dir(tmp_path, "SeriZ", 3)
    datasets = [
        _FakeDataset(np.zeros((4, 4), dtype=np.float32), position=0),
        _FakeDataset(np.zeros((4, 4), dtype=np.float32), position=1),
        _FakeDataset(np.zeros((2, 2), dtype=np.float32), position=2),  # odd one out
    ]
    with patch.object(loading_module.pydicom, "dcmread", side_effect=datasets):
        s = load_series("SeriZ")

    assert s.n_slices == 2  # the mismatched-shape frame was dropped


def test_load_series_raises_on_empty_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    (tmp_path / "Empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_series("Empty")


def test_load_series_is_cached_per_name(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_series_dir(tmp_path, "SeriCached", 1)
    ds = _FakeDataset(np.zeros((2, 2), dtype=np.float32), position=0)
    with patch.object(loading_module.pydicom, "dcmread", return_value=ds) as mock_read:
        load_series("SeriCached")
        load_series("SeriCached")

    assert mock_read.call_count == 1  # second call served from the lru_cache


def test_inspect_series_is_header_only_and_counts_files(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_series_dir(tmp_path, "SeriHdr", 5)
    ds = SimpleNamespace(Modality="MR")
    with patch.object(loading_module.pydicom, "dcmread", return_value=ds):
        info = inspect_series("SeriHdr")

    assert info["n_slices"] == 5
    assert info["tags"]["modality"] == "MR"


def test_inspect_series_returns_empty_when_all_files_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_series_dir(tmp_path, "SeriBad", 2)
    with patch.object(loading_module.pydicom, "dcmread", side_effect=ValueError("bad file")):
        info = inspect_series("SeriBad")

    assert info == {"name": "SeriBad", "n_slices": 0, "tags": {}}


def test_load_series_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="escapes the data directory"):
        load_series("../../../../etc")


def test_inspect_series_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="escapes the data directory"):
        inspect_series("../../../../etc")
