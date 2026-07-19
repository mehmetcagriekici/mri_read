from unittest.mock import patch

import pytest

import mri_read.dwi.bucketing as bucketing_module
from mri_read.dwi.bucketing import (_bucket_key, count_bvalue_buckets,
                                    count_bvalue_buckets_from_values)


def test_bucket_key_rounds_to_nearest_ten():
    assert _bucket_key(999.7) == _bucket_key(1000.2) == 1000.0


def test_bucket_key_none_stays_none():
    assert _bucket_key(None) is None


def test_count_buckets_collapses_close_values():
    assert count_bvalue_buckets_from_values([0.0, 0.3, 999.7, 1000.2]) == 2


def test_count_buckets_untagged_slices_count_as_their_own_bucket():
    assert count_bvalue_buckets_from_values([1000.0, None, None]) == 2


def test_count_buckets_empty_list():
    assert count_bvalue_buckets_from_values([]) == 0


def _make_dir(tmp_path, name, n_files):
    folder = tmp_path / name
    folder.mkdir()
    for i in range(n_files):
        (folder / f"s{i}.dcm").write_bytes(b"")
    return folder


def test_count_bvalue_buckets_reads_headers_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_dir(tmp_path, "SeriDWI", 3)
    with patch.object(bucketing_module.pydicom, "dcmread") as mock_read, \
         patch.object(bucketing_module, "read_bvalue",
                      side_effect=[0.0, 1000.0, 1000.0]):
        result = count_bvalue_buckets("SeriDWI")

    assert result == 2
    assert mock_read.call_count == 3


def test_count_bvalue_buckets_skips_unreadable_files(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    _make_dir(tmp_path, "SeriBad", 2)
    with patch.object(bucketing_module.pydicom, "dcmread",
                      side_effect=[ValueError("bad"), object()]), \
         patch.object(bucketing_module, "read_bvalue", return_value=0.0):
        result = count_bvalue_buckets("SeriBad")

    assert result == 1  # only the one readable file counted


def test_count_bvalue_buckets_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("mri_read.paths.locations.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="escapes the data directory"):
        count_bvalue_buckets("../../../../etc")
