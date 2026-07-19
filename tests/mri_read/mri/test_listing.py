import pytest

import mri_read.mri.listing as listing_module
from mri_read.mri.listing import list_series


def test_list_series_sorts_numerically_not_lexically(tmp_path, monkeypatch):
    monkeypatch.setattr(listing_module, "DATA_DIR", tmp_path)
    for name in ("Seri10", "Seri2", "Seri1"):
        (tmp_path / name).mkdir()

    assert list_series() == ["Seri1", "Seri2", "Seri10"]


def test_list_series_ignores_files_only_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(listing_module, "DATA_DIR", tmp_path)
    (tmp_path / "Seri1").mkdir()
    (tmp_path / "not_a_series.txt").write_text("x")

    assert list_series() == ["Seri1"]


def test_list_series_names_without_digits_sort_last(tmp_path, monkeypatch):
    monkeypatch.setattr(listing_module, "DATA_DIR", tmp_path)
    (tmp_path / "Seri1").mkdir()
    (tmp_path / "NoDigits").mkdir()

    assert list_series() == ["Seri1", "NoDigits"]


def test_list_series_raises_clear_error_when_data_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(listing_module, "DATA_DIR", tmp_path / "does-not-exist")
    with pytest.raises(FileNotFoundError, match="No data folder"):
        list_series()
