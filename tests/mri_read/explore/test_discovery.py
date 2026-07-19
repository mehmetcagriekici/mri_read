from mri_read.explore.discovery import find_series, series_sort_key


def test_find_series_groups_by_parent_folder(tmp_path):
    (tmp_path / "Seri1").mkdir()
    (tmp_path / "Seri2").mkdir()
    (tmp_path / "Seri1" / "a.dcm").write_bytes(b"")
    (tmp_path / "Seri1" / "b.dcm").write_bytes(b"")
    (tmp_path / "Seri2" / "c.dcm").write_bytes(b"")

    result = find_series(tmp_path)

    assert set(result.keys()) == {"Seri1", "Seri2"}
    assert len(result["Seri1"]) == 2
    assert len(result["Seri2"]) == 1


def test_find_series_ignores_non_dcm_files(tmp_path):
    (tmp_path / "Seri1").mkdir()
    (tmp_path / "Seri1" / "a.dcm").write_bytes(b"")
    (tmp_path / "Seri1" / "readme.txt").write_bytes(b"")

    result = find_series(tmp_path)
    assert len(result["Seri1"]) == 1


def test_series_sort_key_numeric_ordering():
    names = ["Seri10", "Seri2", "Seri1"]
    assert sorted(names, key=series_sort_key) == ["Seri1", "Seri2", "Seri10"]


def test_series_sort_key_no_digits_sorts_last():
    assert series_sort_key("NoDigits") > series_sort_key("Seri999")
