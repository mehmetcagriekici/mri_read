import pytest

from mri_read.paths import DATA_DIR, OUT, ROOT, series_dir


def test_root_is_the_repo_root():
    assert (ROOT / "pyproject.toml").exists()
    assert (ROOT / "src" / "mri_read").is_dir()


def test_data_dir_and_out_are_under_root():
    assert DATA_DIR == ROOT / "mri_test_data"
    assert OUT == ROOT / "output"


class TestSeriesDirSecurity:
    """series_dir() is the single choke point for turning a series name into
    a filesystem path -- see mri.loading, dwi.loading, dwi.bucketing,
    qc.header_metrics, all of which join DATA_DIR with a `name` that isn't
    always guaranteed to come from list_series()'s own safe enumeration
    (cmd/dwi.py passes a raw CLI argument straight through). Without this
    check, a name like "../../../../etc" resolves clean outside DATA_DIR via
    plain Path division -- Path doesn't sanitize ".." components.
    """

    def test_plain_name_resolves_under_data_dir(self):
        assert series_dir("Seri1") == (DATA_DIR / "Seri1").resolve()

    def test_relative_traversal_is_rejected(self):
        with pytest.raises(ValueError, match="escapes the data directory"):
            series_dir("../../../../etc")

    def test_traversal_hidden_in_the_middle_is_rejected(self):
        with pytest.raises(ValueError, match="escapes the data directory"):
            series_dir("Seri1/../../../../etc")

    def test_absolute_path_escape_is_rejected(self):
        with pytest.raises(ValueError, match="escapes the data directory"):
            series_dir("/etc/passwd")

    def test_a_traversal_that_stays_within_data_dir_is_allowed(self):
        # "Seri1/../Seri2" never actually leaves DATA_DIR -- the check is
        # about the RESOLVED destination, not merely the presence of "..".
        assert series_dir("Seri1/../Seri2") == (DATA_DIR / "Seri2").resolve()
