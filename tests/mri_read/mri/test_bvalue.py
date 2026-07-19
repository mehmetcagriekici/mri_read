from mri_read.mri.bvalue import read_bvalue


class _FakeElem:
    def __init__(self, value):
        self.value = value


class _FakeDataset(dict):
    """Minimal stand-in for a pydicom Dataset: `(group, elem) in ds` + ds[group, elem].value."""
    def __getitem__(self, key):
        return _FakeElem(super().__getitem__(key))


def test_reads_standard_tag():
    ds = _FakeDataset({(0x0018, 0x9087): 1000.0})
    assert read_bvalue(ds) == 1000.0


def test_falls_back_to_ge_private_tag():
    ds = _FakeDataset({(0x0043, 0x1039): 800.0})
    assert read_bvalue(ds) == 800.0


def test_unpacks_ge_packed_bvalue():
    ds = _FakeDataset({(0x0043, 0x1039): 1000001000})
    assert read_bvalue(ds) == 1000.0


def test_multi_valued_tag_uses_first_entry():
    ds = _FakeDataset({(0x0018, 0x9087): [1000.0, 0.0]})
    assert read_bvalue(ds) == 1000.0


def test_no_tag_present_returns_none():
    ds = _FakeDataset({})
    assert read_bvalue(ds) is None
