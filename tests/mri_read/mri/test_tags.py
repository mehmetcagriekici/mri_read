from types import SimpleNamespace

from mri_read.mri.tags import extract_tags


def test_extract_tags_reads_known_fields():
    ds = SimpleNamespace(
        Modality="MR", ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
        ScanningSequence="SE", SequenceVariant="SS", ProtocolName="T2 AX",
        SeriesNumber=6, EchoTime="96.36", RepetitionTime="7316.0",
        InversionTime="2000.0", BodyPartExamined="BRAIN", SliceThickness="5.0",
        Manufacturer="GE MEDICAL SYSTEMS", ManufacturerModelName="SIGNA Pioneer",
        MagneticFieldStrength="3.0",
    )
    tags = extract_tags(ds)
    assert tags["modality"] == "MR"
    assert tags["plane"] == "Axial"
    assert tags["echo_time_TE"] == 96.36
    assert tags["repetition_TR"] == 7316.0
    assert tags["thickness_mm"] == 5.0
    assert tags["field_T"] == 3.0
    assert tags["protocol"] == "T2 AX"


def test_extract_tags_missing_fields_use_friendly_defaults():
    ds = SimpleNamespace()  # nothing set
    tags = extract_tags(ds)
    assert tags["modality"] == "—"
    assert tags["plane"] == "unknown"
    assert tags["echo_time_TE"] is None
    assert tags["series_number"] is None


def test_extract_tags_non_numeric_value_becomes_none():
    ds = SimpleNamespace(EchoTime="not-a-number")
    tags = extract_tags(ds)
    assert tags["echo_time_TE"] is None


def test_extract_tags_handles_unicode_string_fields():
    ds = SimpleNamespace(ProtocolName="Étude cérébrale — T2*  (日本語)",
                         Manufacturer="Ünïcode™ Devices")
    tags = extract_tags(ds)
    assert tags["protocol"] == "Étude cérébrale — T2*  (日本語)"
    assert tags["manufacturer"] == "Ünïcode™ Devices"


def test_extract_tags_handles_nan_and_inf_numeric_values():
    ds = SimpleNamespace(EchoTime=float("nan"), SliceThickness=float("inf"))
    tags = extract_tags(ds)
    assert tags["echo_time_TE"] != tags["echo_time_TE"]  # NaN != itself
    assert tags["thickness_mm"] == float("inf")


def test_extract_tags_handles_negative_thickness():
    # Not physically meaningful, but shouldn't crash -- some anonymization
    # or export pipelines have been known to zero/sign-flip odd tags.
    ds = SimpleNamespace(SliceThickness=-5.0)
    tags = extract_tags(ds)
    assert tags["thickness_mm"] == -5.0
