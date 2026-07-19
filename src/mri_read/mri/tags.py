"""Normalizing a DICOM dataset's acquisition/geometry tags into a plain dict."""

from __future__ import annotations

from mri_read.mri.geometry import plane_from_orientation


def extract_tags(ds) -> dict:
    """Copy the handful of DICOM tags we use into a plain, JSON-friendly dict.

    Using getattr with defaults means a missing tag yields "—" or None instead
    of raising — important because this anonymized data has several blank tags.
    """
    def num(v):
        """Coerce a DICOM value to float, or None if it isn't numeric."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "modality": str(getattr(ds, "Modality", "—")),            # "MR"
        "plane": plane_from_orientation(ds),                       # Axial/...
        "scanning_sequence": str(getattr(ds, "ScanningSequence", "—")),  # SE/IR/GR/EP
        "sequence_variant": str(getattr(ds, "SequenceVariant", "—")),
        "protocol": str(getattr(ds, "ProtocolName", "—")),
        "series_number": getattr(ds, "SeriesNumber", None),
        "echo_time_TE": num(getattr(ds, "EchoTime", None)),        # TE (ms)
        "repetition_TR": num(getattr(ds, "RepetitionTime", None)), # TR (ms)
        "inversion_TI": num(getattr(ds, "InversionTime", None)),   # TI (ms)
        "body_part": str(getattr(ds, "BodyPartExamined", "—")),
        "thickness_mm": num(getattr(ds, "SliceThickness", None)),
        "manufacturer": str(getattr(ds, "Manufacturer", "—")),
        "model": str(getattr(ds, "ManufacturerModelName", "—")),
        "field_T": num(getattr(ds, "MagneticFieldStrength", None)),  # e.g. 3.0
    }
