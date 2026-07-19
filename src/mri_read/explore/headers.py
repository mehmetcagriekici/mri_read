"""Reading single DICOM headers, forgivingly."""

from __future__ import annotations

from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError


def read_header(path: Path):
    """Read one DICOM header (no pixel data), or None if the file won't parse.

    stop_before_pixels=True skips decoding the image = fast. force=True lets us
    read files that lack the formal DICOM preamble. We swallow all errors and
    return None so one bad file never stops the whole exploration.
    """
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except (InvalidDicomError, Exception):  # noqa: BLE001 - explore, be forgiving
        return None


def tag(ds, name, default="—"):
    """getattr with a friendly default, so a missing tag prints "—" not a crash."""
    return getattr(ds, name, default)
