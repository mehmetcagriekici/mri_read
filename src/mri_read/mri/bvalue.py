"""Best-effort diffusion b-value extraction from a DICOM header."""

from __future__ import annotations

# Diffusion b-value (the DWI "diffusion weighting strength") is stored in the
# standard tag on modern scanners, but older GE data hides it in private tags.
# We try the standard one first, then the two common GE locations.
_BVALUE_TAGS = [
    (0x0018, 0x9087),   # DiffusionBValue (standard)
    (0x0043, 0x1039),   # GE private
    (0x0019, 0x100C),   # GE private (alternate)
]


def read_bvalue(ds) -> float | None:
    """Best-effort diffusion b-value for one slice, or None if not tagged.

    Checks the standard tag then GE private tags. GE occasionally packs the
    b-value into the low 5 digits of a larger slop integer (e.g. 1000001000
    encodes b=1000), so we unpack values above 100000 with a modulo.
    """
    for group, elem in _BVALUE_TAGS:
        if (group, elem) in ds:                      # tag present in this file?
            try:
                val = ds[group, elem].value
                if isinstance(val, (list, tuple)):   # sometimes multi-valued
                    val = val[0]
                b = float(val)
                if b > 100000:                       # unpack GE's packed form
                    b = b % 100000
                return b
            except (ValueError, TypeError):
                continue                             # try the next candidate tag
    return None
