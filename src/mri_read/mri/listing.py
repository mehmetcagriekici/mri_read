"""Enumerating series folders on disk (no DICOM parsing)."""

from __future__ import annotations

from mri_read.paths import DATA_DIR


def list_series() -> list[str]:
    """List series folder names sorted numerically (Seri2 before Seri10).

    Plain sorted() would order these as strings ("Seri10" < "Seri2"), so we key
    on the integer embedded in the name.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No data folder at {DATA_DIR}. Place your DICOM series folders "
            "there (see README.md 'Data') before running the pipeline."
        )
    def key(n: str):
        d = "".join(c for c in n if c.isdigit())     # pull digits out of the name
        return int(d) if d else 10**9                # names w/o digits sort last
    return sorted((p.name for p in DATA_DIR.iterdir() if p.is_dir()), key=key)
