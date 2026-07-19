"""Shared filesystem locations, computed once relative to the repo root.

Every module used to recompute this itself via `Path(__file__).resolve()...`,
which broke when files moved a directory deeper into mri_read/. One shared
computation here means only this file needs updating if the layout moves again.
"""

from pathlib import Path

# mri_read/paths/locations.py -> paths/ -> mri_read/ -> src/ -> repo root
ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = ROOT / "mri_test_data"
OUT = ROOT / "output"


def series_dir(name: str) -> Path:
    """Resolve a series name to its folder under DATA_DIR, refusing to escape it.

    `name` is normally an enumerated folder name from mri.list_series(), which
    is always safe. But some callers take a series name directly from a less
    trusted source -- most notably `cmd/dwi.py <name>`, a CLI argument -- and
    a plain `DATA_DIR / name` join doesn't stop `name` containing `..`
    components from resolving outside DATA_DIR entirely (e.g.
    "../../../../etc" resolves to "/etc"). This is the single choke point
    every DATA_DIR-joining module should go through instead of joining paths
    inline.
    """
    folder = (DATA_DIR / name).resolve()
    if not folder.is_relative_to(DATA_DIR.resolve()):
        raise ValueError(f"Invalid series name {name!r}: escapes the data directory")
    return folder
