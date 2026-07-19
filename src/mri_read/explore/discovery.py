"""Finding and ordering series folders on disk."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from mri_read.paths import DATA_DIR


def find_series(data_dir: Path = DATA_DIR) -> dict[str, list[Path]]:
    """Walk the data folder and bucket every .dcm by the folder it lives in.

    That folder name IS the series (Seri1, Seri2, ...).
    """
    series: dict[str, list[Path]] = defaultdict(list)
    for f in data_dir.rglob("*.dcm"):
        series[f.parent.name].append(f)
    return series


def series_sort_key(n: str) -> int:
    """Numeric sort so Seri2 comes before Seri10 (string sort would not)."""
    digits = "".join(c for c in n if c.isdigit())
    return int(digits) if digits else 10**9
