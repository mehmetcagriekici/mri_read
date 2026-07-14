"""
Step 1 — Explore the MRI DICOM data.

The very first script we wrote. Its only job is to answer "what is in this
dataset?" before any analysis exists: how many series, what modality/sequence,
image size, slice counts, and whether anything looks broken. It reads HEADERS
ONLY (no pixels), so it's fast and safe to run repeatedly.

This is intentionally standalone — it predates mri.py and does its own light
reading — so you can run it on a fresh copy of the data with nothing else set up.

Run:  python src/explore.py

Output is a printed report; nothing is written to disk.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

# mri_test_data lives next to this repo's root (../mri_test_data from src/)
DATA_DIR = Path(__file__).resolve().parent.parent / "mri_test_data"


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


def summarize_series(name: str, files: list[Path]) -> dict:
    """Read all slice headers in one series and collapse them to a summary row.

    Reads every file (not just the first) so we can (a) count how many are
    unreadable and (b) detect tags that unexpectedly VARY across the series
    (size/thickness are collected as sets — more than one value means the series
    is inhomogeneous).
    """
    headers = [(f, read_header(f)) for f in files]
    good = [(f, ds) for f, ds in headers if ds is not None]   # parsed OK
    unreadable = [f for f, ds in headers if ds is None]       # failed to parse

    if not good:
        return {"series": name, "slices": 0, "unreadable": len(unreadable)}

    first = good[0][1]     # representative slice for constant tags

    # Gather a tag across all slices as a SET, so a single value means "constant"
    # and multiple values reveal an inhomogeneous series.
    def collect(attr):
        vals = {str(tag(ds, attr)) for _, ds in good}
        return vals

    rows = collect("Rows")
    cols = collect("Columns")
    thicknesses = collect("SliceThickness")

    # Missing-slice heuristic: InstanceNumber is the scanner's slice index. If
    # the numbers run 1..N but we have fewer than N, some slices are missing.
    instance_numbers = sorted(
        int(tag(ds, "InstanceNumber", 0)) for _, ds in good
        if str(tag(ds, "InstanceNumber", "")).isdigit()
    )
    gaps = []
    if instance_numbers:
        # The complete run of indices we'd expect, minus the ones we actually
        # have = the missing slice numbers.
        full = set(range(instance_numbers[0], instance_numbers[-1] + 1))
        gaps = sorted(full - set(instance_numbers))

    return {
        "series": name,
        "desc": tag(first, "SeriesDescription"),
        "modality": tag(first, "Modality"),
        "size": f"{'/'.join(sorted(rows))} x {'/'.join(sorted(cols))}",
        "slices": len(good),
        "thickness_mm": "/".join(sorted(thicknesses)),
        "pixel_spacing": str(tag(first, "PixelSpacing")),
        "field_T": tag(first, "MagneticFieldStrength"),
        "manufacturer": tag(first, "Manufacturer"),
        "missing_instances": gaps,
        "unreadable": len(unreadable),
    }


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data folder not found: {DATA_DIR}")

    # Walk the whole tree and bucket every .dcm by the folder it lives in — that
    # folder name IS the series (Seri1, Seri2, ...).
    series: dict[str, list[Path]] = defaultdict(list)
    for f in DATA_DIR.rglob("*.dcm"):
        series[f.parent.name].append(f)

    print(f"Data root: {DATA_DIR}")
    print(f"Series found: {len(series)} | Total slices: "
          f"{sum(len(v) for v in series.values())}\n")

    # Print study/patient/scanner info once, from the first file that parses.
    # (These tags are study-wide, so any slice will do.) The for/else/break
    # dance stops as soon as we've printed one — outer `break` on success,
    # `else: continue` to try the next series only if none parsed.
    for files in series.values():
        for f in files:
            ds = read_header(f)
            if ds is not None:
                print(f"Study    : {tag(ds, 'StudyDescription')}")
                print(f"Patient  : sex={tag(ds, 'PatientSex')} "
                      f"age={tag(ds, 'PatientAge')}")
                print(f"Scanner  : {tag(ds, 'Manufacturer')} "
                      f"{tag(ds, 'ManufacturerModelName')}\n")
                break
        else:
            continue
        break

    # Numeric sort so Seri2 comes before Seri10 (string sort would not).
    def series_key(n: str):
        digits = "".join(c for c in n if c.isdigit())
        return int(digits) if digits else 10**9

    for name in sorted(series, key=series_key):
        s = summarize_series(name, series[name])
        print(f"── {s['series']} " + "─" * (40 - len(s['series'])))
        if s["slices"] == 0:
            print(f"   No readable slices ({s['unreadable']} unreadable)\n")
            continue
        print(f"   description : {s.get('desc')}")
        print(f"   modality    : {s.get('modality')}")
        print(f"   image size  : {s.get('size')}  ({s['slices']} slices)")
        print(f"   thickness   : {s.get('thickness_mm')} mm   "
              f"spacing: {s.get('pixel_spacing')}")
        print(f"   field / mfr : {s.get('field_T')} T   {s.get('manufacturer')}")
        if s["missing_instances"]:
            print(f"   ⚠ missing instance numbers: {s['missing_instances']}")
        if s["unreadable"]:
            print(f"   ⚠ unreadable files: {s['unreadable']}")
        print()


if __name__ == "__main__":
    main()
