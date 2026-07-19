"""Collapsing one series' headers into a single summary row."""

from __future__ import annotations

from pathlib import Path

from mri_read.explore.headers import read_header, tag


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
