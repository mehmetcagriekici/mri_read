"""
CLI entry point for Step 1 — explore the MRI DICOM data.

All the real logic lives in mri_read.explore; this just prints the report.

Run:  python src/cmd/explore.py

Output is a printed report; nothing is written to disk.
"""

from __future__ import annotations

from mri_read.explore import (DATA_DIR, find_series, read_header,
                              series_sort_key, summarize_series, tag)


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data folder not found: {DATA_DIR}")

    series = find_series()
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

    for name in sorted(series, key=series_sort_key):
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
