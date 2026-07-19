"""
Step 1 — Explore the MRI DICOM data.

The very first script we wrote. Its only job is to answer "what is in this
dataset?" before any analysis exists: how many series, what modality/sequence,
image size, slice counts, and whether anything looks broken. It reads HEADERS
ONLY (no pixels), so it's fast and safe to run repeatedly.

This package is intentionally standalone — it predates mri/ and does its own
light reading (no dependency on mri.load_series) — so you can run it on a
fresh copy of the data with nothing else set up.

Layout:
  headers.py   : read_header/tag, single-file header reading.
  discovery.py : find_series/series_sort_key, finding and ordering series
                 folders on disk.
  summary.py   : summarize_series, collapsing one series' headers into a
                 summary row.

CLI entry point: src/cmd/explore.py

Output is a printed report; nothing is written to disk.
"""

from mri_read.explore.discovery import find_series, series_sort_key
from mri_read.explore.headers import read_header, tag
from mri_read.explore.summary import summarize_series
from mri_read.paths import DATA_DIR

__all__ = ["DATA_DIR", "read_header", "tag", "find_series", "summarize_series",
          "series_sort_key"]
