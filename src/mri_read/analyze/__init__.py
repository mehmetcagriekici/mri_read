"""
Step 4 — Orchestrator.

Reads output/manifest.json, selects representative slices from the primary
series, hands them to an engine, and writes a report. The engine is chosen by
name (--engine) so nothing here is tied to Claude specifically.

Layout:
  slice_selection.py : content_indices, picking which slice indices to use.
  png_encoding.py     : volume_to_pngs, windowing + PNG-encoding chosen slices.
  images.py           : build_series_images, per-manifest-row engine input
                        (DWI routed specially through dwi.diffusion_views).
  ranking.py          : _rank_key, picking the BEST candidate per sequence type.
  select.py           : select_series, the top-level selection entry point.
  report_json.py      : write_json, persisting output/report.json.
  report_markdown.py  : write_markdown, rendering output/report.md.
  report.py           : write_report, calling both writers.

CLI entry point: src/cmd/analyze.py
"""

from mri_read.analyze.images import build_series_images
from mri_read.analyze.png_encoding import volume_to_pngs
from mri_read.analyze.report import write_report
from mri_read.analyze.select import select_series
from mri_read.analyze.slice_selection import content_indices

__all__ = ["content_indices", "volume_to_pngs", "build_series_images",
          "select_series", "write_report"]
