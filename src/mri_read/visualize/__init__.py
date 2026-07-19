"""
Step 2 — Load and visualize.

Turns series into images we can actually look at — the fastest way to sanity
check what each sequence is and confirm the loader works. Uses mri.load_series
for the heavy lifting and PIL to write PNGs.

Two outputs, written to output/:
  1. Overview: one montage PNG per series (a grid of evenly spaced slices) plus
     a printed tag table (plane, TE/TR, thickness, protocol) to identify every
     sequence at a glance.
  2. Deep-dive: every slice of a series exported as individual PNGs.

Layout:
  montage.py   : montage(), one grid image from a volume's slices.
  overview.py  : overview(), one montage per series + printed tag table.
  deep_dive.py : deep_dive()/deep_dive_all(), full per-slice PNG export.

CLI entry point: src/cmd/visualize.py

Note: this uses PER-SLICE windowing (window_to_uint8) — fine for eyeballing, but
the analysis path uses volume-level windowing instead (see analyze/).
"""

from mri_read.visualize.deep_dive import deep_dive, deep_dive_all
from mri_read.visualize.montage import montage
from mri_read.visualize.overview import overview

__all__ = ["montage", "overview", "deep_dive", "deep_dive_all"]
