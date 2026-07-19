"""
Diffusion (DWI) handling.

A DWI series often stacks multiple b-values together (e.g. b0 + b1000). The
diffusion signal lives in the CONTRAST between them, not in any single stack:

  - high-b images show restricted diffusion as bright (e.g. acute stroke),
  - the ADC map, computed from two b-values, separates true restriction from
    "T2 shine-through".

    ADC = -1/(b_high - b_low) * ln(S_high / S_low)

This package splits a DWI series by b-value and computes an ADC map when two
b-values are available. Falls back gracefully when b-values aren't tagged.

Layout:
  bucketing.py : counting distinct b-value groupings (series selection).
  loading.py   : load_by_bvalue, splitting a series folder by b-value.
  adc.py       : compute_adc, the ADC map formula.
  views.py     : diffusion_views, what analysis should look at.

Usage (standalone check):
  python src/cmd/dwi.py Seri1
"""

from mri_read.dwi.adc import compute_adc
from mri_read.dwi.bucketing import (count_bvalue_buckets,
                                    count_bvalue_buckets_from_values)
from mri_read.dwi.loading import load_by_bvalue
from mri_read.dwi.views import diffusion_views

__all__ = [
    "count_bvalue_buckets_from_values", "count_bvalue_buckets",
    "load_by_bvalue", "compute_adc", "diffusion_views",
]
