"""Regression guard for run_qc()'s wall-clock cost on real data.

Profiling found run_qc() on a 150+-slice 3D T1 series costs ~4-5s on this
machine, split roughly evenly between DICOM pixel decode (load_series,
expected/unavoidable) and volume_window_bounds()'s exact np.percentile over
the full-resolution foreground (the low-contrast check) -- np.partition alone
accounted for ~2.9s of a 16-series qc.py run in profiling. That's real cost
paid for EVERY use_for_analysis candidate in run_agent()'s manifest+QC loop,
including candidates that lose the later ranking (e.g. both 3D T1 series get
fully QC'd even though only the thinner one is ever sent to the vision
model). This isn't asserting a target to hit -- it pins current behavior at a
generous multiple so an accidental regression (e.g. losing the lru_cache, an
accidental O(n^2) pass) gets caught, without failing on ordinary machine
variance.
"""

from __future__ import annotations

import time

import pytest

from mri_read.qc import run_qc

pytestmark = pytest.mark.data


def test_run_qc_on_a_large_3d_series_completes_within_a_generous_bound():
    # Seri7/Seri11 in mri_test_data/ are ~150-slice 3D T1 volumes -- the
    # single most expensive per-series QC case in this dataset.
    for name in ("Seri7", "Seri11"):
        t0 = time.time()
        run_qc(name)
        elapsed = time.time() - t0
        assert elapsed < 20, (
            f"run_qc({name!r}) took {elapsed:.1f}s, expected well under 20s -- "
            "possible perf regression (lost caching, accidental O(n^2), etc.)"
        )
