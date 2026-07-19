"""Ranking candidate series so select_series() can pick the BEST per label."""

from __future__ import annotations

from mri_read.dwi import count_bvalue_buckets


def _rank_key(row: dict) -> tuple:
    """Higher is better. QC status always leads, then a label-specific tie-break.

    - status first, always: a series whose pixel load already failed QC
      ("error") or that's flagged ("warn") never outranks a clean one, no
      matter how it scores on the label-specific criteria below.
    - DWI: more distinct b-value buckets (needed for an ADC map — reused from
      QC's own header pass when available, see count_bvalue_buckets_from_values
      in dwi.py, falling back to a fresh header scan otherwise), then more slices.
    - 3D T1: thinner slices (more volumetric detail).
    - Everything else: higher SNR: an unmeasurable SNR (None — e.g. a flat,
      hard-masked FOV; see qc.py's _background_snr) is treated as a borderline
      PASS (the qc.py low-snr threshold) rather than as the worst possible
      score, since "unmeasured" isn't evidence of a noisy scan.
    Used to pick the single BEST row per label when several series share one
    sequence type (e.g. Seri1/Seri8/Seri9 all classified DWI).
    """
    qc = row.get("qc") or {}
    status_rank = {"pass": 2, "warn": 1, "error": 0}.get(qc.get("status"), 0)
    metrics = qc.get("metrics") or {}
    snr = metrics.get("snr")
    if snr is None:
        snr = 8.0                                      # qc.py's low-snr pass/warn threshold

    if row["label"] == "DWI":
        buckets = metrics.get("bvalue_buckets")
        if buckets is None:                            # QC didn't run on this row
            buckets = count_bvalue_buckets(row["series"])
        return (status_rank, buckets, row.get("n_slices", 0))
    if row["label"] == "3D T1":
        thickness = (row.get("acq") or {}).get("thickness_mm")
        return (status_rank, -thickness if thickness is not None else float("-inf"))
    return (status_rank, snr)
