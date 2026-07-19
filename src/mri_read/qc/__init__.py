"""
Step 3b — Deterministic quality control.

Before any analysis, flag series that are geometrically or visually suspect so
the engine (and you) know how much to trust them. No AI — plain measurements.

Checks per series:
  - slice count vs InstanceNumber range  -> missing slices
  - spacing between slices               -> uneven / irregular geometry
  - foreground contrast                  -> low-contrast (flat) volume
  - background noise vs signal (SNR)     -> noisy volume
  - fraction of near-empty slices        -> mostly-empty acquisition

Augments output/manifest.json with a "qc" block per series.

Layout:
  header_metrics.py : header-only geometry measurement (_positions_and_instances).
  signal_metrics.py : pixel-based noise measurement (_background_snr).
  checks.py         : run_qc, running every check and assembling flags/status.

CLI entry point: src/cmd/qc.py
"""

from mri_read.qc.checks import run_qc

__all__ = ["run_qc"]
