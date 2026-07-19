"""
Step 3 — Sequence classifier + study manifest.

This is the "understanding" layer. The anonymized data has blank
SeriesDescriptions, so we can't just read the sequence name — we INFER it from
MR physics and write a structured manifest.json that every later step consumes.

Each series gets {label, confidence, reason} plus a use_for_analysis flag so the
engine only looks at real diagnostic sequences (not reformats/localizers).

Layout:
  classify.py : classify(), the per-series rule-based classifier + PRIMARY set.
  build.py    : build_manifest(), walking every series into the full manifest.

CLI entry point: src/cmd/manifest.py
"""

from mri_read.manifest.build import build_manifest
from mri_read.manifest.classify import PRIMARY, classify

__all__ = ["classify", "PRIMARY", "build_manifest"]
