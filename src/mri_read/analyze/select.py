"""Selecting the primary series from a manifest and building engine input."""

from __future__ import annotations

import logging

from mri_read.analyze.images import build_series_images
from mri_read.analyze.ranking import _rank_key

logger = logging.getLogger(__name__)


def select_series(manifest: dict, slices: int, one_per_label: bool,
                  skip_qc_warn: bool):
    """Build SeriesImages for the primary series in the manifest.

    one_per_label picks the single BEST candidate per sequence type via
    _rank_key (not just the first one encountered in manifest order).
    """
    candidates = []
    for row in manifest["series"]:
        if not row.get("use_for_analysis"):
            continue
        qc = row.get("qc", {})
        if skip_qc_warn and qc.get("status") == "warn":
            logger.info("skipping %s (%s) — QC: %s",
                       row["series"], row["label"], ", ".join(qc.get("flags", [])))
            continue
        candidates.append(row)

    if one_per_label:
        best: dict[str, dict] = {}
        best_rank: dict[str, tuple] = {}
        for row in candidates:
            label = row["label"]
            rank = _rank_key(row)                       # computed once per row, not per comparison
            if label not in best or rank > best_rank[label]:
                best[label] = row
                best_rank[label] = rank
        candidates = list(best.values())

    out = []
    for row in candidates:
        out.extend(build_series_images(row, slices))
    return out
