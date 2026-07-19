"""Assembling the full study manifest by walking every series."""

from __future__ import annotations

from mri_read.manifest.classify import PRIMARY, classify
from mri_read.mri import inspect_series, list_series


def build_manifest() -> dict:
    """Walk every series, classify it, and assemble the full manifest dict.

    Study-level fields (body part, scanner) are captured once from the first
    series that has tags. Each series row carries its label, why, and the raw
    acquisition numbers so nothing downstream has to re-read DICOM.
    """
    series_out = []
    study = {}
    for name in list_series():
        info = inspect_series(name)                  # header-only (fast)
        tags = info["tags"]
        if not study and tags:                       # capture study info once
            study = {
                "body_part": tags.get("body_part"),
                "manufacturer": tags.get("manufacturer"),
                "model": tags.get("model"),
                "field_T": tags.get("field_T"),
            }
        cls = classify(tags, info["n_slices"])
        series_out.append({
            "series": name,
            "n_slices": info["n_slices"],
            "plane": tags.get("plane"),
            "label": cls["label"],
            "confidence": cls["confidence"],
            "reason": cls["reason"],
            "use_for_analysis": cls["label"] in PRIMARY,   # engine filter
            "acq": {                                        # raw numbers kept for audit

                "scanning_sequence": tags.get("scanning_sequence"),
                "TE": tags.get("echo_time_TE"),
                "TR": tags.get("repetition_TR"),
                "TI": tags.get("inversion_TI"),
                "thickness_mm": tags.get("thickness_mm"),
            },
        })
    return {"study": study, "series": series_out}
