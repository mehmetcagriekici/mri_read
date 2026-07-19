"""Persisting an AnalysisResult as output/report.json."""

from __future__ import annotations

import json

from mri_read.paths import OUT


def write_json(result, study_meta: dict) -> None:
    (OUT / "report.json").write_text(json.dumps({
        "engine": result.engine,
        "study": study_meta,
        "sequences_reviewed": result.sequences_reviewed,
        "observations": result.observations,
        "impression": result.impression,
        "flags": result.flags,
        "disclaimer": result.disclaimer,
    }, indent=2))
