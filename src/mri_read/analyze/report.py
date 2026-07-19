"""write_report() — persisting an AnalysisResult in every output format."""

from __future__ import annotations

from mri_read.analyze.report_json import write_json
from mri_read.analyze.report_markdown import write_markdown


def write_report(result, study_meta: dict) -> None:
    write_json(result, study_meta)
    write_markdown(result, study_meta)
