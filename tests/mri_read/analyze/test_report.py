import json

from mri_read.analyze.report import write_report
from mri_read.engine import AnalysisResult


def _result():
    return AnalysisResult(
        engine="ollama:llava", sequences_reviewed=["T2", "T2 FLAIR"],
        observations=[{"sequence": "T2", "finding": "normal", "location": "n/a",
                      "confidence": "high"}],
        impression="No acute findings.", confidence="high",
        flags=["low-snr on Seri9"],
        disclaimer="Research prototype only.",
    )


def test_write_report_produces_both_files(out_dir):
    write_report(_result(), {"body_part": "BRAIN", "model": "SIGNA", "field_T": 3.0})

    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()


def test_report_json_round_trips_result_fields(out_dir):
    write_report(_result(), {"body_part": "BRAIN", "model": "SIGNA", "field_T": 3.0})

    data = json.loads((out_dir / "report.json").read_text())
    assert data["engine"] == "ollama:llava"
    assert data["impression"] == "No acute findings."
    assert data["confidence"] == "high"
    assert data["flags"] == ["low-snr on Seri9"]
    assert data["study"]["body_part"] == "BRAIN"


def test_report_markdown_contains_key_sections(out_dir):
    write_report(_result(), {"body_part": "BRAIN", "model": "SIGNA", "field_T": 3.0})

    md = (out_dir / "report.md").read_text()
    assert "# MRI Analysis Report" in md
    assert "No acute findings." in md
    assert "**Overall confidence:** high" in md
    assert "## Observations" in md
    assert "## Flags" in md
    assert "low-snr on Seri9" in md


def test_report_markdown_shows_placeholder_when_confidence_not_computed(out_dir):
    result = AnalysisResult(engine="ollama:llava", sequences_reviewed=[],
                            impression="", disclaimer="x")  # confidence left at default ""
    write_report(result, {})

    md = (out_dir / "report.md").read_text()
    assert "**Overall confidence:** _(not computed)_" in md


def test_report_markdown_handles_no_observations_or_flags(out_dir):
    result = AnalysisResult(engine="ollama:llava", sequences_reviewed=[],
                            impression="", disclaimer="x")
    write_report(result, {})

    md = (out_dir / "report.md").read_text()
    assert "_(none reported)_" in md
    assert "## Flags" not in md


def test_report_json_handles_unicode_content(out_dir):
    result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"],
                            impression="Étude normale — 所見なし", disclaimer="x")
    write_report(result, {"body_part": "BRAIN"})

    data = json.loads((out_dir / "report.json").read_text())
    assert data["impression"] == "Étude normale — 所見なし"


def test_report_json_does_not_crash_on_nan_study_field(out_dir):
    """study_meta can carry NaN (e.g. an unparseable MagneticFieldStrength --
    see mri.tags.extract_tags' `num()` helper). json.dumps allows NaN/
    Infinity by default (non-standard JSON, but doesn't raise) -- pinning
    that write_report doesn't crash on it, since it's a plausible real value.
    """
    result = AnalysisResult(engine="ollama:llava", sequences_reviewed=[],
                            impression="ok", disclaimer="x")
    write_report(result, {"field_T": float("nan")})

    text = (out_dir / "report.json").read_text()
    assert "NaN" in text  # documents current (non-strict-JSON) behavior
