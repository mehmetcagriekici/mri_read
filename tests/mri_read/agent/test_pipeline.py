"""run_agent(): a vision or synthesis timeout must not lose the already-built
manifest/QC work -- it becomes a PipelineError carrying ctx, not a bare crash.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from mri_read.agent.context import PipelineError
from mri_read.agent.pipeline import run_agent
from mri_read.engine import AnalysisResult

FAKE_MANIFEST = {
    "study": {"body_part": "BRAIN"},
    "series": [{"series": "Seri1", "label": "T2", "use_for_analysis": True}],
}


@pytest.fixture(autouse=True)
def _patch_pipeline_steps():
    with patch("mri_read.agent.pipeline.build_manifest", return_value=FAKE_MANIFEST), \
         patch("mri_read.agent.pipeline.run_qc", return_value={"status": "pass", "flags": [], "metrics": {}}), \
         patch("mri_read.agent.pipeline.select_series", return_value=[]), \
         patch("mri_read.agent.pipeline.write_report"):
        yield


def test_vision_timeout_raises_pipeline_error_with_manifest_preserved():
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine:
        mock_get_engine.return_value.analyze.side_effect = socket.timeout("timed out")
        with pytest.raises(PipelineError, match="vision engine failed") as exc_info:
            run_agent()

    # the manifest built before the vision call must survive the failure --
    # this is what lets the CLI persist output/manifest.json for debugging.
    assert exc_info.value.ctx.manifest == FAKE_MANIFEST
    assert exc_info.value.ctx.last_result is None


def test_synthesis_timeout_raises_pipeline_error_with_manifest_preserved():
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"])
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize", side_effect=socket.timeout("timed out")):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        with pytest.raises(PipelineError, match="text-synthesis model failed") as exc_info:
            run_agent()

    assert exc_info.value.ctx.manifest == FAKE_MANIFEST
    assert exc_info.value.ctx.last_result is None


def test_synth_timeout_kwarg_forwarded_to_synthesize():
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=[])
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "ok", "flags": []}) as mock_synth:
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        run_agent(timeout=123)

    assert mock_synth.call_args[0][-1] == 123  # _synthesize(..., timeout)


def test_host_injected_into_vision_kwargs_for_ollama_engine():
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=[])
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "ok", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        run_agent(host="http://example:11434", engine_name="ollama")

    _, kwargs = mock_get_engine.call_args
    assert kwargs["host"] == "http://example:11434"


def test_host_not_injected_for_non_ollama_engine():
    fake_vision_result = AnalysisResult(engine="claude-vision", sequences_reviewed=[])
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "ok", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        run_agent(host="http://example:11434", engine_name="claude")

    _, kwargs = mock_get_engine.call_args
    assert "host" not in kwargs


# --- hallucination guard wiring ----------------------------------------------

def test_uncorroborated_concerning_observation_is_suppressed_end_to_end():
    from mri_read.agent.guard import REDACTED_FINDING

    concerning_obs = [{"sequence": "T2", "finding": "this appears to be a tumor",
                       "location": "frontal lobe", "confidence": "high"}]
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"],
                                        observations=concerning_obs)
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "See observations.", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        _, ctx = run_agent()

    assert ctx.last_result.observations[0]["finding"] == REDACTED_FINDING
    assert any("suppressed" in f for f in ctx.last_result.flags)


def test_unsupported_claim_in_synthesized_impression_is_redacted_end_to_end():
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"])
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "Findings suggest a possible tumor.", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        _, ctx = run_agent()

    assert "tumor" not in ctx.last_result.impression.lower()
    assert "[unconfirmed finding]" in ctx.last_result.impression
    assert ctx.last_result.confidence == "low"


def test_clean_report_gets_a_computed_confidence():
    clean_obs = [{"sequence": "T2", "finding": "no abnormality", "location": "n/a",
                 "confidence": "high"}]
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"],
                                        observations=clean_obs)
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "No acute findings.", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        _, ctx = run_agent()

    assert ctx.last_result.confidence == "high"
    assert ctx.last_result.impression == "No acute findings."


# --- stage-level duration logging -------------------------------------------

def test_run_logs_every_stage_with_a_duration(caplog):
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"])
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "ok", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        with caplog.at_level("INFO", logger="mri_read.agent.pipeline"):
            run_agent()

    messages = [r.message for r in caplog.records]
    assert any("starting run" in m for m in messages)
    assert any("manifest + QC done in" in m for m in messages)
    assert any("selected" in m and "series-images" in m for m in messages)
    assert any("vision phase" in m and "done in" in m for m in messages)
    assert any("synthesis phase" in m and "done in" in m for m in messages)
    assert any("run complete in" in m and "confidence:" in m for m in messages)


def test_correlation_guard_suppression_is_logged_as_a_warning(caplog):
    concerning_obs = [{"sequence": "T2", "finding": "this appears to be a tumor",
                       "location": "frontal lobe", "confidence": "high"}]
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"],
                                        observations=concerning_obs)
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "ok", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        with caplog.at_level("INFO", logger="mri_read.agent.pipeline"):
            run_agent()

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("suppressed" in r.message for r in warnings)


def test_dwi_correlation_flag_alone_does_not_log_a_false_suppression_warning(caplog):
    """Regression test for a real bug: a DWI finding lacking a structural
    correlate produces a flag from apply_correlation_guard, same as an
    actual suppression -- but logging len(correlation_flags) as "suppressed"
    claimed a redaction happened even though nothing in `observations` was
    actually changed. Confirmed live: a real run logged "correlation guard
    suppressed 1 uncorroborated claim(s)" when the only flag was the DWI/
    structural-correlation note, and no observation's finding was
    REDACTED_FINDING. The warning must only fire on an ACTUAL suppression.
    """
    from mri_read.agent.guard import REDACTED_FINDING

    dwi_only_obs = [
        {"sequence": "DWI", "finding": "Restricted diffusion observed.",
         "location": "left parietal lobe", "confidence": "moderate"},
        {"sequence": "T2 FLAIR", "finding": "No abnormal signal intensity observed.",
         "location": "whole brain", "confidence": "low"},
    ]
    fake_vision_result = AnalysisResult(engine="ollama:llava", sequences_reviewed=["DWI", "T2 FLAIR"],
                                        observations=dwi_only_obs)
    with patch("mri_read.agent.pipeline.get_engine") as mock_get_engine, \
         patch("mri_read.agent.pipeline.ensure_model", return_value="meditron:7b"), \
         patch("mri_read.agent.pipeline._synthesize",
              return_value={"impression": "ok", "flags": []}):
        mock_get_engine.return_value.analyze.return_value = fake_vision_result
        with caplog.at_level("INFO", logger="mri_read.agent.pipeline"):
            _, ctx = run_agent()

    assert not any(o["finding"] == REDACTED_FINDING for o in ctx.last_result.observations)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any("suppressed" in r.message for r in warnings)
