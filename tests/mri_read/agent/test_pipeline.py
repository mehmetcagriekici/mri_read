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
