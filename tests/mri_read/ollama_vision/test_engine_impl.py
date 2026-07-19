"""OllamaVisionEngine: timeout threading + per-series failure isolation.

The whole point of the per-series call split (see engine_impl.py's docstring)
is that one slow/timed-out sequence must not sink the whole report. These
tests pin that behavior with mocks -- no real Ollama server, no real waiting.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from mri_read.engine import SeriesImages
from mri_read.ollama_vision.engine_impl import OllamaVisionEngine


def _series(label: str, series: str = "Seri1") -> SeriesImages:
    return SeriesImages(series=series, label=label, plane="Axial",
                        slice_indices=[1, 2], slice_pngs=[b"png1", b"png2"])


@pytest.fixture
def engine():
    with patch("mri_read.ollama_vision.engine_impl.ensure_model",
              return_value="llava:13b"):
        return OllamaVisionEngine(model="llava:13b", host="http://localhost:11434",
                                  timeout=42)


def test_init_resolves_model_via_ensure_model_when_auto_pull():
    with patch("mri_read.ollama_vision.engine_impl.ensure_model",
              return_value="llava:13b-resolved") as mock_ensure:
        eng = OllamaVisionEngine(model="llava:13b", auto_pull=True)
    mock_ensure.assert_called_once()
    assert eng.model == "llava:13b-resolved"


def test_init_skips_ensure_model_when_auto_pull_false():
    with patch("mri_read.ollama_vision.engine_impl.ensure_model") as mock_ensure:
        eng = OllamaVisionEngine(model="llava:13b", auto_pull=False)
    mock_ensure.assert_not_called()
    assert eng.model == "llava:13b"


def test_analyze_one_forwards_configured_timeout_to_post(engine):
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content": "{}"}}) as mock_post:
        engine._analyze_one({}, _series("T2"))
    assert mock_post.call_args[0][-1] == 42  # post(host, path, payload, timeout)


def test_one_series_timeout_becomes_a_flag_not_a_crash(engine):
    """A per-series socket timeout must not stop the other series from being
    analyzed, and must not propagate out of analyze() -- it becomes a flag.
    """
    good_reply = {"message": {"content":
        '{"sequences_reviewed": ["T2"], "observations": [], '
        '"impression": "looks normal", "flags": []}'}}

    def fake_post(host, path, payload, timeout):
        if "T1" in payload["messages"][1]["content"]:
            raise socket.timeout("timed out")
        return good_reply

    with patch("mri_read.ollama_vision.engine_impl.post", side_effect=fake_post):
        result = engine.analyze({}, [_series("T1"), _series("T2")])

    assert "T2" in result.sequences_reviewed
    assert any("T1" in f and "failed" in f for f in result.flags)
    assert "Seri1" in result.raw  # only the T2 call's raw payload made it in


def test_all_series_failing_raises_instead_of_returning_empty_result(engine):
    with patch("mri_read.ollama_vision.engine_impl.post",
              side_effect=socket.timeout("timed out")):
        with pytest.raises(RuntimeError, match="all series failed"):
            engine.analyze({}, [_series("T1"), _series("T2")])


def test_malformed_json_reply_degrades_to_unparsed_flag_not_a_crash(engine):
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content": "not json at all"}}):
        result = engine.analyze({}, [_series("T2")])

    assert "unparsed" in result.flags


def test_malformed_reply_does_not_leak_raw_text_as_impression(engine):
    """A hallucinated reply that fails to parse must not surface verbatim in
    the report -- see agent.synthesis' matching regression test for the real
    incident this covers (a fake patient-record schema landed in a report).
    """
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content":
                  '{"patient": {"first_name": "John"}, "second": 00}'}}):
        result = engine.analyze({}, [_series("T2")])

    assert "John" not in result.impression
    assert result.raw["Seri1"]["impression"] == "unknown"


def test_ground_truth_label_used_even_when_model_self_reports_wrong_sequence(engine):
    """Regression test for a real incident: a call scoped to a T2 series got
    a reply self-reporting "sequences_reviewed": ["T2 FLAIR"] -- the wrong
    sequence -- which silently made T2 vanish from the report with no
    failure flag, its real finding merged under T2 FLAIR instead. s.label
    (what was actually sent) must always win over the model's self-report.
    """
    mislabeled_reply = {"message": {"content":
        '{"sequences_reviewed": ["T2 FLAIR"], '
        '"observations": [{"sequence": "T2 FLAIR", "finding": "real finding", '
        '"location": "frontal lobe", "confidence": "moderate"}], '
        '"impression": "ok", "flags": []}'}}

    with patch("mri_read.ollama_vision.engine_impl.post", return_value=mislabeled_reply):
        result = engine.analyze({}, [_series("T2", series="Seri3")])

    assert result.sequences_reviewed == ["T2"]          # ground truth, not "T2 FLAIR"
    assert result.observations[0]["sequence"] == "T2"   # corrected, not the model's claim
    assert result.observations[0]["finding"] == "real finding"  # content preserved


def test_placeholder_echo_observation_is_dropped_and_flagged(engine):
    reply_with_echo = {"message": {"content":
        '{"sequences_reviewed": ["T2 FLAIR"], '
        '"observations": [{"sequence": "T2 FLAIR", "finding": "...", '
        '"location": "...", "confidence": "low|moderate|high"}], '
        '"impression": "ok", "flags": []}'}}

    with patch("mri_read.ollama_vision.engine_impl.post", return_value=reply_with_echo):
        result = engine.analyze({}, [_series("T2 FLAIR")])

    assert result.observations == []
    assert any("dropped" in f and "hallucinated" in f for f in result.flags)
