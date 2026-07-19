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
from mri_read.ollama_vision.engine_impl import OllamaVisionEngine, _format_acq


def _series(label: str, series: str = "Seri1", acq: dict | None = None) -> SeriesImages:
    return SeriesImages(series=series, label=label, plane="Axial",
                        slice_indices=[1, 2], slice_pngs=[b"png1", b"png2"],
                        acq=acq or {})


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


# --- acquisition parameters (TE/TR/TI) in the prompt --------------------

def test_format_acq_renders_present_values():
    assert _format_acq({"TE": 96.36, "TR": 8200.0, "TI": 2376.04}) == (
        "Acquisition: TE=96.36ms, TR=8200.0ms, TI=2376.04ms.\n"
    )


def test_format_acq_skips_missing_values():
    assert _format_acq({"TE": 90.24, "TR": None}) == "Acquisition: TE=90.24ms.\n"


def test_format_acq_empty_when_nothing_present():
    assert _format_acq({}) == ""
    assert _format_acq({"TE": None, "TR": None, "TI": None}) == ""


def test_analyze_one_includes_acq_params_in_the_user_message(engine):
    series = _series("T2 FLAIR", acq={"TE": 96.36, "TR": 8200.0, "TI": 2376.04})
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content": "{}"}}) as mock_post:
        engine._analyze_one({}, series)

    user_content = mock_post.call_args[0][2]["messages"][1]["content"]
    assert "TE=96.36ms" in user_content
    assert "TR=8200.0ms" in user_content
    assert "TI=2376.04ms" in user_content


def test_analyze_one_omits_acquisition_line_when_no_acq_data(engine):
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content": "{}"}}) as mock_post:
        engine._analyze_one({}, _series("T2"))  # no acq given

    user_content = mock_post.call_args[0][2]["messages"][1]["content"]
    assert "Acquisition:" not in user_content


def test_analyze_one_caps_reply_length():
    """Regression test for a real incident: a stuck generation with no
    output-length cap ran for 50+ minutes with a single Ollama processing
    slot (-np 1), queuing every subsequent series behind it -- all 5 series
    in that run timed out, not just the stuck one. num_predict bounds
    worst-case generation time regardless of whether the model ever finds a
    natural stop point.
    """
    from mri_read.ollama_vision.engine_impl import MAX_REPLY_TOKENS

    with patch("mri_read.ollama_vision.engine_impl.ensure_model",
              return_value="llava:13b"):
        eng = OllamaVisionEngine(model="llava:13b", host="http://localhost:11434")
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content": "{}"}}) as mock_post:
        eng._analyze_one({}, _series("T2"))

    payload = mock_post.call_args[0][2]  # post(host, path, payload, timeout)
    assert payload["options"]["num_predict"] == MAX_REPLY_TOKENS


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


# --- duration logging (see logging_setup for why this exists) --------------

def test_successful_call_logs_its_duration(engine, caplog):
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content": "{}"}}):
        with caplog.at_level("INFO", logger="mri_read.ollama_vision.engine_impl"):
            engine.analyze({}, [_series("T2")])

    done_logs = [r.message for r in caplog.records if "T2 (Seri1) done in" in r.message]
    assert len(done_logs) == 1


def test_failed_call_logs_its_duration_as_a_warning(engine, caplog):
    with patch("mri_read.ollama_vision.engine_impl.post",
              side_effect=socket.timeout("timed out")):
        with caplog.at_level("INFO", logger="mri_read.ollama_vision.engine_impl"):
            with pytest.raises(RuntimeError):
                engine.analyze({}, [_series("T2")])

    failed_logs = [r for r in caplog.records if "FAILED after" in r.message]
    assert len(failed_logs) == 1
    assert failed_logs[0].levelname == "WARNING"


def test_analyze_logs_an_overall_summary(engine, caplog):
    with patch("mri_read.ollama_vision.engine_impl.post",
              return_value={"message": {"content": "{}"}}):
        with caplog.at_level("INFO", logger="mri_read.ollama_vision.engine_impl"):
            engine.analyze({}, [_series("T2"), _series("T2 FLAIR", series="Seri2")])

    summary_logs = [r.message for r in caplog.records if "all 2 series done" in r.message]
    assert len(summary_logs) == 1
    assert "2 succeeded, 0 failed" in summary_logs[0]
