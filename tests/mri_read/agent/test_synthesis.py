from unittest.mock import patch

from mri_read.agent.synthesis import _synthesize
from mri_read.engine import AnalysisResult


def _vision_result():
    return AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"],
                          observations=[], impression="draft", flags=["a-flag"])


def test_synthesize_forwards_timeout_and_returns_parsed_json():
    manifest = {"series": [{"series": "Seri1", "label": "T2",
                            "use_for_analysis": True, "qc": {"status": "pass"}}]}
    with patch("mri_read.agent.synthesis.post",
              return_value={"message": {"content": '{"impression": "ok", "flags": []}'}}) as mock_post:
        result = _synthesize("meditron:7b", "http://localhost:11434", {}, manifest,
                             _vision_result(), timeout=77)

    assert result == {"impression": "ok", "flags": []}
    assert mock_post.call_args[0][-1] == 77


def test_synthesize_tolerates_malformed_reply():
    manifest = {"series": []}
    with patch("mri_read.agent.synthesis.post",
              return_value={"message": {"content": "not json"}}):
        result = _synthesize("meditron:7b", "http://x", {}, manifest, _vision_result(), 10)

    assert result["flags"] == ["unparsed"]
    assert result["impression"] == "not json"


def test_synthesize_treats_explicit_null_flags_as_empty_list():
    manifest = {"series": []}
    with patch("mri_read.agent.synthesis.post",
              return_value={"message": {"content": '{"impression": "ok", "flags": null}'}}):
        result = _synthesize("meditron:7b", "http://x", {}, manifest, _vision_result(), 10)

    assert result["flags"] == []
