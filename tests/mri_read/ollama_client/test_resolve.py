"""resolve_model(): tag matching, the fixed 15s connectivity-check timeout,
and failing fast (not hanging) when Ollama is unreachable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from mri_read.ollama_client.resolve import resolve_model


def _mock_tags_response(names: list[str]):
    resp = MagicMock()
    resp.read.return_value = json.dumps(
        {"models": [{"name": n} for n in names]}
    ).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_exact_tag_match():
    with patch.object(urllib.request, "urlopen",
                      return_value=_mock_tags_response(["llava:13b", "qwen2.5:7b"])):
        assert resolve_model("http://localhost:11434", "llava:13b") == "llava:13b"


def test_falls_back_to_base_name_match():
    with patch.object(urllib.request, "urlopen",
                      return_value=_mock_tags_response(["qwen2.5:7b-instruct"])):
        assert resolve_model("http://localhost:11434", "qwen2.5") == "qwen2.5:7b-instruct"


def test_no_match_returns_none():
    with patch.object(urllib.request, "urlopen",
                      return_value=_mock_tags_response(["llava:13b"])):
        assert resolve_model("http://localhost:11434", "meditron:7b") is None


def test_uses_a_short_fixed_timeout_for_the_connectivity_check():
    """15s, not caller-configurable -- this check must fail fast even if the
    caller's own --vision-timeout/--synth-timeout is set to several minutes.
    """
    with patch.object(urllib.request, "urlopen",
                      return_value=_mock_tags_response([])) as mock_urlopen:
        resolve_model("http://localhost:11434", "llava:13b")

    _, kwargs = mock_urlopen.call_args
    assert kwargs["timeout"] == 15


def test_unreachable_host_raises_clear_error_instead_of_hanging():
    with patch.object(urllib.request, "urlopen",
                      side_effect=urllib.error.URLError("connection refused")):
        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            resolve_model("http://localhost:11434", "llava:13b")
