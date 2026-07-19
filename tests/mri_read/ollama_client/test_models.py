"""model_present()/ensure_model(): the pull-if-missing path.

ensure_model's pull request applies PULL_STALL_TIMEOUT (a per-read, not
total-duration, socket timeout) so a pull that goes completely silent fails
with a clear error instead of hanging forever -- previously this was
timeout=None (unbounded). It only fires when a model name doesn't resolve to
something already present locally.
"""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from mri_read.ollama_client.models import (PULL_STALL_TIMEOUT, ensure_model,
                                           model_present)


def _mock_tags_response(names: list[str]):
    resp = MagicMock()
    resp.read.return_value = json.dumps(
        {"models": [{"name": n} for n in names]}
    ).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _mock_pull_stream(lines: list[bytes]):
    resp = MagicMock()
    resp.__iter__.return_value = iter(lines)
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_model_present_true_when_tag_resolves():
    with patch.object(urllib.request, "urlopen",
                      return_value=_mock_tags_response(["llava:13b"])):
        assert model_present("http://localhost:11434", "llava:13b") is True


def test_model_present_false_when_no_match():
    with patch.object(urllib.request, "urlopen",
                      return_value=_mock_tags_response([])):
        assert model_present("http://localhost:11434", "llava:13b") is False


def test_ensure_model_skips_pull_when_already_present():
    with patch.object(urllib.request, "urlopen",
                      return_value=_mock_tags_response(["meditron:7b"])) as mock_urlopen:
        result = ensure_model("http://localhost:11434", "meditron:7b")

    assert result == "meditron:7b"
    assert mock_urlopen.call_count == 1  # only the /api/tags check, no pull


def test_ensure_model_pulls_when_missing():
    tags_resp = _mock_tags_response([])
    pull_resp = _mock_pull_stream([json.dumps({"status": "pulling"}).encode()])
    with patch.object(urllib.request, "urlopen",
                      side_effect=[tags_resp, pull_resp]) as mock_urlopen:
        result = ensure_model("http://localhost:11434", "new-model:latest")

    assert result == "new-model:latest"
    assert mock_urlopen.call_count == 2


def test_pull_has_a_bounded_stall_timeout():
    """A stalled pull must fail with a clear error, not hang forever."""
    tags_resp = _mock_tags_response([])
    pull_resp = _mock_pull_stream([])
    with patch.object(urllib.request, "urlopen",
                      side_effect=[tags_resp, pull_resp]) as mock_urlopen:
        ensure_model("http://localhost:11434", "new-model:latest")

    pull_call_kwargs = mock_urlopen.call_args_list[1][1]
    assert pull_call_kwargs["timeout"] == PULL_STALL_TIMEOUT
    assert pull_call_kwargs["timeout"] is not None


def test_stalled_pull_raises_instead_of_hanging():
    tags_resp = _mock_tags_response([])
    with patch.object(urllib.request, "urlopen",
                      side_effect=[tags_resp, TimeoutError("timed out")]):
        with pytest.raises(TimeoutError):
            ensure_model("http://localhost:11434", "new-model:latest")
