"""post() must forward the caller's timeout to urllib untouched.

This is the seam every configurable --vision-timeout/--synth-timeout flag
ultimately depends on: if post() ever hardcoded or dropped the timeout
argument, every per-call timeout in the app would silently stop working.
"""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import MagicMock, patch

from mri_read.ollama_client.http import post


def _mock_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_post_forwards_timeout_to_urlopen():
    with patch.object(urllib.request, "urlopen", return_value=_mock_response({"ok": True})) as mock_urlopen:
        post("http://localhost:11434", "/api/chat", {"model": "x"}, timeout=123)

    _, kwargs = mock_urlopen.call_args
    assert kwargs["timeout"] == 123


def test_post_returns_parsed_json():
    with patch.object(urllib.request, "urlopen", return_value=_mock_response({"message": {"content": "hi"}})):
        result = post("http://localhost:11434", "/api/chat", {}, timeout=5)

    assert result == {"message": {"content": "hi"}}


def test_post_strips_trailing_slash_from_host():
    with patch.object(urllib.request, "urlopen", return_value=_mock_response({})) as mock_urlopen:
        post("http://localhost:11434/", "/api/tags", {}, timeout=5)

    request = mock_urlopen.call_args[0][0]
    assert request.full_url == "http://localhost:11434/api/tags"
