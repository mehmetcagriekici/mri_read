"""ClaudeVisionEngine tests.

The `anthropic` SDK isn't installed by default in this project (see
CLAUDE.md -- it's a non-local, opt-in engine). A fake module is injected
into sys.modules for the duration of each test so `import anthropic` inside
__init__ succeeds without requiring the real package.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mri_read.engine import SeriesImages


@pytest.fixture
def fake_anthropic_module():
    fake_module = MagicMock()
    with patch.dict(sys.modules, {"anthropic": fake_module}):
        yield fake_module


@pytest.fixture(autouse=True)
def _no_dotenv_and_has_key(monkeypatch):
    monkeypatch.setattr("mri_read.claude_vision.engine_impl.load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


def test_raises_clear_error_when_api_key_missing(monkeypatch, fake_anthropic_module):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mri_read.claude_vision.engine_impl import ClaudeVisionEngine

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
        ClaudeVisionEngine()


def test_init_constructs_anthropic_client(fake_anthropic_module):
    from mri_read.claude_vision.engine_impl import ClaudeVisionEngine

    ClaudeVisionEngine()
    fake_anthropic_module.Anthropic.assert_called_once()


def test_analyze_sends_interleaved_text_and_image_blocks(fake_anthropic_module):
    from mri_read.claude_vision.engine_impl import ClaudeVisionEngine

    fake_client = fake_anthropic_module.Anthropic.return_value
    fake_client.messages.create.return_value = SimpleNamespace(content=[
        SimpleNamespace(type="text", text='{"impression": "ok", "observations": [], '
                                          '"flags": [], "sequences_reviewed": ["T2"]}'),
    ])

    engine = ClaudeVisionEngine()
    series = [SeriesImages("Seri1", "T2", "Axial", [0, 1], [b"png1", b"png2"])]
    result = engine.analyze({"body_part": "BRAIN"}, series)

    assert result.impression == "ok"
    assert result.sequences_reviewed == ["T2"]

    content = fake_client.messages.create.call_args.kwargs["messages"][0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    assert len(image_blocks) == 2


def test_analyze_tolerates_malformed_json_reply(fake_anthropic_module):
    from mri_read.claude_vision.engine_impl import ClaudeVisionEngine

    fake_client = fake_anthropic_module.Anthropic.return_value
    fake_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="not json")])

    engine = ClaudeVisionEngine()
    series = [SeriesImages("Seri1", "T2", "Axial", [0], [b"png1"])]
    result = engine.analyze({}, series)

    assert "unparsed" in result.flags


def test_format_acq_renders_present_values():
    from mri_read.claude_vision.engine_impl import _format_acq
    assert _format_acq({"TE": 96.36, "TR": 8200.0}) == " Acquisition: TE=96.36ms, TR=8200.0ms."


def test_format_acq_empty_when_nothing_present():
    from mri_read.claude_vision.engine_impl import _format_acq
    assert _format_acq({}) == ""


def test_analyze_includes_acq_params_in_the_sequence_label(fake_anthropic_module):
    from mri_read.claude_vision.engine_impl import ClaudeVisionEngine

    fake_client = fake_anthropic_module.Anthropic.return_value
    fake_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"impression": "ok"}')])

    engine = ClaudeVisionEngine()
    series = [SeriesImages("Seri6", "T2 FLAIR", "Axial", [0], [b"png1"],
                          acq={"TE": 96.36, "TR": 8200.0, "TI": 2376.04})]
    engine.analyze({}, series)

    content = fake_client.messages.create.call_args.kwargs["messages"][0]["content"]
    label_block = next(b for b in content if b["type"] == "text" and "T2 FLAIR" in b["text"])
    assert "TE=96.36ms" in label_block["text"]
    assert "TR=8200.0ms" in label_block["text"]
    assert "TI=2376.04ms" in label_block["text"]
