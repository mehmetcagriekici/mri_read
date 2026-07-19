import sys

import pytest

from mri_read.engine import get_engine


def test_unknown_engine_name_raises():
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine("does-not-exist")


def test_ollama_alias_names_route_to_the_same_engine(monkeypatch):
    from mri_read.ollama_vision import OllamaVisionEngine
    monkeypatch.setattr(OllamaVisionEngine, "__init__", lambda self, **kw: None)
    for name in ("ollama", "local"):
        eng = get_engine(name)
        assert isinstance(eng, OllamaVisionEngine)


def test_choosing_ollama_never_imports_anthropic_sdk(monkeypatch):
    """get_engine's lazy per-branch imports are the whole point of the
    factory: picking the local engine must never pull in the Claude SDK.
    """
    from mri_read.ollama_vision import OllamaVisionEngine
    monkeypatch.setattr(OllamaVisionEngine, "__init__", lambda self, **kw: None)
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    get_engine("ollama")

    assert "anthropic" not in sys.modules
