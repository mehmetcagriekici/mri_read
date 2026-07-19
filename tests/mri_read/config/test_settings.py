import importlib

import mri_read.config.settings as settings_module


def test_default_config_has_expected_fields():
    cfg = settings_module.DEFAULT
    assert isinstance(cfg.ollama_host, str) and cfg.ollama_host
    assert isinstance(cfg.vision_model, str) and cfg.vision_model
    assert isinstance(cfg.agent_model, str) and cfg.agent_model


def test_config_is_frozen():
    cfg = settings_module.Config()
    try:
        cfg.ollama_host = "changed"
        assert False, "Config should be immutable (frozen dataclass)"
    except AttributeError:
        pass


def test_env_vars_override_defaults(monkeypatch):
    # Defaults are read via os.environ.get() at class-definition time, so
    # picking up a changed env var requires reloading the module.
    monkeypatch.setenv("OLLAMA_HOST", "http://example:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "test-vision-model")
    monkeypatch.setenv("OLLAMA_AGENT_MODEL", "test-agent-model")
    try:
        reloaded = importlib.reload(settings_module)
        assert reloaded.DEFAULT.ollama_host == "http://example:9999"
        assert reloaded.DEFAULT.vision_model == "test-vision-model"
        assert reloaded.DEFAULT.agent_model == "test-agent-model"
    finally:
        importlib.reload(settings_module)  # restore real-env defaults for other tests
