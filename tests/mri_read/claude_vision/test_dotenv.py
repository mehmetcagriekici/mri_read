import os

import mri_read.claude_vision.dotenv as dotenv_module
from mri_read.claude_vision.dotenv import load_dotenv


def test_loads_key_value_pairs_from_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)
    (tmp_path / ".env").write_text('ANTHROPIC_API_KEY="sk-test-123"\n# comment\nFOO=bar\n')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FOO", raising=False)

    load_dotenv()

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123"
    assert os.environ["FOO"] == "bar"


def test_does_not_overwrite_existing_env_var(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-file\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-real-env")

    load_dotenv()

    assert os.environ["ANTHROPIC_API_KEY"] == "from-real-env"


def test_missing_env_file_is_a_silent_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)  # no .env written
    load_dotenv()  # must not raise


def test_lines_without_equals_sign_are_skipped_not_crashed_on(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("this line has no equals sign\nFOO=bar\n")
    monkeypatch.delenv("FOO", raising=False)

    load_dotenv()  # must not raise

    assert os.environ["FOO"] == "bar"


def test_empty_file_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("")
    load_dotenv()  # must not raise


def test_blank_lines_and_comment_only_file_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("\n\n# just a comment\n   \n")
    load_dotenv()  # must not raise


def test_value_containing_an_equals_sign_is_kept_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("SOME_URL=https://example.com/?a=1&b=2\n")
    monkeypatch.delenv("SOME_URL", raising=False)

    load_dotenv()

    assert os.environ["SOME_URL"] == "https://example.com/?a=1&b=2"


def test_key_with_empty_value_is_set_to_empty_string(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv_module, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("EMPTY_VAL=\n")
    monkeypatch.delenv("EMPTY_VAL", raising=False)

    load_dotenv()

    assert os.environ["EMPTY_VAL"] == ""
