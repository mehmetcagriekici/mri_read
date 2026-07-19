"""configure_logging(): console + optional persistent file handler.

The root logger is process-global state, so every test here snapshots and
restores it -- these tests must not leak handlers into the rest of the suite
(or into pytest's own logging capture).
"""

from __future__ import annotations

import logging

import pytest

from mri_read.logging_setup import configure_logging


@pytest.fixture(autouse=True)
def _isolated_root_logger():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        yield
    finally:
        for h in list(root.handlers):
            if h not in original_handlers:
                root.removeHandler(h)
        root.handlers.extend(h for h in original_handlers if h not in root.handlers)
        root.setLevel(original_level)


def _managed_handlers():
    return [h for h in logging.getLogger().handlers
           if getattr(h, "_mri_read_managed", False)]


def test_adds_a_console_handler_with_plain_format():
    configure_logging()
    console_handlers = [h for h in _managed_handlers()
                        if isinstance(h, logging.StreamHandler)
                        and not isinstance(h, logging.FileHandler)]
    assert len(console_handlers) == 1
    assert console_handlers[0].formatter._fmt == "%(message)s"


def test_no_file_handler_when_log_path_omitted():
    configure_logging()
    file_handlers = [h for h in _managed_handlers() if isinstance(h, logging.FileHandler)]
    assert file_handlers == []


def test_adds_a_timestamped_file_handler_when_log_path_given(tmp_path):
    log_path = tmp_path / "agent.log"
    configure_logging(log_path)
    file_handlers = [h for h in _managed_handlers() if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert "%(asctime)s" in file_handlers[0].formatter._fmt


def test_file_handler_appends_not_truncates(tmp_path):
    log_path = tmp_path / "agent.log"
    configure_logging(log_path)
    logging.getLogger("test").info("first run")
    configure_logging(log_path)  # simulates a second CLI invocation
    logging.getLogger("test").info("second run")

    content = log_path.read_text()
    assert "first run" in content
    assert "second run" in content


def test_creates_parent_directory_if_missing(tmp_path):
    log_path = tmp_path / "nested" / "output" / "agent.log"
    configure_logging(log_path)
    assert log_path.parent.is_dir()


def test_calling_twice_does_not_accumulate_duplicate_handlers(tmp_path):
    log_path = tmp_path / "agent.log"
    configure_logging(log_path)
    configure_logging(log_path)
    configure_logging(log_path)

    handlers = _managed_handlers()
    console_handlers = [h for h in handlers if not isinstance(h, logging.FileHandler)]
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    assert len(console_handlers) == 1
    assert len(file_handlers) == 1


def test_root_logger_level_is_set():
    configure_logging(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG
