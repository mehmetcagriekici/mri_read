"""Root logging configuration: console + persistent file handler."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(log_path: Path | None = None,
                      level: int = logging.INFO) -> None:
    """Configure the root logger for a CLI run.

    Console handler: plain, unprefixed (`%(message)s`) -- matches the look
    print() used to produce before mri_read/ modules switched to logging.
    File handler (only when `log_path` is given): timestamped and appended,
    not overwritten, so a run's history survives past its own process --
    useful for a long CPU-bound run watched from a different terminal, or
    inspected after the fact instead of live.

    Safe to call more than once per process (e.g. a script re-entered in a
    test): clears any handlers this function previously added instead of
    stacking duplicates.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_mri_read_managed", False):
            root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    console._mri_read_managed = True
    root.addHandler(console)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        file_handler._mri_read_managed = True
        root.addHandler(file_handler)
