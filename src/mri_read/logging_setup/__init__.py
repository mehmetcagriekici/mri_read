"""
Shared root-logging configuration for CLI entry points (src/cmd/*.py).

Every module used to configure this differently (or not at all) -- only
cmd/agent.py and cmd/analyze.py called logging.basicConfig(), each with
their own copy of the same console-only setup, and neither persisted
anything to disk. For a run that can take 30+ minutes on CPU-only hardware
and is often watched from a separate terminal or polled in the background,
that meant the only record of what happened was whatever scrollback
happened to still be on screen.

configure_logging() gives every CLI entry point the same two handlers:
  - console : plain, unprefixed -- unchanged from the previous look.
  - file    : timestamped, appended to output/agent.log (or wherever the
              caller points it) -- survives past the process, so a
              backgrounded or detached run's timing is inspectable
              afterward, not just live.

Layout:
  configure.py : configure_logging(), the only function here.
"""

from mri_read.logging_setup.configure import configure_logging

__all__ = ["configure_logging"]
