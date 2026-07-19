"""Minimal .env loader so ANTHROPIC_API_KEY can live in a file."""

from __future__ import annotations

import os

from mri_read.paths import ROOT


def load_dotenv() -> None:
    """Load ROOT/.env into os.environ (without overwriting existing vars)."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
