"""Shared filesystem locations, computed once relative to the repo root.

Every module used to recompute this itself via `Path(__file__).resolve()...`,
which broke when files moved a directory deeper into mri_read/. One shared
computation here means only this file needs updating if the layout moves again.
"""

from pathlib import Path

# mri_read/paths.py -> mri_read/ -> src/ -> repo root
ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT / "mri_test_data"
OUT = ROOT / "output"
