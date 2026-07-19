"""Static regression guard: no dynamic-execution primitives in src/mri_read/.

This project reads local DICOM files and talks to a local Ollama server --
nothing here should ever need eval/exec/os.system/subprocess/pickle. A grep-
style check is cheap insurance against one creeping in later (e.g. a
"convenient" os.system() call in a new CLI wrapper).
"""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN = re.compile(
    r"\b(eval|exec)\s*\(|os\.system|subprocess\.|__import__\s*\(|"
    r"pickle\.(load|loads)|shell\s*=\s*True"
)

SRC = Path(__file__).resolve().parent.parent.parent / "src" / "mri_read"


def test_no_dynamic_execution_or_shell_primitives():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")

    assert not offenders, "Found dangerous execution patterns:\n" + "\n".join(offenders)
