"""
Step 4 — Analysis engine interface (the "port" in a ports-and-adapters design).

This file defines the CONTRACT, not any model. The orchestrator (analyze.py)
only ever depends on the shapes here — it hands an engine the same input (study
metadata + selected slice images per series) and always gets back the same
AnalysisResult. That decoupling is what lets us swap Ollama for Claude for a
future specialized brain-MRI model without touching analyze.py.

Three things live here:
  - SeriesImages   : the per-series input given to an engine.
  - AnalysisResult : the uniform output every engine must return.
  - AnalysisEngine : the abstract base class (one method: analyze()).
  - get_engine()   : a factory that maps a name string to a concrete engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SeriesImages:
    """One series' worth of engine input: a few chosen slices, ready to send.

    slice_pngs are already windowed to 8-bit and PNG-encoded (bytes), so an
    engine just has to base64 them — it never touches raw DICOM.
    """
    series: str                   # folder name, e.g. "Seri6"
    label: str                    # sequence label from the manifest, e.g. "T2 FLAIR"
    plane: str                    # Axial / Coronal / Sagittal
    slice_indices: list[int]      # which slice numbers were chosen
    slice_pngs: list[bytes]       # PNG bytes, one per chosen slice


@dataclass
class AnalysisResult:
    """The uniform result shape, regardless of which engine produced it.

    Keeping this fixed means the report writer in analyze.py works for every
    engine. `raw` retains the engine's original parsed payload for debugging.
    """
    engine: str                                     # which engine + model ran
    sequences_reviewed: list[str]
    observations: list[dict] = field(default_factory=list)  # {sequence,finding,location,confidence}
    impression: str = ""                            # short overall summary
    flags: list[str] = field(default_factory=list)  # caveats / limitations
    disclaimer: str = ""
    raw: dict | None = None                         # engine-specific, for debugging


class AnalysisEngine(ABC):
    """Abstract base every engine subclasses. The whole contract is analyze()."""
    name: str = "base"

    @abstractmethod
    def analyze(self, study_meta: dict,
                series: list[SeriesImages]) -> AnalysisResult:
        """Look at the provided slices and return a structured AnalysisResult."""
        ...


def get_engine(name: str, **kwargs) -> AnalysisEngine:
    """Factory so analyze.py can pick an engine by name from the CLI.

    Default is 'ollama' — fully local, nothing leaves the machine. The Claude
    engine is a NON-LOCAL option kept only to demonstrate the swappable
    interface; do not use it on sensitive data.
    """
    # Imports are LAZY (inside each branch) so choosing the local engine never
    # imports the Claude SDK, and vice versa — you only need the deps you use.
    if name in ("ollama", "local"):
        from mri_read.ollama_vision import OllamaVisionEngine
        return OllamaVisionEngine(**kwargs)
    if name in ("claude", "claude-vision"):
        from mri_read.claude_vision import ClaudeVisionEngine  # 3rd-party / non-local
        return ClaudeVisionEngine(**kwargs)
    # To add a specialized model later, add a branch here pointing at a new file
    # that implements AnalysisEngine — nothing else in the project changes:
    #   if name == "medmodel": from mri_read.med_model import MedModelEngine; return ...
    raise ValueError(f"Unknown engine: {name!r}")
