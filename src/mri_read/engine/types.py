"""The data shapes every AnalysisEngine consumes and produces.

Keeping these fixed is what lets the orchestrator (analyze.py) and the report
writer work unchanged regardless of which engine ran.
"""

from __future__ import annotations

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
