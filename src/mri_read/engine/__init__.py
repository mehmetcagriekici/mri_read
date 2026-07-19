"""Step 4 — Analysis engine interface (the "port" in a ports-and-adapters design).

This package defines the CONTRACT, not any model. The orchestrator (analyze.py)
only ever depends on the shapes here — it hands an engine the same input (study
metadata + selected slice images per series) and always gets back the same
AnalysisResult. That decoupling is what lets us swap Ollama for Claude for a
future specialized brain-MRI model without touching analyze.py.

Three things live here:
  - SeriesImages   : the per-series input given to an engine (types.py).
  - AnalysisResult : the uniform output every engine must return (types.py).
  - AnalysisEngine : the abstract base class, one method: analyze() (base.py).
  - get_engine()   : a factory that maps a name string to a concrete engine
                      (factory.py).

Also here: CONFIDENCE_LEVELS / normalize_confidence (types.py) -- the strict
schema for the `confidence` field carried by both AnalysisResult and every
observation dict (`{sequence, finding, location, confidence}`).
"""

from mri_read.engine.base import AnalysisEngine
from mri_read.engine.factory import get_engine
from mri_read.engine.types import (CONFIDENCE_LEVELS, AnalysisResult,
                                   SeriesImages, normalize_confidence)

__all__ = ["SeriesImages", "AnalysisResult", "AnalysisEngine", "get_engine",
          "CONFIDENCE_LEVELS", "normalize_confidence"]
