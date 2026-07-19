"""The AnalysisEngine contract (the "port" in a ports-and-adapters design)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mri_read.engine.types import AnalysisResult, SeriesImages


class AnalysisEngine(ABC):
    """Abstract base every engine subclasses. The whole contract is analyze()."""
    name: str = "base"

    @abstractmethod
    def analyze(self, study_meta: dict,
                series: list[SeriesImages]) -> AnalysisResult:
        """Look at the provided slices and return a structured AnalysisResult."""
        ...
