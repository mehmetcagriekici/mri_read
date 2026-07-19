"""State/error types the agent pipeline hands back to its CLI caller."""

from __future__ import annotations

from dataclasses import dataclass

from mri_read.engine import AnalysisResult


@dataclass
class AgentContext:
    """State returned to the CLI after one pipeline run."""
    manifest: dict | None = None
    last_result: AnalysisResult | None = None


class PipelineError(RuntimeError):
    """The vision engine or text-synthesis call failed.

    Carries the AgentContext built so far (manifest + QC already ran) so the
    caller can still persist output/manifest.json for debugging instead of
    losing that work to an uncaught traceback.
    """
    def __init__(self, message: str, ctx: AgentContext):
        super().__init__(message)
        self.ctx = ctx
