"""Enforcing the strict observation schema and filtering hallucinated content.

Two distinct problems, both handled here:
  1. Schema violations: a missing/empty finding or location, or a confidence
     value that isn't one of engine.CONFIDENCE_LEVELS.
  2. Prompt-echo hallucinations: syntactically valid, schema-conformant JSON
     that still isn't a real observation -- llava:13b has been observed
     literally echoing its own system prompt's example schema (e.g.
     "finding": "...", "confidence": "low|moderate|high") or the per-call
     user message's formatting string, instead of describing what's
     actually in the images. parse_json_reply can't catch either of these
     (the JSON is well-formed); both are separate, semantic checks.
"""

from __future__ import annotations

from mri_read.engine import normalize_confidence

_PLACEHOLDER_VALUES = {"..."}


def _is_prompt_echo(value: object, echo_sources: list[str]) -> bool:
    """True if `value` is unfilled template text or copied verbatim from the
    prompt/user-message sent to the model, rather than real model output.
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    if v in _PLACEHOLDER_VALUES:
        return True
    return any(v in source for source in echo_sources if source)


def _fails_required_fields(obs: dict) -> bool:
    """True if `sequence`/`finding`/`location` aren't all non-empty strings."""
    for key in ("sequence", "finding", "location"):
        value = obs.get(key)
        if not isinstance(value, str) or not value.strip():
            return True
    return False


def filter_hallucinated_observations(observations: list[dict],
                                     echo_sources: list[str]) -> tuple[list[dict], int]:
    """Drop observations that fail the strict schema or look hallucinated.

    An observation is dropped if any of:
      - a required field (sequence/finding/location) is missing or empty,
      - confidence doesn't normalize to one of engine.CONFIDENCE_LEVELS,
      - finding/location is an unfilled template placeholder or a verbatim
        echo of the prompt/user-message sent to the model.

    Returns (clean_observations, dropped_count).
    """
    clean = []
    dropped = 0
    for obs in observations:
        if (_fails_required_fields(obs)
                or normalize_confidence(obs.get("confidence")) is None
                or _is_prompt_echo(obs.get("finding"), echo_sources)
                or _is_prompt_echo(obs.get("location"), echo_sources)):
            dropped += 1
            continue
        clean.append(obs)
    return clean, dropped
