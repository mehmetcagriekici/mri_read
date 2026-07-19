"""Detecting and filtering hallucinated placeholder-echo content.

llava:13b has been observed returning syntactically valid JSON that still
isn't a real observation -- it literally echoes its own system prompt's
example schema (e.g. "finding": "...", "confidence": "low|moderate|high")
or the per-call user message's formatting string, instead of describing
what's actually in the images. parse_json_reply can't catch this (the JSON
is well-formed); this is a separate, semantic check.
"""

from __future__ import annotations

_PLACEHOLDER_VALUES = {"...", "low|moderate|high"}


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


def filter_hallucinated_observations(observations: list[dict],
                                     echo_sources: list[str]) -> tuple[list[dict], int]:
    """Drop observations whose finding/location/confidence look like an
    unfilled template or a verbatim echo of the prompt itself.

    Returns (clean_observations, dropped_count).
    """
    clean = []
    dropped = 0
    for obs in observations:
        confidence = obs.get("confidence")
        confidence_is_placeholder = (isinstance(confidence, str)
                                     and confidence.strip() in _PLACEHOLDER_VALUES)
        if (_is_prompt_echo(obs.get("finding"), echo_sources)
                or _is_prompt_echo(obs.get("location"), echo_sources)
                or confidence_is_placeholder):
            dropped += 1
            continue
        clean.append(obs)
    return clean, dropped
