"""Deterministic hallucination guard: multi-sequence correlation and
overconfident-language suppression for the agent's final report.

No LLM call here on purpose -- asking a model to judge another model's
hallucinations just compounds the risk, not reduces it. This is plain,
testable Python logic over the manifest and the (already schema-validated,
see ollama_vision.sanitize) observations.

Design stance: it is fine for the report to say nothing about a finding --
it is NOT fine for it to state an unsupported diagnostic-sounding claim as
if it were established. Uncorroborated concerning findings are SUPPRESSED
(the claim text is replaced with an explicit "unconfirmed" marker), not
merely flagged with a lower confidence number sitting next to the original
claim -- a caveat next to the word "tumor" is too easy to miss on a skim.
The original wording is preserved only inside a flag, clearly marked as
unverified, so nothing is silently lost for a human reviewer who wants it.

Two passes, run from agent.pipeline:
  1. apply_correlation_guard()  -- pre-synthesis, over vision observations.
  2. guard_final_impression()   -- post-synthesis, over the synthesized text.
"""

from __future__ import annotations

import re

from mri_read.engine import normalize_confidence

# Diagnostic-CONCLUSION language, not general radiological description.
# Words like "lesion", "abnormality", "asymmetry", "mass effect" are normal
# descriptive vocabulary for what's VISIBLE and are deliberately NOT in this
# list. These specific terms assert a diagnosis outright -- exactly what a
# research prototype must never do without strong (multi-sequence) evidence.
CONCERNING_TERMS = (
    "tumor", "tumour", "cancer", "carcinoma", "malignant", "malignancy",
    "neoplasm", "neoplastic", "metastasis", "metastases", "metastatic",
    "sarcoma", "glioma", "lymphoma",
)

REDACTED_FINDING = ("Uncorroborated diagnostic-sounding claim withheld from "
                    "this observation — not confirmed by any other sequence. "
                    "See flags for the original wording.")

_STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "region", "area", "side",
             "lobe", "lobes", "brain", "cerebral", "cerebrum"}


def _term_pattern(term: str) -> re.Pattern:
    """A pattern matching `term` plus any trailing word characters.

    Deliberately no TRAILING \\b right after `term`: that would miss plain
    plurals ("tumors") and other suffixed forms ("malignantly") -- a false
    negative here means an unsupported claim slips through uncaught, which
    matters more than the small risk of over-matching a rare compound word.
    The trailing \\w* also keeps text substitution clean (consumes the whole
    inflected word, not just its stem).
    """
    return re.compile(rf"\b{re.escape(term)}\w*", re.IGNORECASE)


def _mentions_concerning_term(text: object) -> str | None:
    """Return the first concerning term found in `text` (case-insensitive,
    matching plurals/suffixed forms too), or None.
    """
    if not isinstance(text, str) or not text:
        return None
    for term in CONCERNING_TERMS:
        if _term_pattern(term).search(text):
            return term
    return None


def _locations_overlap(a: object, b: object) -> bool:
    """Loose overlap check between two location strings.

    Deliberately permissive -- this only needs to catch variants like
    "right frontal lobe" vs. "frontal lobe (right)", not do real anatomical
    NLP. A shared non-stopword is treated as overlap.
    """
    if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip():
        return False
    words_a = set(re.findall(r"[a-z]+", a.lower())) - _STOPWORDS
    words_b = set(re.findall(r"[a-z]+", b.lower())) - _STOPWORDS
    return bool(words_a & words_b)


def apply_correlation_guard(observations: list[dict]) -> tuple[list[dict], list[str]]:
    """Suppress concerning findings that no other sequence corroborates.

    A finding is corroborated if ANOTHER observation, from a DIFFERENT
    sequence, also mentions a concerning term AND references an overlapping
    location. Corroborated findings are left untouched entirely -- real
    cross-sequence evidence is exactly what this check exists to reward,
    not penalize.

    Returns (adjusted_observations, extra_flags).
    """
    adjusted = []
    flags: list[str] = []
    for i, obs in enumerate(observations):
        term = _mentions_concerning_term(obs.get("finding"))
        if term is None:
            adjusted.append(obs)
            continue

        corroborated = any(
            j != i
            and other.get("sequence") != obs.get("sequence")
            and _mentions_concerning_term(other.get("finding")) is not None
            and _locations_overlap(obs.get("location"), other.get("location"))
            for j, other in enumerate(observations)
        )
        if corroborated:
            adjusted.append(obs)
            continue

        sequence = obs.get("sequence", "?")
        original_finding = obs.get("finding", "")
        flags.append(
            f"{sequence}: suppressed an uncorroborated {term!r} claim "
            f"(original: {original_finding[:120]!r}) — not confirmed by any other sequence"
        )
        suppressed = dict(obs)
        suppressed["finding"] = REDACTED_FINDING
        suppressed["confidence"] = "low"
        adjusted.append(suppressed)

    return adjusted, flags


def _overall_confidence_from(observations: list[dict]) -> str:
    """Roll up per-observation confidence into one overall level.

    No observations, or none with a valid confidence, -> "low" (nothing to
    be confident about). All "high" -> "high". Any "low" -> "low" (one weak
    link is enough to cap the whole report). Otherwise "moderate".
    """
    levels = [normalize_confidence(o.get("confidence")) for o in observations]
    levels = [lv for lv in levels if lv]
    if not levels:
        return "low"
    if any(lv == "low" for lv in levels):
        return "low"
    if all(lv == "high" for lv in levels):
        return "high"
    return "moderate"


def guard_final_impression(impression: str, observations: list[dict],
                           manifest: dict) -> tuple[str, str, list[str]]:
    """Post-synthesis check on the FINAL synthesized impression text.

    Word-level substitution rather than full-sentence redaction here: the
    impression is free prose from a different model than the one that
    produced the observations, so sentence structure can't be assumed the
    way apply_correlation_guard can for a single JSON field.

    Also checks the impression against the manifest for a sequence being
    discussed that was never actually analyzed (QC-failed, a reformat, or
    simply not selected) -- the same "hallucinated a sequence" failure mode
    ollama_vision.engine_impl's ground-truth-label fix guards against, but
    here specifically for what the TEXT model claims in its own prose.

    Returns (adjusted_impression, overall_confidence, extra_flags).
    """
    if not isinstance(impression, str):
        impression = ""

    # Terms already backed by a corroborated (i.e. NOT suppressed) observation
    # are allowed to stand in the prose -- real cross-sequence evidence earns
    # the language. Everything else gets redacted, regardless of how the text
    # model phrased it.
    corroborated_terms = {
        _mentions_concerning_term(obs.get("finding"))
        for obs in observations
        if obs.get("finding") != REDACTED_FINDING
        and _mentions_concerning_term(obs.get("finding")) is not None
    }

    flags: list[str] = []
    adjusted = impression
    for term in CONCERNING_TERMS:
        if term in corroborated_terms:
            continue
        pattern = _term_pattern(term)
        if pattern.search(adjusted):
            adjusted = pattern.sub("[unconfirmed finding]", adjusted)
            flags.append(
                f"impression: replaced an unsupported {term!r} claim with "
                "'[unconfirmed finding]' — not corroborated across sequences"
            )

    # Consistency check: does the impression discuss a sequence that was
    # never actually analyzed? Sequence labels are a small, specific
    # vocabulary (e.g. "T2 FLAIR", "3D T1"), so a substring check is safe --
    # unlike free-text claims, there's no risk of matching ordinary prose.
    reviewed = {row["label"] for row in manifest.get("series", []) if row.get("use_for_analysis")}
    all_labels = {row["label"] for row in manifest.get("series", []) if row.get("label")}
    for label in sorted(all_labels - reviewed):
        # (?!\w) rather than a trailing \b: \b never matches right after a
        # non-word character (e.g. the ")" ending "Reformat (MPR)"), which
        # would silently make that label impossible to ever detect.
        if re.search(rf"\b{re.escape(label)}(?!\w)", adjusted, re.IGNORECASE):
            flags.append(
                f"impression: mentions {label!r}, a sequence that was not "
                "actually analyzed — likely hallucinated"
            )

    if flags:
        adjusted = ("Note: this summary has been edited to remove one or more "
                   "unsupported claims not corroborated across sequences "
                   "(see flags for details). " + adjusted)
        overall_confidence = "low"  # something needed correcting -> never "high"
    else:
        overall_confidence = _overall_confidence_from(observations)

    return adjusted, overall_confidence, flags
