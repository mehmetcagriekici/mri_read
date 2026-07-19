"""filter_hallucinated_observations: strict schema enforcement plus catching
well-formed-JSON hallucinations that parse_json_reply's syntax check can't
catch -- see the real incident this guards against: llava:13b echoed its own
prompt's example schema back as if it were a genuine observation.
"""

from __future__ import annotations

from mri_read.ollama_vision.sanitize import filter_hallucinated_observations

SYSTEM_PROMPT = 'Return JSON like {"finding": "...", "confidence": "low|moderate|high"}'
USER_MESSAGE = "=== T2 FLAIR (Axial, Seri6) — slice indices [0, 8, 17, 26] ==="
SOURCES = [SYSTEM_PROMPT, USER_MESSAGE]


def test_real_observation_is_kept():
    obs = [{"sequence": "T2 FLAIR", "finding": "increased signal in the right frontal lobe",
           "location": "right frontal lobe", "confidence": "moderate"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == obs
    assert dropped == 0


def test_ellipsis_placeholder_finding_is_dropped():
    obs = [{"sequence": "T2 FLAIR", "finding": "...", "location": "...",
           "confidence": "low|moderate|high"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_confidence_enum_placeholder_alone_is_dropped():
    obs = [{"sequence": "T2", "finding": "a real-looking finding",
           "location": "frontal lobe", "confidence": "low|moderate|high"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_finding_echoing_the_user_message_header_is_dropped():
    """The real incident: the model returned "Axial, Seri6" as a finding and
    "slice indices [0, 8, 17, 26]" as a location -- both literal substrings
    of the per-call prompt header, not a description of the images.
    """
    obs = [{"sequence": "T2 FLAIR", "finding": "Axial, Seri6",
           "location": "slice indices [0, 8, 17, 26]", "confidence": "moderate"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_finding_echoing_the_system_prompt_is_dropped():
    obs = [{"sequence": "T2", "finding": SYSTEM_PROMPT, "location": "brain",
           "confidence": "high"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_mixed_real_and_hallucinated_observations_only_drops_the_bad_one():
    obs = [
        {"sequence": "T2", "finding": "real finding here", "location": "temporal lobe",
         "confidence": "high"},
        {"sequence": "T2", "finding": "...", "location": "...", "confidence": "low|moderate|high"},
    ]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert len(clean) == 1
    assert clean[0]["finding"] == "real finding here"
    assert dropped == 1


def test_non_string_confidence_is_dropped_not_guessed():
    """A missing/unreadable confidence must not silently pass through --
    "no claim" is fine, guessing a confidence level is not.
    """
    obs = [{"sequence": "T2", "finding": "ok", "location": "ok", "confidence": None}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_missing_fields_are_dropped_not_crashed_on():
    obs = [{"sequence": "T2"}]  # no finding/location/confidence at all
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_empty_observations_list():
    assert filter_hallucinated_observations([], SOURCES) == ([], 0)


# --- strict schema enforcement (required fields + confidence enum) ---------

def test_missing_sequence_is_dropped():
    obs = [{"finding": "real finding", "location": "brain", "confidence": "high"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_empty_string_location_is_dropped():
    obs = [{"sequence": "T2", "finding": "real finding", "location": "   ",
           "confidence": "high"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_unrecognized_confidence_value_is_dropped():
    obs = [{"sequence": "T2", "finding": "real finding", "location": "brain",
           "confidence": "uncertain"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == []
    assert dropped == 1


def test_confidence_is_case_insensitive():
    obs = [{"sequence": "T2", "finding": "real finding", "location": "brain",
           "confidence": "Moderate"}]
    clean, dropped = filter_hallucinated_observations(obs, SOURCES)
    assert clean == obs
    assert dropped == 0
