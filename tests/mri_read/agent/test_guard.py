"""agent.guard: deterministic multi-sequence correlation + overconfident-
language suppression. No LLM involved -- every case here is pure logic.

Design stance under test: an unsupported diagnostic-sounding claim must
never survive unqualified into the report. Silence ("not enough evidence")
is fine; a false claim is not.
"""

from __future__ import annotations

from mri_read.agent.guard import (REDACTED_FINDING, apply_correlation_guard,
                                  guard_final_impression)


def _obs(sequence, finding, location="frontal lobe", confidence="high"):
    return {"sequence": sequence, "finding": finding, "location": location,
           "confidence": confidence}


# --- apply_correlation_guard -------------------------------------------------

def test_no_concerning_language_passes_through_unchanged():
    obs = [_obs("T2", "mild signal asymmetry noted")]
    adjusted, flags = apply_correlation_guard(obs)
    assert adjusted == obs
    assert flags == []


def test_uncorroborated_concerning_finding_is_suppressed():
    obs = [_obs("T2 FLAIR", "large mass lesion, this appears to be a tumor",
               location="right frontal lobe")]
    adjusted, flags = apply_correlation_guard(obs)
    assert adjusted[0]["finding"] == REDACTED_FINDING
    assert adjusted[0]["confidence"] == "low"
    assert len(flags) == 1
    assert "T2 FLAIR" in flags[0]
    assert "tumor" in flags[0]
    assert "not confirmed" in flags[0]


def test_original_wording_is_preserved_in_the_flag():
    obs = [_obs("T2 FLAIR", "this appears to be a tumor in the right frontal lobe")]
    _, flags = apply_correlation_guard(obs)
    assert "this appears to be a tumor" in flags[0]


def test_corroborated_finding_across_two_sequences_is_kept():
    obs = [
        _obs("T2 FLAIR", "possible mass lesion, tumor suspected", location="right frontal lobe"),
        _obs("T1 (IR)", "corresponding mass consistent with tumor", location="frontal lobe, right side"),
    ]
    adjusted, flags = apply_correlation_guard(obs)
    assert adjusted[0]["finding"] != REDACTED_FINDING
    assert adjusted[1]["finding"] != REDACTED_FINDING
    assert flags == []


def test_same_term_different_locations_is_not_corroboration():
    obs = [
        _obs("T2 FLAIR", "tumor in right frontal lobe", location="right frontal lobe"),
        _obs("T1 (IR)", "tumor in left occipital lobe", location="left occipital lobe"),
    ]
    adjusted, flags = apply_correlation_guard(obs)
    assert adjusted[0]["finding"] == REDACTED_FINDING
    assert adjusted[1]["finding"] == REDACTED_FINDING
    assert len(flags) == 2


def test_two_findings_from_the_same_sequence_do_not_corroborate_each_other():
    obs = [
        _obs("T2 FLAIR", "tumor noted here", location="frontal lobe"),
        _obs("T2 FLAIR", "tumor noted there too", location="frontal lobe"),
    ]
    adjusted, flags = apply_correlation_guard(obs)
    assert adjusted[0]["finding"] == REDACTED_FINDING
    assert adjusted[1]["finding"] == REDACTED_FINDING


def test_plural_and_suffixed_forms_are_caught():
    obs = [_obs("T2", "multiple tumors visible")]
    adjusted, flags = apply_correlation_guard(obs)
    assert adjusted[0]["finding"] == REDACTED_FINDING


def test_clean_and_concerning_findings_are_handled_independently():
    obs = [
        _obs("T2", "mild signal asymmetry"),
        _obs("T2 FLAIR", "this appears to be a tumor", location="frontal lobe"),
    ]
    adjusted, flags = apply_correlation_guard(obs)
    assert adjusted[0]["finding"] == "mild signal asymmetry"  # untouched
    assert adjusted[1]["finding"] == REDACTED_FINDING
    assert len(flags) == 1


def test_empty_observations_list():
    assert apply_correlation_guard([]) == ([], [])


# --- guard_final_impression --------------------------------------------------

MANIFEST = {"series": [
    {"label": "T2", "use_for_analysis": True},
    {"label": "T2 FLAIR", "use_for_analysis": True},
    {"label": "3D T1", "use_for_analysis": True},
    {"label": "Reformat (MPR)", "use_for_analysis": False},
]}


def test_clean_impression_with_no_observations_gets_low_confidence():
    text, confidence, flags = guard_final_impression("No acute findings.", [], MANIFEST)
    assert text == "No acute findings."
    assert confidence == "low"  # nothing to be confident about
    assert flags == []


def test_confidence_rolls_up_from_observations():
    obs = [_obs("T2", "clean", confidence="high"), _obs("T2 FLAIR", "clean", confidence="high")]
    _, confidence, _ = guard_final_impression("No acute findings.", obs, MANIFEST)
    assert confidence == "high"


def test_one_low_confidence_observation_caps_overall_confidence():
    obs = [_obs("T2", "clean", confidence="high"), _obs("T2 FLAIR", "clean", confidence="low")]
    _, confidence, _ = guard_final_impression("No acute findings.", obs, MANIFEST)
    assert confidence == "low"


def test_mixed_high_and_moderate_rolls_up_to_moderate():
    obs = [_obs("T2", "clean", confidence="high"), _obs("T2 FLAIR", "clean", confidence="moderate")]
    _, confidence, _ = guard_final_impression("No acute findings.", obs, MANIFEST)
    assert confidence == "moderate"


def test_unsupported_diagnostic_term_in_impression_is_redacted():
    text, confidence, flags = guard_final_impression(
        "The scan shows a tumor in the frontal lobe.", [], MANIFEST)
    assert "tumor" not in text.lower()
    assert "[unconfirmed finding]" in text
    assert confidence == "low"
    assert any("tumor" in f for f in flags)


def test_term_backed_by_a_corroborated_observation_is_not_redacted():
    """If the observations already earned the term (survived
    apply_correlation_guard, i.e. wasn't suppressed), the synthesis text
    is allowed to use it too.
    """
    obs = [_obs("T2 FLAIR", "corroborated tumor finding")]  # NOT the REDACTED_FINDING text
    text, confidence, flags = guard_final_impression(
        "The scan shows a tumor in the frontal lobe.", obs, MANIFEST)
    assert "tumor" in text.lower()
    assert not any("tumor" in f for f in flags)


def test_term_from_a_suppressed_observation_does_not_count_as_corroborated():
    from mri_read.agent.guard import REDACTED_FINDING as RF
    obs = [{"sequence": "T2 FLAIR", "finding": RF, "location": "frontal lobe", "confidence": "low"}]
    text, confidence, flags = guard_final_impression(
        "The scan shows a tumor in the frontal lobe.", obs, MANIFEST)
    assert "[unconfirmed finding]" in text
    assert confidence == "low"


def test_impression_mentioning_an_unreviewed_sequence_is_flagged():
    text, confidence, flags = guard_final_impression(
        "Reformat (MPR) shows no abnormality.", [], MANIFEST)
    assert any("Reformat (MPR)" in f and "not actually analyzed" in f for f in flags)
    assert confidence == "low"


def test_impression_mentioning_only_reviewed_sequences_is_not_flagged():
    text, confidence, flags = guard_final_impression(
        "T2 FLAIR and 3D T1 show no abnormality.", [], MANIFEST)
    assert flags == []


def test_multiple_concerning_terms_are_all_redacted():
    text, confidence, flags = guard_final_impression(
        "Findings suggest a possible tumor and cannot rule out malignancy.", [], MANIFEST)
    assert "tumor" not in text.lower()
    assert "malignan" not in text.lower()
    assert len(flags) == 2


def test_non_string_impression_does_not_crash():
    text, confidence, flags = guard_final_impression(None, [], MANIFEST)
    assert text == ""
    assert confidence == "low"


def test_edited_impression_gets_a_visible_disclaimer_prefix():
    text, _, _ = guard_final_impression("Shows a tumor.", [], MANIFEST)
    assert text.startswith("Note: this summary has been edited")
