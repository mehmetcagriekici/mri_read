from mri_read.agent.prompts import SYNTH_SYSTEM


def test_synth_system_forbids_diagnosis_language():
    assert "NOT clinical care" in SYNTH_SYSTEM or "not clinical" in SYNTH_SYSTEM.lower()
    assert "diagnosis" in SYNTH_SYSTEM.lower()


def test_synth_system_specifies_the_expected_json_shape():
    assert '"impression"' in SYNTH_SYSTEM
    assert '"flags"' in SYNTH_SYSTEM
