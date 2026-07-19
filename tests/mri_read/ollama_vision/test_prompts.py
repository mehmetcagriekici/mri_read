from mri_read.ollama_vision.prompts import SYSTEM


def test_system_prompt_forbids_patient_instructions():
    assert "NOT clinical care" in SYSTEM
    assert "patient" in SYSTEM.lower()


def test_system_prompt_specifies_the_expected_json_shape():
    assert '"sequences_reviewed"' in SYSTEM
    assert '"observations"' in SYSTEM
    assert '"impression"' in SYSTEM
    assert '"flags"' in SYSTEM
