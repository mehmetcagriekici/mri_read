import pytest

from mri_read.engine import AnalysisEngine


def test_analysis_engine_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AnalysisEngine()


def test_subclass_must_implement_analyze():
    class Incomplete(AnalysisEngine):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_implementing_analyze_can_be_instantiated():
    class Complete(AnalysisEngine):
        name = "fake"

        def analyze(self, study_meta, series):
            return None

    Complete()  # must not raise
