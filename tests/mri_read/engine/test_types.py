from mri_read.engine import AnalysisResult, SeriesImages


def test_series_images_holds_given_fields():
    s = SeriesImages(series="Seri1", label="T2", plane="Axial",
                     slice_indices=[1, 2], slice_pngs=[b"a", b"b"])
    assert s.series == "Seri1"
    assert s.label == "T2"
    assert s.slice_indices == [1, 2]
    assert s.slice_pngs == [b"a", b"b"]


def test_analysis_result_defaults():
    r = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"])
    assert r.observations == []
    assert r.impression == ""
    assert r.flags == []
    assert r.disclaimer == ""
    assert r.raw is None


def test_analysis_result_default_lists_are_independent_per_instance():
    a = AnalysisResult(engine="e", sequences_reviewed=[])
    b = AnalysisResult(engine="e", sequences_reviewed=[])
    a.flags.append("x")
    assert b.flags == []
