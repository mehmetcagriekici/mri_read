from mri_read.engine import (CONFIDENCE_LEVELS, AnalysisResult, SeriesImages,
                             normalize_confidence)


def test_series_images_holds_given_fields():
    s = SeriesImages(series="Seri1", label="T2", plane="Axial",
                     slice_indices=[1, 2], slice_pngs=[b"a", b"b"])
    assert s.series == "Seri1"
    assert s.label == "T2"
    assert s.slice_indices == [1, 2]
    assert s.slice_pngs == [b"a", b"b"]


def test_series_images_acq_defaults_to_empty_dict():
    s = SeriesImages(series="Seri1", label="T2", plane="Axial",
                     slice_indices=[1], slice_pngs=[b"a"])
    assert s.acq == {}


def test_series_images_acq_holds_given_value():
    s = SeriesImages(series="Seri1", label="T2", plane="Axial",
                     slice_indices=[1], slice_pngs=[b"a"],
                     acq={"TE": 90.24, "TR": 7163.0})
    assert s.acq == {"TE": 90.24, "TR": 7163.0}


def test_series_images_acq_is_independent_per_instance():
    a = SeriesImages(series="a", label="T2", plane="Axial", slice_indices=[], slice_pngs=[])
    b = SeriesImages(series="b", label="T2", plane="Axial", slice_indices=[], slice_pngs=[])
    a.acq["TE"] = 1.0
    assert b.acq == {}


def test_analysis_result_defaults():
    r = AnalysisResult(engine="ollama:llava", sequences_reviewed=["T2"])
    assert r.observations == []
    assert r.impression == ""
    assert r.confidence == ""
    assert r.flags == []
    assert r.disclaimer == ""
    assert r.raw is None


def test_normalize_confidence_accepts_known_levels_case_insensitively():
    assert normalize_confidence("low") == "low"
    assert normalize_confidence("Moderate") == "moderate"
    assert normalize_confidence("HIGH") == "high"
    assert normalize_confidence("  high  ") == "high"


def test_normalize_confidence_rejects_unknown_values():
    assert normalize_confidence("uncertain") is None
    assert normalize_confidence("low|moderate|high") is None
    assert normalize_confidence("") is None
    assert normalize_confidence(None) is None
    assert normalize_confidence(3) is None


def test_confidence_levels_are_exactly_low_moderate_high():
    assert CONFIDENCE_LEVELS == ("low", "moderate", "high")


def test_analysis_result_default_lists_are_independent_per_instance():
    a = AnalysisResult(engine="e", sequences_reviewed=[])
    b = AnalysisResult(engine="e", sequences_reviewed=[])
    a.flags.append("x")
    assert b.flags == []
