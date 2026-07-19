from mri_read.analyze.ranking import _rank_key


def test_qc_status_dominates_the_ranking():
    passing = {"label": "T2", "qc": {"status": "pass", "metrics": {"snr": 5.0}}}
    warning = {"label": "T2", "qc": {"status": "warn", "metrics": {"snr": 50.0}}}
    assert _rank_key(passing) > _rank_key(warning)


def test_dwi_prefers_more_bvalue_buckets_then_more_slices():
    two_bvalues = {"label": "DWI", "qc": {"status": "pass", "metrics": {"bvalue_buckets": 2}},
                  "n_slices": 27}
    one_bvalue = {"label": "DWI", "qc": {"status": "pass", "metrics": {"bvalue_buckets": 1}},
                 "n_slices": 100}
    assert _rank_key(two_bvalues) > _rank_key(one_bvalue)


def test_dwi_falls_back_to_counting_buckets_when_qc_missing(monkeypatch):
    from mri_read.analyze import ranking
    monkeypatch.setattr(ranking, "count_bvalue_buckets", lambda series: 3)
    row = {"label": "DWI", "series": "Seri1", "n_slices": 27}
    # no "qc" key at all -> status_rank defaults to the WORST rank (0), same
    # as an explicit "error" -- missing QC is not treated as an implicit pass.
    assert ranking._rank_key(row) == (0, 3, 27)


def test_3d_t1_prefers_thinner_slices():
    thin = {"label": "3D T1", "qc": {"status": "pass"}, "acq": {"thickness_mm": 1.0}}
    thick = {"label": "3D T1", "qc": {"status": "pass"}, "acq": {"thickness_mm": 3.0}}
    assert _rank_key(thin) > _rank_key(thick)


def test_3d_t1_missing_thickness_ranks_worst():
    known = {"label": "3D T1", "qc": {"status": "pass"}, "acq": {"thickness_mm": 5.0}}
    unknown = {"label": "3D T1", "qc": {"status": "pass"}, "acq": {}}
    assert _rank_key(known) > _rank_key(unknown)


def test_other_labels_prefer_higher_snr():
    high_snr = {"label": "T2", "qc": {"status": "pass", "metrics": {"snr": 50.0}}}
    low_snr = {"label": "T2", "qc": {"status": "pass", "metrics": {"snr": 5.0}}}
    assert _rank_key(high_snr) > _rank_key(low_snr)


def test_unmeasured_snr_treated_as_borderline_not_worst():
    unmeasured = {"label": "T2", "qc": {"status": "pass", "metrics": {"snr": None}}}
    clearly_bad = {"label": "T2", "qc": {"status": "pass", "metrics": {"snr": 1.0}}}
    assert _rank_key(unmeasured) > _rank_key(clearly_bad)
