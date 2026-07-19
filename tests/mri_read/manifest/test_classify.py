from mri_read.manifest.classify import PRIMARY, classify


def _tags(**overrides):
    base = {"scanning_sequence": "", "echo_time_TE": None,
           "repetition_TR": None, "thickness_mm": None}
    base.update(overrides)
    return base


def test_echo_planar_is_dwi():
    r = classify(_tags(scanning_sequence="['EP', 'SE']", echo_time_TE=85.4), n_slices=27)
    assert r["label"] == "DWI"


def test_single_gr_slice_is_localizer():
    r = classify(_tags(scanning_sequence="GR"), n_slices=1)
    assert r["label"] == "Localizer/stray"


def test_gr_with_zero_thickness_is_reformat():
    r = classify(_tags(scanning_sequence="GR", thickness_mm=0.0), n_slices=12)
    assert r["label"] == "Reformat (MPR)"


def test_gr_thin_and_many_slices_is_3d_t1():
    r = classify(_tags(scanning_sequence="GR", thickness_mm=1.39), n_slices=152)
    assert r["label"] == "3D T1"


def test_gr_thick_or_few_slices_is_generic_gre():
    r = classify(_tags(scanning_sequence="GR", thickness_mm=5.0), n_slices=30)
    assert r["label"] == "GRE (T1-ish)"


def test_ir_with_long_te_is_flair():
    r = classify(_tags(scanning_sequence="IR", echo_time_TE=96.36), n_slices=27)
    assert r["label"] == "T2 FLAIR"


def test_ir_with_short_te_is_t1_ir():
    r = classify(_tags(scanning_sequence="IR", echo_time_TE=9.4), n_slices=27)
    assert r["label"] == "T1 (IR)"


def test_long_te_long_tr_is_t2():
    r = classify(_tags(echo_time_TE=122.0, repetition_TR=7316.0), n_slices=27)
    assert r["label"] == "T2"


def test_short_te_short_tr_is_t1():
    r = classify(_tags(echo_time_TE=10.0, repetition_TR=500.0), n_slices=27)
    assert r["label"] == "T1"


def test_short_te_long_tr_is_pd_t1():
    r = classify(_tags(echo_time_TE=10.0, repetition_TR=2000.0), n_slices=27)
    assert r["label"] == "PD/T1"


def test_no_usable_tags_is_unknown():
    r = classify(_tags(), n_slices=10)
    assert r["label"] == "Unknown"


def test_reason_is_populated_and_confidence_in_range():
    r = classify(_tags(scanning_sequence="['EP', 'SE']", echo_time_TE=85.4), n_slices=27)
    assert r["reason"]
    assert 0.0 <= r["confidence"] <= 1.0


def test_primary_set_matches_the_diagnostic_labels():
    assert PRIMARY == {"DWI", "T2 FLAIR", "T2", "T1", "T1 (IR)", "3D T1"}
    assert "Reformat (MPR)" not in PRIMARY


def test_zero_slices_does_not_crash():
    r = classify(_tags(scanning_sequence="GR"), n_slices=0)
    assert r["label"]  # some label, no exception


def test_negative_thickness_does_not_crash():
    r = classify(_tags(scanning_sequence="GR", thickness_mm=-1.0), n_slices=100)
    assert r["label"]


def test_absurdly_large_slice_count_does_not_crash():
    r = classify(_tags(scanning_sequence="GR", thickness_mm=1.0), n_slices=10_000_000)
    assert r["label"] == "3D T1"  # still matches the thin+many-slices rule


def test_unicode_in_scanning_sequence_does_not_crash():
    r = classify(_tags(scanning_sequence="日本語 EP テスト", echo_time_TE=85.4), n_slices=27)
    assert r["label"] == "DWI"  # "EP" substring still matches despite surrounding unicode
    assert "Localizer/stray" not in PRIMARY
