from unittest.mock import patch

from mri_read.manifest.build import build_manifest


def _info(name, n_slices, **tags):
    base = {"body_part": "BRAIN", "manufacturer": "GE", "model": "SIGNA",
           "field_T": 3.0, "scanning_sequence": "", "echo_time_TE": None,
           "repetition_TR": None, "inversion_TI": None, "thickness_mm": None,
           "plane": "Axial"}
    base.update(tags)
    return {"name": name, "n_slices": n_slices, "tags": base}


def test_build_manifest_captures_study_info_once():
    infos = [_info("Seri1", 27, scanning_sequence="EP", echo_time_TE=85.4),
            _info("Seri2", 27, scanning_sequence="EP", echo_time_TE=85.4)]
    with patch("mri_read.manifest.build.list_series", return_value=["Seri1", "Seri2"]), \
         patch("mri_read.manifest.build.inspect_series", side_effect=infos):
        m = build_manifest()

    assert m["study"] == {"body_part": "BRAIN", "manufacturer": "GE",
                          "model": "SIGNA", "field_T": 3.0}
    assert len(m["series"]) == 2


def test_build_manifest_marks_use_for_analysis_from_primary_set():
    infos = [_info("Seri1", 27, scanning_sequence="EP", echo_time_TE=85.4),  # DWI -> primary
            _info("Seri2", 12, scanning_sequence="GR", thickness_mm=0.0)]  # reformat -> not
    with patch("mri_read.manifest.build.list_series", return_value=["Seri1", "Seri2"]), \
         patch("mri_read.manifest.build.inspect_series", side_effect=infos):
        m = build_manifest()

    by_name = {row["series"]: row for row in m["series"]}
    assert by_name["Seri1"]["use_for_analysis"] is True
    assert by_name["Seri1"]["label"] == "DWI"
    assert by_name["Seri2"]["use_for_analysis"] is False
    assert by_name["Seri2"]["label"] == "Reformat (MPR)"


def test_build_manifest_keeps_raw_acquisition_numbers_for_audit():
    infos = [_info("Seri1", 27, scanning_sequence="EP", echo_time_TE=85.4)]
    with patch("mri_read.manifest.build.list_series", return_value=["Seri1"]), \
         patch("mri_read.manifest.build.inspect_series", side_effect=infos):
        m = build_manifest()

    assert m["series"][0]["acq"]["TE"] == 85.4
    assert m["series"][0]["acq"]["scanning_sequence"] == "EP"


def test_build_manifest_handles_no_series():
    with patch("mri_read.manifest.build.list_series", return_value=[]):
        m = build_manifest()

    assert m == {"study": {}, "series": []}
