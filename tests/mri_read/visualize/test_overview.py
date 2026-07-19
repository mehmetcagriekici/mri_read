from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mri_read.visualize.overview import overview


def _fake_series(name):
    vol = np.random.default_rng(0).normal(50, 10, (3, 8, 8)).astype(np.float32)
    tags = {"plane": "Axial", "scanning_sequence": "SE", "echo_time_TE": 90.0,
           "repetition_TR": 7000.0, "thickness_mm": 5.0, "protocol": "T2"}
    return SimpleNamespace(name=name, volume=vol, tags=tags, n_slices=3)


def test_overview_writes_one_montage_per_series(out_dir, capsys):
    with patch("mri_read.visualize.overview.list_series", return_value=["Seri1", "Seri2"]), \
         patch("mri_read.visualize.overview.load_series", side_effect=_fake_series):
        overview()

    assert (out_dir / "Seri1.png").exists()
    assert (out_dir / "Seri2.png").exists()
    out = capsys.readouterr().out
    assert "Seri1" in out and "Seri2" in out


def test_overview_continues_after_one_series_fails_to_load(out_dir, capsys):
    def loader(name):
        if name == "Seri1":
            raise RuntimeError("corrupt file")
        return _fake_series(name)

    with patch("mri_read.visualize.overview.list_series", return_value=["Seri1", "Seri2"]), \
         patch("mri_read.visualize.overview.load_series", side_effect=loader):
        overview()

    assert not (out_dir / "Seri1.png").exists()
    assert (out_dir / "Seri2.png").exists()
    assert "failed to load" in capsys.readouterr().out
