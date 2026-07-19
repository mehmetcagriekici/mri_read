from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mri_read.visualize.deep_dive import deep_dive, deep_dive_all


def _fake_series(name, n=3):
    vol = np.random.default_rng(0).normal(50, 10, (n, 8, 8)).astype(np.float32)
    return SimpleNamespace(name=name, volume=vol, n_slices=n)


def test_deep_dive_exports_one_png_per_slice(out_dir, capsys):
    with patch("mri_read.visualize.deep_dive.load_series", return_value=_fake_series("Seri1")):
        deep_dive("Seri1")

    folder = out_dir / "Seri1_slices"
    assert folder.is_dir()
    assert sorted(p.name for p in folder.glob("*.png")) == ["000.png", "001.png", "002.png"]


def test_deep_dive_all_skips_series_that_fail(out_dir, capsys):
    def loader(name):
        if name == "Seri2":
            raise RuntimeError("bad series")
        return _fake_series(name)

    with patch("mri_read.visualize.deep_dive.list_series", return_value=["Seri1", "Seri2"]), \
         patch("mri_read.visualize.deep_dive.load_series", side_effect=loader):
        deep_dive_all()

    assert sorted((out_dir / "Seri1_slices").glob("*.png"))
    # deep_dive() creates the {name}_slices folder BEFORE calling
    # load_series(), so a failed series still leaves an empty folder behind
    # -- just with no PNGs in it. Pinning this rather than asserting the
    # folder is absent, since that's what the code actually does.
    assert (out_dir / "Seri2_slices").is_dir()
    assert not list((out_dir / "Seri2_slices").glob("*.png"))
    assert "skipped" in capsys.readouterr().out
