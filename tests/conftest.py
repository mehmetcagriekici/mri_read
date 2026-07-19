"""Shared fixtures.

Most tests are pure unit tests: no disk, no network, everything mocked. Two
opt-in categories exist for the parts that can't be meaningfully tested that
way (see pyproject.toml's [tool.pytest.ini_options] markers):
  - @pytest.mark.data   : needs mri_test_data/ on disk (gitignored). Skipped
                          automatically when the folder is absent -- this is
                          about missing fixture data, not about being slow,
                          so "skip if the environment can't support it" is
                          the right default.
  - @pytest.mark.ollama : needs a live local Ollama server with models
                          pulled. Real CPU inference -- MINUTES per test
                          (one measured run: 434.6s for a single call). This
                          must stay opt-in via `--run-ollama` regardless of
                          whether a server happens to be reachable --
                          checking reachability alone would make a bare
                          `pytest` silently take 5+ minutes on any machine
                          that happens to have `ollama serve` running, which
                          defeats the point of a fast default test run.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from mri_read.paths import DATA_DIR

HAS_TEST_DATA = DATA_DIR.exists() and any(DATA_DIR.iterdir())


def _ollama_reachable(host: str = "http://localhost:11434") -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def pytest_addoption(parser):
    parser.addoption("--run-ollama", action="store_true", default=False,
                     help="also run @pytest.mark.ollama tests (real CPU "
                          "inference against a live local Ollama server; "
                          "minutes, not seconds, per test)")


def pytest_collection_modifyitems(config, items):
    skip_data = pytest.mark.skip(reason="mri_test_data/ not present on disk")
    run_ollama = config.getoption("--run-ollama")
    skip_ollama = pytest.mark.skip(
        reason="ollama-marked tests are opt-in: pass --run-ollama "
               "(needs a live local Ollama server with models pulled)")
    skip_ollama_unreachable = pytest.mark.skip(reason="no reachable local Ollama server")

    for item in items:
        if "data" in item.keywords and not HAS_TEST_DATA:
            item.add_marker(skip_data)
        if "ollama" in item.keywords:
            if not run_ollama:
                item.add_marker(skip_ollama)
            elif not _ollama_reachable():
                item.add_marker(skip_ollama_unreachable)


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """Redirect mri_read.paths.OUT to a throwaway directory for this test.

    Report-writing code (analyze/report_*.py, visualize/*.py) reads OUT at
    call time via `from mri_read.paths import OUT`, so patching the OUT name
    in each module that imported it is what actually takes effect.

    Submodules are fetched via importlib rather than `import a.b.c as x`:
    several of these packages' __init__.py does
    `from pkg.module import func_with_same_name_as_module`, which overwrites
    the submodule attribute on the package with the function of the same
    name -- `import a.b.c as x` resolves through that (now-shadowed)
    attribute and would silently bind x to the function instead of the
    module. importlib.import_module goes through sys.modules directly and
    isn't affected.
    """
    import importlib

    paths_pkg = importlib.import_module("mri_read.paths")
    report_json = importlib.import_module("mri_read.analyze.report_json")
    report_markdown = importlib.import_module("mri_read.analyze.report_markdown")
    overview = importlib.import_module("mri_read.visualize.overview")
    deep_dive = importlib.import_module("mri_read.visualize.deep_dive")

    monkeypatch.setattr(paths_pkg, "OUT", tmp_path)
    monkeypatch.setattr(report_json, "OUT", tmp_path)
    monkeypatch.setattr(report_markdown, "OUT", tmp_path)
    monkeypatch.setattr(overview, "OUT", tmp_path)
    monkeypatch.setattr(deep_dive, "OUT", tmp_path)
    return tmp_path
