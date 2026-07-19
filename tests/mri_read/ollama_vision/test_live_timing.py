"""Opt-in, real-inference regression test for vision-call latency.

Investigation finding: a single per-series /api/chat call to llava:13b (4
images) measured 434.6s on this CPU-only host -- close enough to the 600s
default --vision-timeout that ordinary variance (more images, system load, a
longer reply) can and does exceed it. That's the actual cause behind "the
code doesn't crash, it just never finishes."

This test is opt-in (@pytest.mark.ollama, skipped unless a local Ollama
server is reachable -- see conftest.py) because it does real CPU inference:
expect minutes, not milliseconds, when it runs. Its job is to catch the
scenario getting WORSE (a bigger default image count, a heavier default
model) before someone finds out from a stuck real run instead.
"""

from __future__ import annotations

import time

import pytest

from mri_read.config import DEFAULT as CFG
from mri_read.engine import SeriesImages
from mri_read.ollama_vision import OllamaVisionEngine

pytestmark = pytest.mark.ollama


def test_single_series_vision_call_completes_within_the_configured_timeout():
    # A tiny synthetic 1x1 white PNG stands in for a real slice -- this test
    # is about round-trip/inference latency at the default image count (see
    # analyze's default `slices`), not about correctness of the analysis.
    tiny_png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
               b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
               b"\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa75\x81"
               b"\x84\x00\x00\x00\x00IEND\xaeB`\x82")
    series = SeriesImages(series="LiveTimingTest", label="T2", plane="Axial",
                          slice_indices=[0, 1, 2, 3], slice_pngs=[tiny_png] * 4)

    engine = OllamaVisionEngine(model=CFG.vision_model, host=CFG.ollama_host,
                                timeout=600, auto_pull=False)

    t0 = time.time()
    engine._analyze_one({"body_part": "BRAIN", "model": "TEST", "field_T": 3.0}, series)
    elapsed = time.time() - t0

    # Not "must be fast" -- CPU vision inference isn't. This flags it getting
    # WORSE than what was measured (434.6s for 4 real slices): if a synthetic
    # 1-pixel image set alone blows past the configured timeout, the model
    # or hardware situation has regressed further than what agent.py's
    # default --vision-timeout=600 was ever going to tolerate.
    assert elapsed < 600, (
        f"single vision call took {elapsed:.1f}s, >= the 600s default "
        "--vision-timeout -- this alone explains real-run timeouts"
    )
