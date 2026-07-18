"""
Step 3b — Deterministic quality control.

Before any analysis, flag series that are geometrically or visually suspect so
the engine (and you) know how much to trust them. No AI — plain measurements.

Checks per series:
  - slice count vs InstanceNumber range  -> missing slices
  - spacing between slices               -> uneven / irregular geometry
  - foreground contrast                  -> low-contrast (flat) volume
  - background noise vs signal (SNR)     -> noisy volume
  - fraction of near-empty slices        -> mostly-empty acquisition

Augments output/manifest.json with a "qc" block per series.

CLI entry point: src/cmd/qc.py
"""

from __future__ import annotations

import numpy as np
import pydicom

from mri_read.mri import (DATA_DIR, _slice_position, foreground_fraction,
                          load_series, volume_window_bounds)


def _positions_and_instances(name: str):
    """Return (positions, instance_numbers) for a series, geometrically sorted.

    Header-only read. `positions` are mm along the slice normal (for the
    spacing check); `instances` are the scanner slice indices (for the
    missing-slice check).
    """
    files = sorted((DATA_DIR / name).glob("*.dcm"))
    heads = []
    for f in files:
        try:
            heads.append(pydicom.dcmread(str(f), stop_before_pixels=True,
                                         force=True))
        except Exception:                            # noqa: BLE001 - skip bad file
            continue
    heads.sort(key=_slice_position)                  # anatomical order
    positions = [_slice_position(h) for h in heads]
    instances = [int(getattr(h, "InstanceNumber", 0)) for h in heads]
    return positions, instances


def _background_snr(volume: np.ndarray) -> float:
    """Rough signal-to-noise ratio of the middle slice.

    Trick: the four image CORNERS are (almost) always empty background, so their
    standard deviation estimates the noise floor. Signal = mean intensity of the
    foreground (tissue). SNR = signal / noise. A low value means a grainy scan.
    """
    z = volume.shape[0] // 2                          # middle slice = most tissue
    img = volume[z]
    h, w = img.shape
    c = max(4, min(h, w) // 16)                       # corner patch size (>=4 px)
    corners = np.concatenate([                        # four corner patches
        img[:c, :c].ravel(), img[:c, -c:].ravel(),
        img[-c:, :c].ravel(), img[-c:, -c:].ravel(),
    ])
    noise = float(corners.std()) or 1e-6             # avoid divide-by-zero
    lo, hi = float(img.min()), float(img.max())
    fg = img[img > lo + 0.10 * (hi - lo)]            # foreground (tissue) pixels
    signal = float(fg.mean()) if fg.size else 0.0
    return signal / noise


def run_qc(name: str) -> dict:
    """Run all QC checks on one series -> {flags, metrics, status}.

    `flags` is a list of short problem tags (empty = clean). `metrics` holds the
    measured numbers behind them. `status` is "pass" (no flags), "warn" (some
    flags), or "error" (couldn't even load the pixels).
    """
    flags: list[str] = []
    metrics: dict = {}

    positions, instances = _positions_and_instances(name)
    n = len(positions)
    metrics["n_slices"] = n

    # 1) MISSING SLICES: if InstanceNumbers span more indices than we have files,
    #    the difference is how many slices are absent.
    valid = [i for i in instances if i > 0]
    if valid:
        span = max(valid) - min(valid) + 1
        if span > n:
            flags.append(f"missing-slices({span - n})")

    # 2) UNEVEN SPACING: consecutive slice positions should be equally spaced.
    #    We use the coefficient of variation (std/mean) of the gaps; >15% is odd.
    if n >= 3:
        diffs = np.diff(sorted(positions))
        diffs = diffs[np.abs(diffs) > 1e-6]          # drop zero-gap duplicates
        if diffs.size:
            rel = float(np.std(diffs) / (np.abs(np.mean(diffs)) + 1e-6))
            metrics["spacing_cv"] = round(rel, 3)
            if rel > 0.15:
                flags.append("uneven-spacing")

    # The remaining checks need actual pixels, so load the full volume now.
    try:
        vol = load_series(name).volume
    except Exception as e:                           # noqa: BLE001
        flags.append(f"load-failed:{e}")
        return {"flags": flags, "metrics": metrics, "status": "error"}

    # 3) LOW CONTRAST: the foreground window width relative to its magnitude. A
    #    nearly-flat volume (little to see) scores low.
    lo, hi = volume_window_bounds(vol)
    rng = (hi - lo) / (abs(hi) + 1e-6)
    metrics["contrast"] = round(float(rng), 3)
    if rng < 0.15:
        flags.append("low-contrast")

    # 4) NOISE: see _background_snr. Below ~8 is a grainy acquisition.
    snr = _background_snr(vol)
    metrics["snr"] = round(snr, 1)
    if snr < 8:
        flags.append("low-snr")

    # 5) EMPTY SLICES: count slices that are essentially background. If more than
    #    ~30% of the stack is empty, something's off with the acquisition/crop.
    empties = sum(foreground_fraction(vol[i]) < 0.02 for i in range(vol.shape[0]))
    metrics["empty_slices"] = int(empties)
    if empties > 0.3 * vol.shape[0]:
        flags.append("mostly-empty")

    status = "pass" if not flags else "warn"
    return {"flags": flags, "metrics": metrics, "status": status}
