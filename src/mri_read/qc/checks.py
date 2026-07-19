"""Running all QC checks on one series and assembling the flags/status."""

from __future__ import annotations

import numpy as np

from mri_read.dwi import count_bvalue_buckets_from_values
from mri_read.mri import foreground_fraction, load_series
from mri_read.qc.header_metrics import _positions_and_instances
from mri_read.qc.signal_metrics import _background_snr


def run_qc(name: str) -> dict:
    """Run all QC checks on one series -> {flags, metrics, status}.

    `flags` is a list of short problem tags (empty = clean). `metrics` holds the
    measured numbers behind them. `status` is "pass" (no flags), "warn" (some
    flags), or "error" (couldn't even load the pixels).
    """
    flags: list[str] = []
    metrics: dict = {}

    positions, instances, bvalues = _positions_and_instances(name)
    n = len(positions)
    metrics["n_slices"] = n
    metrics["bvalue_buckets"] = count_bvalue_buckets_from_values(bvalues)

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
        series = load_series(name)
        vol = series.volume
    except Exception as e:                           # noqa: BLE001
        flags.append(f"load-failed:{e}")
        return {"flags": flags, "metrics": metrics, "status": "error"}

    # 3) LOW CONTRAST: the foreground window width relative to its magnitude. A
    #    nearly-flat volume (little to see) scores low.
    # series.window_bounds is cached on the Series instance -- analyze's
    # slice encoding reuses the same cached value for this series instead of
    # recomputing the same expensive percentile a second time (see
    # mri.types.Series.window_bounds).
    lo, hi = series.window_bounds
    rng = (hi - lo) / (abs(hi) + 1e-6)
    metrics["contrast"] = round(float(rng), 3)
    if rng < 0.15:
        flags.append("low-contrast")

    # 4) NOISE: see _background_snr. Below ~8 is a grainy acquisition; None means
    #    it couldn't be measured (flat corners), not that the volume is noisy.
    snr = _background_snr(vol)
    metrics["snr"] = round(snr, 1) if snr is not None else None
    if snr is not None and snr < 8:
        flags.append("low-snr")

    # 5) EMPTY SLICES: count slices that are essentially background. If more than
    #    ~30% of the stack is empty, something's off with the acquisition/crop.
    empties = sum(foreground_fraction(vol[i]) < 0.02 for i in range(vol.shape[0]))
    metrics["empty_slices"] = int(empties)
    if empties > 0.3 * vol.shape[0]:
        flags.append("mostly-empty")

    status = "pass" if not flags else "warn"
    return {"flags": flags, "metrics": metrics, "status": status}
