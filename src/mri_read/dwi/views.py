"""What analysis should look at for a DWI series."""

from __future__ import annotations

import numpy as np

from mri_read.dwi.adc import compute_adc
from mri_read.dwi.loading import load_by_bvalue


def diffusion_views(name: str) -> dict:
    """What analysis should look at for a DWI series.

    Returns {'high_b': volume, 'b_value': b, 'adc': volume|None, 'note': str}.
    Falls back to the whole stack if no b-values are tagged.
    """
    by_b = load_by_bvalue(name)
    real_bs = sorted(b for b in by_b if b is not None and b >= 0)

    if not real_bs:
        whole = np.concatenate(list(by_b.values()), axis=0) if by_b else None
        return {"high_b": whole, "b_value": None, "adc": None,
                "note": "no b-value tags found; using full stack"}

    high_b = real_bs[-1]
    return {
        "high_b": by_b[high_b],
        "b_value": high_b,
        "adc": compute_adc(by_b),
        "note": f"b-values found: {real_bs}",
    }
