"""
CLI entry point — quick standalone check of DWI b-value/ADC handling.

All the real logic lives in mri_read.dwi; this just prints a summary for one
series.

Usage:
  python src/cmd/dwi.py Seri1
"""

from __future__ import annotations

import sys

from mri_read.dwi import diffusion_views


def main() -> None:
    n = sys.argv[1] if len(sys.argv) > 1 else "Seri1"
    v = diffusion_views(n)
    print(f"{n}: {v['note']}")
    if v["high_b"] is not None:
        print(f"  high-b (b={v['b_value']}) volume: {v['high_b'].shape}")
    print(f"  ADC map: {None if v['adc'] is None else v['adc'].shape}")


if __name__ == "__main__":
    main()
