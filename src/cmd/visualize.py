"""
CLI entry point for Step 2 — load and visualize.

All the real logic lives in mri_read.visualize; this just wires up argparse.

Usage:
  python src/cmd/visualize.py                 # overview montage of every series
  python src/cmd/visualize.py --deep          # + full slice export for ALL series
  python src/cmd/visualize.py --deep Seri7    # + full slice export for one series
"""

from __future__ import annotations

import argparse

from mri_read.visualize import deep_dive, deep_dive_all, overview


def main() -> None:
    ap = argparse.ArgumentParser()
    # nargs='?' with const: bare "--deep" -> sentinel "__all__" (all series);
    # "--deep Seri7" -> that one series; flag absent -> args.deep is None.
    ap.add_argument("--deep", nargs="?", const="__all__", metavar="SERIES",
                    help="export every slice; no value = ALL series, "
                         "or name one (e.g. Seri7)")
    args = ap.parse_args()

    overview()                                       # always produce montages
    if args.deep == "__all__":
        deep_dive_all()
    elif args.deep:
        # Accept both "--deep all" and "--deep Seri7".
        deep_dive_all() if args.deep.lower() == "all" else deep_dive(args.deep)


if __name__ == "__main__":
    main()
