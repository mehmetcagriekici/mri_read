"""
CLI entry point for Step 3 — sequence classifier + study manifest.

Development/debugging tool: prints the manifest table and persists
output/manifest.json standalone, for inspecting classification in isolation.
The primary entry point for running the project is `src/cmd/agent.py`, which
builds the manifest itself.

Usage:
  python src/cmd/manifest.py            # print manifest + write output/manifest.json
"""

from __future__ import annotations

import json

from mri_read.manifest import build_manifest
from mri_read.paths import OUT


def main() -> None:
    OUT.mkdir(exist_ok=True)
    try:
        m = build_manifest()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None

    s = m["study"]
    print(f"Study: {s.get('body_part')} | {s.get('manufacturer')} "
          f"{s.get('model')} @ {s.get('field_T')}T\n")
    print(f"{'series':7} {'label':16} {'plane':9} {'slices':>6} {'conf':>5}  reason")
    print("-" * 92)
    for r in m["series"]:
        flag = "*" if r["use_for_analysis"] else " "   # star = feeds the engine
        print(f"{flag}{r['series']:6} {r['label']:16} {r['plane'][:9]:9} "
              f"{r['n_slices']:>6} {r['confidence']:>5}  {r['reason']}")
    print("\n(* = used for analysis)")

    # This JSON is the contract handed to qc.py and analyze.py.
    path = OUT / "manifest.json"
    path.write_text(json.dumps(m, indent=2))
    print(f"\nManifest written to: {path}")


if __name__ == "__main__":
    main()
