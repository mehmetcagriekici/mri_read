"""
CLI entry point for Step 3b — deterministic quality control.

Development/debugging tool: prints the QC table and augments
output/manifest.json standalone, for inspecting QC in isolation. The primary
entry point for running the project is `src/cmd/agent.py`, whose run_qc tool
calls the same mri_read.qc.run_qc() but doesn't persist to manifest.json.

Usage:
  python src/cmd/qc.py            # prints QC table, updates manifest.json
"""

from __future__ import annotations

import json

from mri_read.mri import list_series
from mri_read.paths import OUT
from mri_read.qc import run_qc

MANIFEST = OUT / "manifest.json"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # If a manifest exists we AUGMENT it in place (adding a "qc" block per row);
    # if not, we still print the QC table but have nowhere to persist it.
    manifest = None
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    # Index series rows by name for quick lookup when attaching qc.
    by_name = {r["series"]: r for r in (manifest["series"] if manifest else [])}

    print(f"{'series':7} {'status':6} {'slices':>6} {'contr':>6} {'snr':>6} "
          f"{'empty':>6}  flags")
    print("-" * 78)
    try:
        names = list_series()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    for name in names:
        qc = run_qc(name)
        m = qc["metrics"]
        if name in by_name:
            by_name[name]["qc"] = qc
        snr = m.get("snr")                            # None -- e.g. flat/masked corners
        print(f"{name:7} {qc['status']:6} {m.get('n_slices',0):>6} "
              f"{m.get('contrast','-'):>6} {snr if snr is not None else '-':>6} "
              f"{m.get('empty_slices','-'):>6}  {', '.join(qc['flags']) or 'ok'}")

    if manifest is not None:
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"\nManifest updated with qc: {MANIFEST}")
    else:
        print("\n(no manifest.json yet — run src/cmd/manifest.py to persist qc)")


if __name__ == "__main__":
    main()
