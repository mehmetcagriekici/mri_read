"""
Step 3 — Sequence classifier + study manifest.

This is the "understanding" layer. The anonymized data has blank
SeriesDescriptions, so we can't just read the sequence name — we INFER it from
MR physics and write a structured manifest.json that every later step consumes.

------------------------------------------------------------------------------
How the classification works (the physics in one paragraph)
------------------------------------------------------------------------------
An MRI sequence's "weighting" (what tissue looks bright) is set by timing:
  - TR (repetition time) and TE (echo time), both in milliseconds.
  - short TE + short TR  -> T1-weighted  (fat bright, fluid dark)
  - long  TE + long  TR  -> T2-weighted  (fluid bright)
  - inversion recovery (IR) with long TE -> FLAIR (fluid SUPPRESSED / dark)
And the ScanningSequence tag names the pulse family:
  SE = spin echo, IR = inversion recovery, GR = gradient echo,
  EP = echo planar (used for diffusion / DWI), RM = research/other.
Geometry finishes the picture: a gradient-echo run that is thin-sliced with many
slices is a 3D T1 volume; a gradient-echo "series" with SliceThickness 0 is a
reformat (a reconstruction derived from a 3D volume, not a real acquisition).

Each series gets {label, confidence, reason} plus a use_for_analysis flag so the
engine only looks at real diagnostic sequences (not reformats/localizers).

Usage:
  python src/manifest.py            # print manifest + write output/manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

from mri import inspect_series, list_series

OUT = Path(__file__).resolve().parent.parent / "output"

# Labels that represent real diagnostic sequences worth analyzing. Anything not
# in this set (reformats, localizers, unknowns) is skipped by the engine.
PRIMARY = {"DWI", "T2 FLAIR", "T2", "T1", "T1 (IR)", "3D T1"}


def _has(seq: str, code: str) -> bool:
    """True if a ScanningSequence string contains a given pulse code (SE/IR/...).

    seq may be a stringified list like "['EP', 'SE']", so a substring test is
    the simplest robust check.
    """
    return code in (seq or "")


def classify(tags: dict, n_slices: int) -> dict:
    """Infer the sequence type of one series -> {label, confidence, reason}.

    Rules are tried in priority order; the FIRST match wins. `reason` records
    which rule fired so the manifest is auditable (you can see *why* Seri6 was
    called FLAIR). `confidence` is a rough 0–1 hand-set trust level.
    """
    seq = tags.get("scanning_sequence", "") or ""
    te = tags.get("echo_time_TE")
    tr = tags.get("repetition_TR")
    thick = tags.get("thickness_mm")

    # 1) Echo-planar acquisitions are diffusion (DWI) in a brain protocol.
    if _has(seq, "EP"):
        return _r("DWI", 0.9, f"EP (echo-planar) in {seq}; TE={te}")

    # 2) Gradient-echo family: distinguish real 3D volumes from derived images.
    if _has(seq, "GR"):
        if n_slices == 1:
            return _r("Localizer/stray", 0.8, "single GR slice")
        if thick == 0.0:
            # SliceThickness 0 is the tell-tale of a reconstructed reformat.
            return _r("Reformat (MPR)", 0.85,
                      f"GR, thickness 0.0 -> derived reformat, {n_slices} slices")
        if (thick is not None and thick <= 3) and n_slices >= 50:
            # Thin slices + many of them = a high-res volumetric T1 acquisition.
            return _r("3D T1", 0.8,
                      f"GR, thin ({thick}mm) x{n_slices} -> volumetric T1")
        return _r("GRE (T1-ish)", 0.5, f"GR, TE={te}, thick={thick}")

    # 3) Inversion recovery: long TE suppresses fluid -> FLAIR; short TE -> T1.
    if _has(seq, "IR"):
        if te is not None and te >= 80:
            return _r("T2 FLAIR", 0.85, f"IR + long TE={te} -> fluid-suppressed")
        return _r("T1 (IR)", 0.7, f"IR + short TE={te}")

    # 4) Fallback on raw TE/TR weighting. Catches plain SE and odd/blank seq
    #    codes (e.g. Seri3's "RM"), which the rules above don't name.
    if te is not None and tr is not None:
        if te >= 80 and tr >= 2000:
            return _r("T2", 0.75, f"long TE={te}, TR={tr}")
        if te < 30 and tr < 1200:
            return _r("T1", 0.7, f"short TE={te}, TR={tr}")
        if te < 30:
            return _r("PD/T1", 0.5, f"short TE={te}, TR={tr}")
    return _r("Unknown", 0.2, f"seq={seq}, TE={te}, TR={tr}")


def _r(label, confidence, reason):
    """Tiny constructor so each rule above stays a readable one-liner."""
    return {"label": label, "confidence": confidence, "reason": reason}


def build_manifest() -> dict:
    """Walk every series, classify it, and assemble the full manifest dict.

    Study-level fields (body part, scanner) are captured once from the first
    series that has tags. Each series row carries its label, why, and the raw
    acquisition numbers so nothing downstream has to re-read DICOM.
    """
    series_out = []
    study = {}
    for name in list_series():
        info = inspect_series(name)                  # header-only (fast)
        tags = info["tags"]
        if not study and tags:                       # capture study info once
            study = {
                "body_part": tags.get("body_part"),
                "manufacturer": tags.get("manufacturer"),
                "model": tags.get("model"),
                "field_T": tags.get("field_T"),
            }
        cls = classify(tags, info["n_slices"])
        series_out.append({
            "series": name,
            "n_slices": info["n_slices"],
            "plane": tags.get("plane"),
            "label": cls["label"],
            "confidence": cls["confidence"],
            "reason": cls["reason"],
            "use_for_analysis": cls["label"] in PRIMARY,   # engine filter
            "acq": {                                        # raw numbers kept for audit

                "scanning_sequence": tags.get("scanning_sequence"),
                "TE": tags.get("echo_time_TE"),
                "TR": tags.get("repetition_TR"),
                "TI": tags.get("inversion_TI"),
                "thickness_mm": tags.get("thickness_mm"),
            },
        })
    return {"study": study, "series": series_out}


def main() -> None:
    """Build the manifest, print a human-readable table, and save the JSON."""
    OUT.mkdir(exist_ok=True)
    m = build_manifest()

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
