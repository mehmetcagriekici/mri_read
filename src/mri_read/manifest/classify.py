"""
Sequence classifier: inferring a series' MRI sequence type from physics.

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
"""

from __future__ import annotations

# Labels that represent real diagnostic sequences worth analyzing. Anything not
# in this set (reformats, localizers, unknowns) is skipped by the engine.
PRIMARY = {"DWI", "T2 FLAIR", "T2", "T1", "T1 (IR)", "3D T1"}


def _has(seq: str, code: str) -> bool:
    """True if a ScanningSequence string contains a given pulse code (SE/IR/...).

    seq may be a stringified list like "['EP', 'SE']", so a substring test is
    the simplest robust check.
    """
    return code in (seq or "")


def _r(label, confidence, reason):
    """Tiny constructor so each rule below stays a readable one-liner."""
    return {"label": label, "confidence": confidence, "reason": reason}


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
