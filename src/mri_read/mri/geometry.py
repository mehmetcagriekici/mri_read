"""Slice ordering and plane classification from DICOM orientation tags."""

from __future__ import annotations

import numpy as np


def _slice_position(ds) -> float:
    """Return a single number that orders a slice within its series.

    Preferred: the slice's real-world position along the slice normal, computed
    from ImageOrientationPatient + ImagePositionPatient. This is robust even
    when file names or InstanceNumbers are missing or out of order.

    Fallback: InstanceNumber (the scanner's own slice index) when geometry tags
    are absent.

    (Leading underscore = "module-internal helper"; other modules may still
    import it, but it's not part of the tidy public surface.)
    """
    iop = getattr(ds, "ImageOrientationPatient", None)   # 6 direction cosines
    ipp = getattr(ds, "ImagePositionPatient", None)      # xyz of the slice
    if iop is not None and ipp is not None:
        row = np.array(iop[:3], dtype=float)             # row direction cosine
        col = np.array(iop[3:], dtype=float)             # column direction cosine
        normal = np.cross(row, col)                      # perpendicular to slice
        # Dot product = how far along the normal this slice sits (mm).
        return float(np.dot(np.array(ipp, dtype=float), normal))
    return float(getattr(ds, "InstanceNumber", 0))


def plane_from_orientation(ds) -> str:
    """Classify the acquisition plane as Axial / Sagittal / Coronal.

    The slice normal points along whichever patient axis the slices are stacked
    on. Its dominant component tells us the plane:
        x dominant -> slices stacked left-right   -> Sagittal
        y dominant -> slices stacked front-back    -> Coronal
        z dominant -> slices stacked head-foot      -> Axial
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        return "unknown"
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:], dtype=float)
    normal = np.abs(np.cross(row, col))          # abs: we only need the axis
    axis = int(np.argmax(normal))                # 0=x, 1=y, 2=z
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[axis]
