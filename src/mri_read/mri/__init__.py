"""
mri — the shared DICOM/MRI core.

Every other package (explore, visualize, manifest, qc, dwi, analyze) imports
from here so there is exactly ONE implementation of "how we read and interpret
DICOM". If a loading rule needs to change, it changes here and everywhere
benefits.

------------------------------------------------------------------------------
DICOM / MRI primer (enough to follow this package)
------------------------------------------------------------------------------
- A DICOM file (.dcm) = one 2D image slice + a big header of "tags" (patient,
  scanner, geometry, acquisition settings). One MRI "series" = a folder of
  slices that together form a 3D volume. One "study" = many series (T1, T2,
  FLAIR, DWI, ...).
- Pixel values are stored as integers and may need a linear rescale to reach
  real units:  real = raw * RescaleSlope + RescaleIntercept.
- Slice geometry comes from two tags:
    ImageOrientationPatient (IOP): 6 numbers = the row & column direction
      cosines of the slice plane in patient space.
    ImagePositionPatient (IPP): xyz of the slice's top-left voxel.
  The slice NORMAL = cross(row_dir, col_dir); projecting IPP onto the normal
  gives a real-world position we can sort by (more reliable than file names or
  InstanceNumber).
- Key acquisition tags used downstream: EchoTime (TE), RepetitionTime (TR),
  InversionTime (TI), ScanningSequence (SE/IR/GR/EP...), SliceThickness.

Nothing here talks to a network or a model — it is pure local file reading.

Layout (one responsibility per file):
  types.py     : the Series dataclass.
  geometry.py  : slice ordering (_slice_position) and plane classification.
  tags.py      : extract_tags, normalizing a DICOM dataset into a plain dict.
  listing.py   : list_series, enumerating series folders (no DICOM parsing).
  loading.py   : load_series/inspect_series, reading one series folder
                 (full pixels vs. header-only).
  windowing.py : mapping float intensities to 8-bit for display/model input.
  foreground.py: foreground_fraction, a tissue-content metric (not windowing).
  bvalue.py    : diffusion b-value tag extraction.
"""

from mri_read.mri.bvalue import read_bvalue
from mri_read.mri.foreground import foreground_fraction
from mri_read.mri.geometry import _slice_position, plane_from_orientation
from mri_read.mri.listing import list_series
from mri_read.mri.loading import inspect_series, load_series
from mri_read.mri.tags import extract_tags
from mri_read.mri.types import Series
from mri_read.mri.windowing import (apply_window, volume_window_bounds,
                                    window_to_uint8)
from mri_read.paths import DATA_DIR

__all__ = [
    "Series", "DATA_DIR",
    "load_series", "extract_tags", "inspect_series", "list_series",
    "plane_from_orientation", "_slice_position",
    "window_to_uint8", "volume_window_bounds", "apply_window",
    "foreground_fraction", "read_bvalue",
]
