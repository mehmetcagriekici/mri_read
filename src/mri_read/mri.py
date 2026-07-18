"""
mri.py — the shared DICOM/MRI core.

Every other script (explore, visualize, manifest, qc, dwi, analyze) imports from
here so there is exactly ONE implementation of "how we read and interpret DICOM".
If a loading rule needs to change, it changes here and everywhere benefits.

------------------------------------------------------------------------------
DICOM / MRI primer (enough to follow this file)
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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pydicom

from mri_read.paths import DATA_DIR


@dataclass
class Series:
    """A fully loaded series: the pixel data as one 3D volume, plus metadata.

    Attributes:
        name:   folder name of the series (e.g. "Seri7").
        volume: numpy array, shape (n_slices, rows, cols), dtype float32,
                slices ordered head-to-foot by geometric position.
        tags:   dict of the acquisition/geometry tags we care about
                (see extract_tags).
    """
    name: str
    volume: np.ndarray                      # (slices, rows, cols), float32
    tags: dict = field(default_factory=dict)

    @property
    def n_slices(self) -> int:
        """Number of slices in the stacked volume (its first axis)."""
        return self.volume.shape[0]


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


def load_series(name: str) -> Series:
    """Read every slice in a series folder and return a sorted 3D Series.

    Steps:
      1. Read all .dcm files (force=True tolerates minor header quirks).
      2. Sort by true geometric position so the volume is anatomically ordered.
      3. Convert each slice to real-valued float32 via RescaleSlope/Intercept.
      4. Stack into a single (slices, rows, cols) array, keeping only slices
         that share the most common shape (a stray odd-sized frame — e.g. a
         localizer mixed in — would otherwise break np.stack).
    """
    folder = DATA_DIR / name
    files = sorted(folder.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files in {folder}")

    # Read full datasets (pixels included). force=True: don't choke on files
    # missing the formal DICOM preamble.
    datasets = [pydicom.dcmread(str(f), force=True) for f in files]
    datasets.sort(key=_slice_position)

    frames = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        # Apply the linear rescale to get real intensities. `or 1`/`or 0` guard
        # against tags present but set to None/empty.
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        frames.append(arr * slope + intercept)

    # Guard against mixed frame sizes: keep only the modal (most common) shape.
    shapes = [f.shape for f in frames]
    modal = max(set(shapes), key=shapes.count)
    frames = [f for f in frames if f.shape == modal]
    volume = np.stack(frames, axis=0)            # -> (slices, rows, cols)

    # Metadata comes from the first (lowest-position) slice; the tags we extract
    # are constant across a series.
    return Series(name=name, volume=volume, tags=extract_tags(datasets[0]))


def extract_tags(ds) -> dict:
    """Copy the handful of DICOM tags we use into a plain, JSON-friendly dict.

    Using getattr with defaults means a missing tag yields "—" or None instead
    of raising — important because this anonymized data has several blank tags.
    """
    def num(v):
        """Coerce a DICOM value to float, or None if it isn't numeric."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "modality": str(getattr(ds, "Modality", "—")),            # "MR"
        "plane": plane_from_orientation(ds),                       # Axial/...
        "scanning_sequence": str(getattr(ds, "ScanningSequence", "—")),  # SE/IR/GR/EP
        "sequence_variant": str(getattr(ds, "SequenceVariant", "—")),
        "protocol": str(getattr(ds, "ProtocolName", "—")),
        "series_number": getattr(ds, "SeriesNumber", None),
        "echo_time_TE": num(getattr(ds, "EchoTime", None)),        # TE (ms)
        "repetition_TR": num(getattr(ds, "RepetitionTime", None)), # TR (ms)
        "inversion_TI": num(getattr(ds, "InversionTime", None)),   # TI (ms)
        "body_part": str(getattr(ds, "BodyPartExamined", "—")),
        "thickness_mm": num(getattr(ds, "SliceThickness", None)),
        "manufacturer": str(getattr(ds, "Manufacturer", "—")),
        "model": str(getattr(ds, "ManufacturerModelName", "—")),
        "field_T": num(getattr(ds, "MagneticFieldStrength", None)),  # e.g. 3.0
    }


def inspect_series(name: str) -> dict:
    """Header-only look at a series (no pixel decode) — fast, for the manifest.

    stop_before_pixels=True makes pydicom skip the (large) pixel data, so this
    is much cheaper than load_series. We read files until one parses, take its
    tags, and count the .dcm files for the slice total.

    Returns {"name", "n_slices", "tags"}.
    """
    folder = DATA_DIR / name
    files = sorted(folder.glob("*.dcm"))
    ds = None
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            break                                    # first readable header wins
        except Exception:                            # noqa: BLE001 - be forgiving
            continue
    if ds is None:
        return {"name": name, "n_slices": 0, "tags": {}}
    return {"name": name, "n_slices": len(files), "tags": extract_tags(ds)}


def window_to_uint8(img: np.ndarray, ds=None) -> np.ndarray:
    """Map one float slice to 0–255 for display (PER-SLICE windowing).

    "Windowing" = choosing an intensity range [lo, hi] to stretch onto 0–255;
    everything below lo is black, above hi is white. Used by visualize.py for
    quick-look montages.

    Priority:
      1. The DICOM-provided WindowCenter/WindowWidth (what the scanner intended).
      2. Otherwise a robust 1st–99th percentile stretch of this slice.

    NOTE: this windows each slice independently, so brightness isn't comparable
    across slices. For model input use volume_window_bounds + apply_window,
    which window a whole series consistently.
    """
    center = width = None
    if ds is not None:
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            # These tags are sometimes multi-valued; take the first entry.
            center = float(wc[0] if hasattr(wc, "__iter__") else wc)
            width = float(ww[0] if hasattr(ww, "__iter__") else ww)

    if center is None or not width:
        lo, hi = np.percentile(img, [1, 99])         # robust auto-window
    else:
        lo, hi = center - width / 2, center + width / 2

    if hi <= lo:                                     # degenerate (flat) slice
        hi = lo + 1
    out = np.clip((img - lo) / (hi - lo), 0, 1)      # normalize to 0..1
    return (out * 255).astype(np.uint8)              # -> 8-bit greyscale


# --- hardening helpers: consistent windowing, foreground, diffusion b-values ---

def volume_window_bounds(volume: np.ndarray) -> tuple[float, float]:
    """Compute ONE (lo, hi) window for a whole series so its slices match.

    Why over the whole volume: a model (or a human) comparing slices needs them
    windowed identically — per-slice windowing makes a dark slice and a bright
    slice look the same, hiding real intensity differences.

    Why foreground-only: MR background is ~0. Including all those zeros drags the
    1st percentile down and washes out tissue contrast, so we take percentiles
    over voxels above the volume minimum only.
    """
    fg = volume[volume > volume.min()]               # drop the background floor
    if fg.size == 0:                                 # entirely flat volume
        fg = volume.ravel()
    lo, hi = np.percentile(fg, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def apply_window(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map a slice to 8-bit using explicit bounds from volume_window_bounds."""
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def foreground_fraction(img: np.ndarray) -> float:
    """Estimate what fraction of a slice is tissue vs. empty background.

    Method: threshold at 10% of the slice's dynamic range and count pixels
    above it. Crude but reliable enough to distinguish an empty end-of-stack
    slice (~0) from one packed with brain (~0.3–0.6). Used by QC (empty-slice
    detection) and by content-aware slice selection in analyze.py.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:                                     # flat slice -> no content
        return 0.0
    thresh = lo + 0.10 * (hi - lo)
    return float((img > thresh).mean())              # mean of a bool array = fraction


# Diffusion b-value (the DWI "diffusion weighting strength") is stored in the
# standard tag on modern scanners, but older GE data hides it in private tags.
# We try the standard one first, then the two common GE locations.
_BVALUE_TAGS = [
    (0x0018, 0x9087),   # DiffusionBValue (standard)
    (0x0043, 0x1039),   # GE private
    (0x0019, 0x100C),   # GE private (alternate)
]


def read_bvalue(ds) -> float | None:
    """Best-effort diffusion b-value for one slice, or None if not tagged.

    Checks the standard tag then GE private tags. GE occasionally packs the
    b-value into the low 5 digits of a larger slop integer (e.g. 1000001000
    encodes b=1000), so we unpack values above 100000 with a modulo.
    """
    for group, elem in _BVALUE_TAGS:
        if (group, elem) in ds:                      # tag present in this file?
            try:
                val = ds[group, elem].value
                if isinstance(val, (list, tuple)):   # sometimes multi-valued
                    val = val[0]
                b = float(val)
                if b > 100000:                       # unpack GE's packed form
                    b = b % 100000
                return b
            except (ValueError, TypeError):
                continue                             # try the next candidate tag
    return None


def list_series() -> list[str]:
    """List series folder names sorted numerically (Seri2 before Seri10).

    Plain sorted() would order these as strings ("Seri10" < "Seri2"), so we key
    on the integer embedded in the name.
    """
    def key(n: str):
        d = "".join(c for c in n if c.isdigit())     # pull digits out of the name
        return int(d) if d else 10**9                # names w/o digits sort last
    return sorted((p.name for p in DATA_DIR.iterdir() if p.is_dir()), key=key)
