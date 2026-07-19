from types import SimpleNamespace

from mri_read.mri.geometry import _slice_position, plane_from_orientation


def _ds(iop=None, ipp=None, instance_number=0):
    return SimpleNamespace(ImageOrientationPatient=iop, ImagePositionPatient=ipp,
                           InstanceNumber=instance_number)


def test_slice_position_uses_geometry_when_available():
    # axial: row=x, col=y -> normal=z; ipp z=42 -> position 42
    ds = _ds(iop=[1, 0, 0, 0, 1, 0], ipp=[0, 0, 42])
    assert _slice_position(ds) == 42.0


def test_slice_position_falls_back_to_instance_number():
    ds = _ds(iop=None, ipp=None, instance_number=7)
    assert _slice_position(ds) == 7.0


def test_plane_from_orientation_axial():
    ds = _ds(iop=[1, 0, 0, 0, 1, 0])  # normal along z
    assert plane_from_orientation(ds) == "Axial"


def test_plane_from_orientation_sagittal():
    ds = _ds(iop=[0, 1, 0, 0, 0, 1])  # normal along x
    assert plane_from_orientation(ds) == "Sagittal"


def test_plane_from_orientation_coronal():
    ds = _ds(iop=[1, 0, 0, 0, 0, 1])  # normal along y
    assert plane_from_orientation(ds) == "Coronal"


def test_plane_from_orientation_unknown_without_iop():
    ds = _ds(iop=None)
    assert plane_from_orientation(ds) == "unknown"
