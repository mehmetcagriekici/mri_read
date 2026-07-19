import numpy as np
from PIL import Image

from mri_read.visualize.montage import montage


def test_montage_returns_a_pil_image_within_size_cap():
    vol = np.random.default_rng(0).normal(50, 10, (10, 20, 20)).astype(np.float32)
    img = montage(vol, cols=4, max_tiles=8)
    assert isinstance(img, Image.Image)
    assert img.width <= 1400 and img.height <= 1400


def test_montage_samples_at_most_max_tiles():
    vol = np.zeros((50, 10, 10), dtype=np.float32)
    img = montage(vol, cols=6, max_tiles=6)
    # 6 tiles at cols=6 -> exactly one row of 10x10 tiles (before thumbnailing)
    assert img.height <= 10 and img.width <= 60


def test_montage_handles_fewer_slices_than_max_tiles():
    vol = np.zeros((3, 10, 10), dtype=np.float32)
    img = montage(vol, cols=6, max_tiles=24)
    assert isinstance(img, Image.Image)
