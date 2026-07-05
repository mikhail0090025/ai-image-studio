import pytest
import torch
import numpy as np
from PIL import Image

from utils import (
    preprocess_image,
    restore_boxes,
    calculate_iterations,
    calculate_dilation,
    build_mask_levels,
    resize_mask
)


# =========================
# 1. preprocess_image
# =========================

def test_preprocess_image_shape():
    img = Image.new("RGB", (1200, 600), color=(255, 255, 255))

    out, meta = preprocess_image(img)

    assert out.size == (512, 512)
    assert meta["original_size"] == (1200, 600)
    assert "scale" in meta
    assert "pad_left" in meta
    assert "pad_top" in meta


# =========================
# 2. restore_boxes
# =========================

def test_restore_boxes():
    boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)

    meta = {
        "scale": 0.5,
        "pad_left": 10,
        "pad_top": 20,
        "original_size": (1000, 800)
    }

    out = restore_boxes(boxes, meta)

    assert out.shape == (1, 4)
    assert (out >= 0).all()


# =========================
# 3. calculate_iterations
# =========================

def test_calculate_iterations():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    iters = calculate_iterations(mask)

    assert isinstance(iters, int)
    assert iters >= 1


# =========================
# 4. calculate_dilation
# =========================

def test_calculate_dilation():
    box = torch.tensor([0, 0, 300, 300])

    d = calculate_dilation(box)

    assert isinstance(d, int)
    assert 10 <= d <= 200


# =========================
# 5. build_mask_levels
# =========================

def test_build_mask_levels():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1

    levels = build_mask_levels(mask, steps=3, shrink=10)

    assert isinstance(levels, list)
    assert len(levels) >= 1
    assert isinstance(levels[0], np.ndarray)


# =========================
# 6. resize_mask
# =========================

def test_resize_mask():
    mask = np.ones((100, 100), dtype=np.uint8)

    metadata = {
        "scale": 0.5,
        "pad_left": 10,
        "pad_top": 10
    }

    out = resize_mask(mask, metadata)

    assert isinstance(out, np.ndarray)
    assert out.shape == (512, 512)
    assert out.dtype == bool