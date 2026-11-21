"""Depth maps on disk: 16-bit PNG holding meters x 256, with 0 meaning "no measurement".

This is the KITTI depth devkit's encoding. CAGE uses it for every dataset and for its predictions, so one
reader and one writer serve everything. The scale comes from the frame registry.
"""
import numpy as np
from PIL import Image

from cage.geometry.frames import DEPTH_PNG_SCALE

MAX_DEPTH_M = 65535 / DEPTH_PNG_SCALE               # the largest depth a uint16 can hold: 255.996 m


def read_depth_png(path):
    """uint16 PNG -> float32 meters, 0 where invalid."""
    im = Image.open(path)
    assert im.mode in ("I;16", "I;16B", "I"), f"{path} is not a 16-bit PNG (mode {im.mode})"
    return np.asarray(im, dtype=np.float32) / DEPTH_PNG_SCALE


def write_depth_png(path, depth_m):
    """float meters -> uint16 PNG; NaN, infinity, negative and out-of-range values become 0."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    d = np.where((d > 0) & (d <= MAX_DEPTH_M), d, 0.0)
    Image.fromarray(np.round(d * DEPTH_PNG_SCALE).astype(np.uint16)).save(path)
