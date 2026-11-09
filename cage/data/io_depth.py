import numpy as np
from PIL import Image

def read_depth_png(path):
    """uint16 PNG -> float32 meters, 0 where invalid (KITTI depth devkit encoding)."""
    im = Image.open(path)
    assert im.mode in ("I;16", "I;16B", "I"), f"{path} is not a 16-bit PNG (mode {im.mode})"
    return np.asarray(im, dtype=np.float32) / 256.0

def write_depth_png(path, depth_m):
    """float meters -> uint16 PNG; NaN, infinity, negative and out-of-range values become 0."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    d = np.where((d > 0) & (d < 256.0), d, 0.0)
    Image.fromarray(np.round(d * 256.0).astype(np.uint16)).save(path)