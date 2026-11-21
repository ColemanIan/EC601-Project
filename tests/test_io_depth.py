# tests/test_io_depth.py
import numpy as np
from cage.data.io_depth import read_depth_png, write_depth_png

def test_round_trip(tmp_path):
    d = np.random.uniform(0.5, 80.0, (64, 96)).astype(np.float32)
    d[::7] = 0.0                                          # some invalid rows
    write_depth_png(tmp_path / "d.png", d)
    back = read_depth_png(tmp_path / "d.png")
    assert np.abs(back - d).max() <= 1 / 512 + 1e-6        # rounding to 1/256 m
    assert (back[::7] == 0).all()
