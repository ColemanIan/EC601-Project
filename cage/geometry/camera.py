"""The pinhole camera (numpy): depth map -> 3D points, and 3D points -> pixels.

Pixel (u, v) has u counting columns to the right and v counting rows down; integer values are pixel centers.
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] is in pixels. Depth is the z coordinate, not the distance.
"""
import numpy as np


def unproject_depth(depth, K):
    """depth [H, W] meters (0 = invalid), K [3, 3] -> points [H, W, 3] in the camera frame, valid [H, W].

    p = depth * K^-1 [u, v, 1]. An invalid pixel gives the point (0, 0, 0); use `valid` to drop it.
    """
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    pix = np.stack([u, v, np.ones_like(u)], -1).astype(np.float64)
    rays = pix @ np.linalg.inv(K).T                 # one ray per pixel, with z = 1
    return rays * depth[..., None], depth > 0


def project_points(points, K):
    """points [..., 3] in the camera frame -> pixels [..., 2]: (u, v) = (fx x / z + cx, fy y / z + cy)."""
    uvw = np.asarray(points) @ np.asarray(K).T
    return uvw[..., :2] / np.clip(uvw[..., 2:], 1e-6, None)
