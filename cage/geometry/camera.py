import numpy as np

def unproject_depth(depth, K):
    """depth [H, W] meters (0 = invalid), K [3, 3] -> points [H, W, 3] in the camera frame, valid [H, W]."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    pix = np.stack([u, v, np.ones_like(u)], -1).astype(np.float64) # [u_i, v_i, 1] , [u_{i+1}, v_{i+1}, 1], ...
    rays = pix @ np.linalg.inv(K).T                 # one ray per pixel, z = 1
    return rays * depth[..., None], depth > 0

def project_points(points, K):
    """points [..., 3] in the camera frame -> pixels [..., 2]."""
    uvw = points @ np.asarray(K).T
    return uvw[..., :2] / np.clip(uvw[..., 2:], 1e-6, None)