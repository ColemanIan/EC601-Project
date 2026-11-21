"""3D boxes in CAGE's convention (numpy).

A box is a center, a size (length, width, height) and a heading: a rotation about the y axis.
Indoor boxes are stored in the upright frame U. KITTI boxes are stored in the camera frame, which on KITTI
is the same thing, because its R_gravity is the identity. The corner order comes from the frame registry.
"""
import numpy as np

from cage.geometry.frames import CAGE_CORNER_SIGNS, rot_y

# The 12 edges of a box as pairs of corner indices: bottom face, top face, then the four vertical edges.
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box3d_corners(center, size_lwh, heading):
    """-> [8, 3] corners in Frustum PointNets' order: corners 0-3 are the bottom face, 4-7 the top face.

    In the box's own frame x is its length, y its height (pointing down) and z its width, so corner i sits
    at sign_i * (l/2, h/2, w/2). rot_y(heading) turns that frame into the box's actual direction.
    """
    length, width, height = size_lwh
    local = CAGE_CORNER_SIGNS * np.array([length, height, width]) / 2.0
    return local @ rot_y(heading).T + np.asarray(center, dtype=np.float64)


def points_in_box(points, center, size_lwh, heading):
    """points [N, 3], in the same frame as the box -> bool [N]."""
    local = (np.asarray(points) - np.asarray(center)) @ rot_y(heading)      # into the box's own frame
    half = np.array([size_lwh[0], size_lwh[2], size_lwh[1]]) / 2.0         # x = length, y = height, z = width
    return np.all(np.abs(local) <= half, axis=1)
