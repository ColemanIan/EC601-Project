import numpy as np

def roty(t):
    """Rotation about the y axis (the vertical axis of the upright frame)."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def box3d_corners(center, size_lwh, heading):
    """
    Input: center point of object, size of bounding box in [length, width, height], heading angle
    Output: points of the bounding box after being transformed by rotation and translation -> [bottom points, ... , top points]
    ###[8, 3] corners in Frustum PointNets' order (D2.1): 0-3 bottom face, 4-7 top face.
    """
    l, w, h = size_lwh
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([h, h, h, h, -h, -h, -h, -h]) / 2
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    return (roty(heading) @ np.stack([x, y, z])).T + np.asarray(center)

def points_in_box(points, center, size_lwh, heading):
    """
    Input: points to be classified, center point of object (points), size of the object in [length, width, height]
    Output: all points that are within +/- 1/2 of lwh from center point
    
    points [N, 3], in the same frame as the box -> bool [N].
    """
    local = (np.asarray(points) - np.asarray(center)) @ roty(heading)    # into the box's own frame
    half = np.array([size_lwh[0], size_lwh[2], size_lwh[1]]) / 2         # x = length, y = height, z = width
    return np.all(np.abs(local) <= half, axis=1)

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]