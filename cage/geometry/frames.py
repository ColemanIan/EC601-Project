"""Frame conventions in code.

configs/frames.json holds the fixed conventions of the project as data: what each frame's axes mean, the
matrices that change axes, Total3D's angle bins, the corner orders, the depth encoding and the class lists.
This module loads that file once and adds the formulas that JSON cannot hold.

Rule of the repository: no other file defines an axis-change matrix, a corner order, an angle bin or a class
list. Every other module imports them from here, so each convention has one definition and one test
(tests/test_frames.py).

Notation used in the comments:
  C  CAGE camera frame (OpenCV): x right, y down, z forward. Origin at the optical center. Meters.
  U  upright frame: C rotated by R_gravity so that y points along gravity. p_U = R_gravity @ p_C.
  W  Total3D world frame: x toward, y up, z right; gravity-aligned; same origin as C.
  V  KITTI Velodyne frame: x forward, y left, z up.
Points are stored as rows of an [N, 3] array, so "apply matrix A to every point" is written  points @ A.T.
"""
import json
import os
from pathlib import Path

import numpy as np

REGISTRY_PATH = Path(os.environ.get("CAGE_FRAMES", Path(__file__).resolve().parents[2] / "configs" / "frames.json"))
REGISTRY = json.loads(REGISTRY_PATH.read_text())


def _matrix(section, name):
    return np.array(REGISTRY[section][name]["matrix"], dtype=np.float64)


# ---------------------------------------------------------------- constants read from the registry

T_TOTAL3D = _matrix("axis_changes", "T")                    # Total3D axes <-> OpenCV axes, and W -> U. Its own inverse.
F_OPEN3D = _matrix("axis_changes", "F_open3d")              # C or U -> Open3D's display frame. For viewing only.
P_PYTORCH3D = _matrix("axis_changes", "P_pytorch3d")        # C -> PyTorch3D camera coordinates.
VELO_TO_CAM_IDEAL = _matrix("axis_changes", "VELO_TO_CAM_IDEAL")   # V -> C without the small calibration angles.
MODEL_TO_BOX = _matrix("axis_changes", "MODEL_TO_BOX")      # Pix3D model axes -> the box's own axes.
M_MIRROR = _matrix("reflections", "M_mirror")               # horizontal flip of the camera frame.

PITCH_BINS = np.deg2rad(REGISTRY["total3d_bins_deg"]["pitch"])     # [2, 2] radians
ROLL_BINS = np.deg2rad(REGISTRY["total3d_bins_deg"]["roll"])

CAGE_CORNER_SIGNS = np.array(REGISTRY["box_corners"]["cage_order"]["signs"], dtype=np.float64)        # [8, 3]
TOTAL3D_CORNER_SIGNS = np.array(REGISTRY["box_corners"]["total3d_order"]["signs"], dtype=np.float64)  # [8, 3]
TOTAL3D_TO_CAGE_CORNER = list(REGISTRY["box_corners"]["total3d_to_cage_index"])

DEPTH_PNG_SCALE = float(REGISTRY["encodings"]["depth_png_scale"])            # stored value = meters x 256
SUNRGBD_DEPTH_CAP_M = float(REGISTRY["encodings"]["sunrgbd_depth_cap_m"])    # Total3D clipped depth at 8 m

KITTI_CLASSES = list(REGISTRY["classes"]["kitti"])
SUNRGBD_CLASSES = list(REGISTRY["classes"]["sunrgbd"])
PIX3D_CLASSES = list(REGISTRY["classes"]["pix3d"])          # index = Total3D's class_id
NYU40_CLASSES = list(REGISTRY["classes"]["nyu40"])          # index = Total3D's size_cls
KINS_TO_KITTI_CLASS = dict(REGISTRY["classes"]["kins_to_kitti"])


# ---------------------------------------------------------------- elementary rotations

def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle):
    """Rotation about the y axis, the vertical axis of the upright frame. rot_y(h) @ [1, 0, 0] = (cos h, 0, -sin h)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(angle):
    """Any angle -> the same direction in [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


# ---------------------------------------------------------------- boxes

def length_axis(heading):
    """Unit vector along a box's length axis (the direction the object faces)."""
    return np.array([np.cos(heading), 0.0, -np.sin(heading)])


def heading_of(direction):
    """Inverse of length_axis: a horizontal direction -> heading."""
    return float(np.arctan2(-direction[2], direction[0]))


# ---------------------------------------------------------------- KITTI

def split_kitti_projection(P2):
    """P2 = K [I | t_cam2]  ->  K [3, 3] and t_cam2 [3] in meters.

    KITTI's labels and LiDAR calibration refer to rectified camera 0; the color images come from camera 2,
    about 6 cm beside it. A point p in rectified camera 0 is p + t_cam2 in camera 2, and projecting that with
    K gives the same pixel as projecting p with the 3 x 4 matrix P2.
    """
    K = np.asarray(P2, dtype=np.float64)[:, :3]
    return K, np.linalg.solve(K, np.asarray(P2, dtype=np.float64)[:, 3])


def velodyne_to_camera2(points_velo, Tr_velo_to_cam, R0_rect, t_cam2):
    """LiDAR points [N, 3] in V -> [N, 3] in camera 2 (CAGE's frame for KITTI).

    p_cam0 = Tr_velo_to_cam @ [p_V, 1]     unrectified camera 0
    p_rect = R0_rect @ p_cam0              rectified camera 0
    p_C    = p_rect + t_cam2               camera 2
    """
    homogeneous = np.c_[points_velo, np.ones(len(points_velo))]
    points_cam0 = homogeneous @ np.asarray(Tr_velo_to_cam).T
    return points_cam0 @ np.asarray(R0_rect).T + np.asarray(t_cam2)


def kitti_label_to_box(dimensions_hwl, location, rotation_y, t_cam2):
    """One KITTI label -> CAGE's (center [3], size_lwh [3], heading).

    KITTI lists dimensions as height, width, length and gives the bottom center of the box in rectified
    camera 0. y points down, so the geometric center is half a height above the bottom: a smaller y.
    KITTI's rotation_y has the same definition as CAGE's heading and is copied unchanged.
    """
    height, width, length = (float(v) for v in dimensions_hwl)
    center = np.asarray(location, dtype=np.float64) + np.asarray(t_cam2) - np.array([0.0, height / 2.0, 0.0])
    return center, [length, width, height], float(rotation_y)


def box_to_kitti_label(center, size_lwh, heading, t_cam2):
    """Inverse of kitti_label_to_box: -> (dimensions_hwl, location, rotation_y)."""
    length, width, height = (float(v) for v in size_lwh)
    location = np.asarray(center, dtype=np.float64) - np.asarray(t_cam2) + np.array([0.0, height / 2.0, 0.0])
    return [height, width, length], location, float(heading)


# ---------------------------------------------------------------- Total3D (SUN RGB-D)

def angle_from_bin(bins, bin_index, residual):
    """Total3D stores an angle as a bin index plus a residual measured in bin widths (num_from_bins)."""
    bin_width = bins[0][1] - bins[0][0]
    return float(bins[int(bin_index)].mean() + float(residual) * bin_width)


def total3d_camera_rotation(pitch, roll):
    """Total3D's cam_R for yaw = 0 (libs/tools.py::R_from_yaw_pitch_roll). Equal to rot_z(pitch) @ rot_x(roll).

    Its columns are the camera's toward, up and right axes written in W. A negative pitch looks down.
    """
    cos_p, sin_p, cos_r, sin_r = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cos_p, -cos_r * sin_p, sin_p * sin_r],
                     [sin_p, cos_p * cos_r, -cos_p * sin_r],
                     [0.0, sin_r, cos_r]])


def gravity_from_total3d(pitch, roll):
    """Total3D's pitch and roll -> R_gravity, with p_U = R_gravity @ p_C. Equal to rot_x(pitch) @ rot_z(roll).

    Reading right to left: T_TOTAL3D turns a C point into Total3D's camera axes, cam_R rotates it into W,
    and T_TOTAL3D turns W's axes (toward, up, right) into U's (right, down, forward).
    """
    return T_TOTAL3D @ total3d_camera_rotation(pitch, roll) @ T_TOTAL3D


def pitch_roll_from_gravity(R_gravity):
    """Inverse of gravity_from_total3d: R_gravity -> (pitch, roll) in radians."""
    R = np.asarray(R_gravity)
    return float(np.arctan2(-R[1, 2], R[2, 2])), float(np.arctan2(-R[0, 1], R[0, 0]))


def total3d_world_to_upright(points_world):
    """Points [N, 3] in W -> [N, 3] in U. The two frames differ only in the order and sign of their axes."""
    return np.asarray(points_world, dtype=np.float64) @ T_TOTAL3D.T


def box_from_total3d_corners(corners_world):
    """Total3D's 8 box corners in W -> CAGE's box in U: {"center", "size_lwh", "heading"}.

    Total3D's corner order (get_corners_of_bb3d_no_index): corner 1 - corner 0 runs along the object's
    front axis, corner 2 - corner 1 along its side axis, corner 0 - corner 4 along its up axis.
    CAGE's length axis is the front axis, so length = front extent, width = side extent, height = up extent.
    """
    corners = total3d_world_to_upright(corners_world)
    half_front = (corners[1] - corners[0]) / 2.0
    half_side = (corners[2] - corners[1]) / 2.0
    half_up = (corners[0] - corners[4]) / 2.0
    return {"center": corners.mean(axis=0).tolist(),
            "size_lwh": [2.0 * float(np.linalg.norm(half_front)), 2.0 * float(np.linalg.norm(half_side)),
                         2.0 * float(np.linalg.norm(half_up))],
            "heading": heading_of(half_front / np.linalg.norm(half_front))}


def place_model_points(model_points, center, size_lwh, heading):
    """Points [N, 3] filling [-1, 1]^3 in the Pix3D model frame -> the same points inside a CAGE box.

    This reproduces Total3D's recover_points_to_world_sys in CAGE's conventions: model z runs along the
    box's length axis, model y is up, model x runs along its width axis.
    """
    length, width, height = size_lwh
    local = np.asarray(model_points, dtype=np.float64) @ MODEL_TO_BOX.T * np.array([length, height, width]) / 2.0
    return local @ rot_y(heading).T + np.asarray(center)


# ---------------------------------------------------------------- frustum frame

def frustum_angle(ray_upright):
    """Angle about the y axis between the +z axis and a ray given in U."""
    return float(np.arctan2(ray_upright[0], ray_upright[2]))


def to_frustum_frame(points, angle):
    """Rotate points [N, 3] about y so that the ray at `angle` becomes the +z axis: p_F = rot_y(-angle) @ p."""
    return np.asarray(points, dtype=np.float64) @ rot_y(-angle).T


# ---------------------------------------------------------------- horizontal flip

def mirror_points(points):
    return np.asarray(points, dtype=np.float64) @ M_MIRROR.T


def mirror_rotation(R):
    """A rotation between two frames that are both mirrored: R -> M R M. Pitch is kept and roll changes sign."""
    return M_MIRROR @ np.asarray(R) @ M_MIRROR


def mirror_heading(heading):
    """The length axis (cos h, 0, -sin h) mirrored in x is (-cos h, 0, -sin h), which is heading pi - h."""
    return wrap_angle(np.pi - heading)


def mirror_K(K, width):
    """Intrinsics of the flipped image: pixel u moves to width - 1 - u, so cx does too."""
    K = np.array(K, dtype=np.float64)
    K[0, 2] = width - 1.0 - K[0, 2]
    return K


def mirror_box2d(box, width):
    """[x1, y1, x2, y2] in the flipped image. The left and right edges swap roles."""
    x1, y1, x2, y2 = box
    return [width - 1.0 - x2, y1, width - 1.0 - x1, y2]


# ---------------------------------------------------------------- viewers

def to_open3d(points):
    """C or U points -> Open3D's display frame (y up, z toward the viewer). Never save the result."""
    return np.asarray(points, dtype=np.float64) @ F_OPEN3D.T
