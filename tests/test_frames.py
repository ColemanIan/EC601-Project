"""Tests for the frame registry (configs/frames.json) and the formulas in cage/geometry/frames.py.

No dataset is needed. Every axis-change matrix is rebuilt from the plain-language axis names of its two
frames and compared with the stored matrix, so a wrong entry cannot pass by agreeing with itself. The
formulas are checked against Total3D's and KITTI's own formulas, copied here from their sources, and
against worked examples with known answers.

    pytest tests/test_frames.py -v
"""
import re
from pathlib import Path

import numpy as np
import pytest

from cage.geometry import frames
from cage.geometry.boxes import box3d_corners, points_in_box
from cage.geometry.camera import project_points, unproject_depth
from cage.geometry.frames import REGISTRY, rot_x, rot_y, rot_z

DIRS = {k: np.array(v, float) for k, v in REGISTRY["directions"].items() if k != "about"}
FRAMES = REGISTRY["frames"]
CHANGES = {k: v for k, v in REGISTRY["axis_changes"].items() if k != "about"}
EXAMPLES = REGISTRY["worked_examples"]
T = frames.T_TOTAL3D
REPO = Path(__file__).resolve().parents[1]


def basis(frame_name):
    """3 x 3 matrix whose columns are the frame's x, y, z axes written in the OpenCV camera frame."""
    return np.stack([DIRS[axis] for axis in FRAMES[frame_name]["axes"]], axis=1)


def is_rotation(A):
    return np.allclose(A @ A.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(A), 1.0, atol=1e-9)


def total3d_R_from_yaw_pitch_roll(yaw, pitch, roll):
    """Copied from Total3D libs/tools.py::R_from_yaw_pitch_roll, so CAGE is tested against Total3D's own formula."""
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(yaw) * np.cos(pitch)
    R[0, 1] = np.sin(yaw) * np.sin(roll) - np.cos(yaw) * np.cos(roll) * np.sin(pitch)
    R[0, 2] = np.cos(roll) * np.sin(yaw) + np.cos(yaw) * np.sin(pitch) * np.sin(roll)
    R[1, 0] = np.sin(pitch)
    R[1, 1] = np.cos(pitch) * np.cos(roll)
    R[1, 2] = -np.cos(pitch) * np.sin(roll)
    R[2, 0] = -np.cos(pitch) * np.sin(yaw)
    R[2, 1] = np.cos(yaw) * np.sin(roll) + np.cos(roll) * np.sin(yaw) * np.sin(pitch)
    R[2, 2] = np.cos(yaw) * np.cos(roll) - np.sin(yaw) * np.sin(pitch) * np.sin(roll)
    return R


def total3d_bin_cls_reg(bins, angle):
    """Copied from Total3D libs/tools.py::bin_cls_reg: angle -> (bin index, residual in bin widths)."""
    centers = bins.mean(axis=1)
    index = int(np.abs(centers - angle).argmin())
    return index, float((angle - centers[index]) / (bins[0][1] - bins[0][0]))


def total3d_corners(basis_rows, coeffs, centroid):
    """Copied from Total3D net_utils/libs.py::get_corners_of_bb3d_no_index. coeffs are half-sizes."""
    return (frames.TOTAL3D_CORNER_SIGNS * coeffs) @ basis_rows + centroid


def total3d_box(orientation):
    """A gravity-aligned Total3D box in W: basis rows (side, up, front), half-sizes, centroid."""
    basis_rows = np.array([[np.cos(orientation), 0, -np.sin(orientation)], [0, 1, 0],
                           [np.sin(orientation), 0, np.cos(orientation)]])
    return basis_rows, np.array([0.4, 0.45, 0.25]), np.array([2.5, -0.8, 0.6])


TILTS_DEG = [(-35, -8), (-25, 0), (-5, 6), (10, 15), (0, 0)]


# ================================================================ 1. the registry: vocabulary and frames

def test_directions_are_unit_vectors_and_opposites_cancel():
    for name, v in DIRS.items():
        assert np.isclose(np.linalg.norm(v), 1.0), name
    for a, b in [("right", "left"), ("up", "down"), ("forward", "back")]:
        assert np.allclose(DIRS[a], -DIRS[b])
    assert np.allclose(DIRS["front"], DIRS["forward"])


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_frame_is_right_handed(name):
    """x cross y must equal z. Axis names that describe a left-handed frame are a typo."""
    B = basis(name)
    assert np.allclose(B.T @ B, np.eye(3)), f"{name}: axes are not mutually perpendicular"
    assert np.isclose(np.linalg.det(B), 1.0), f"{name}: axes {FRAMES[name]['axes']} are left-handed"


# ================================================================ 2. the registry: axis-change matrices

@pytest.mark.parametrize("name", sorted(CHANGES))
def test_axis_change_matches_frame_definitions(name):
    """p_to = B_to^T B_from p_from, built from the axis names alone, must equal the stored matrix."""
    entry = CHANGES[name]
    derived = basis(entry["to"]).T @ basis(entry["from"])
    assert np.allclose(entry["matrix"], derived), f"{name}: stored {entry['matrix']} but the frames imply\n{derived}"


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_axis_change_is_a_rotation_with_a_correct_involution_flag(name):
    A = np.array(CHANGES[name]["matrix"], float)
    assert is_rotation(A), f"{name} is not a proper rotation"
    assert np.allclose(A @ A, np.eye(3)) == CHANGES[name]["involution"], f"{name}: the 'involution' flag is wrong"
    p = np.random.default_rng(0).normal(size=(50, 3))
    assert np.allclose(p @ A.T @ A, p)                                       # A^T undoes A


def test_mirror_is_a_reflection():
    assert np.allclose(frames.M_MIRROR @ frames.M_MIRROR.T, np.eye(3)) and np.isclose(np.linalg.det(frames.M_MIRROR), -1.0)


def test_module_constants_are_the_registry_entries():
    assert np.allclose(frames.T_TOTAL3D, CHANGES["T"]["matrix"])
    assert np.allclose(frames.F_OPEN3D, CHANGES["F_open3d"]["matrix"])
    assert np.allclose(frames.P_PYTORCH3D, CHANGES["P_pytorch3d"]["matrix"])
    assert np.allclose(frames.MODEL_TO_BOX, CHANGES["MODEL_TO_BOX"]["matrix"])
    assert np.allclose(np.rad2deg(frames.PITCH_BINS), REGISTRY["total3d_bins_deg"]["pitch"])
    assert frames.DEPTH_PNG_SCALE == 256.0


# ================================================================ 3. elementary rotations and boxes

@pytest.mark.parametrize("rot", [rot_x, rot_y, rot_z])
def test_elementary_rotations(rot):
    assert is_rotation(rot(0.7)) and np.allclose(rot(0.0), np.eye(3)) and np.allclose(rot(0.4) @ rot(-0.4), np.eye(3))


def test_rotation_directions():
    assert np.allclose(rot_y(np.pi / 2) @ [1, 0, 0], [0, 0, -1])            # heading +90 deg faces the camera
    assert np.allclose(rot_x(np.pi / 2) @ [0, 0, 1], [0, -1, 0])            # +x rotation lifts forward to up
    assert np.allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0])


@pytest.mark.parametrize("heading", [-3.0, -1.56, 0.0, 0.5, 2.9])
def test_heading_and_length_axis_are_inverse(heading):
    assert np.allclose(frames.length_axis(heading), rot_y(heading) @ [1, 0, 0])
    assert np.isclose(frames.heading_of(frames.length_axis(heading)), heading)


def test_cage_corner_order():
    order = REGISTRY["box_corners"]["cage_order"]
    signs = frames.CAGE_CORNER_SIGNS
    assert len({tuple(s) for s in signs}) == 8
    assert all(signs[i, 1] > 0 for i in order["bottom_face"])               # +y is down
    assert all(signs[i, 0] > 0 for i in order["front_face"])                # +x is the length axis
    corners = box3d_corners([1.0, 2.0, 9.0], [4.0, 1.6, 1.5], 0.0)
    assert np.allclose(corners[0], [1 + 2.0, 2 + 0.75, 9 + 0.8])             # corner 0 = (+l/2, +h/2, +w/2)
    assert np.allclose(corners.mean(axis=0), [1.0, 2.0, 9.0])


def test_points_in_box_agrees_with_the_corners():
    center, size_lwh, heading = np.array([1.0, 1.0, 8.0]), [4.0, 1.6, 1.5], 0.6
    corners = box3d_corners(center, size_lwh, heading)
    assert points_in_box(center + 0.98 * (corners - center), center, size_lwh, heading).all()
    assert not points_in_box(center + 1.02 * (corners - center), center, size_lwh, heading).any()


# ================================================================ 4. camera

def test_unproject_then_project_returns_the_pixel_grid():
    K = np.array([[529.5, 0, 365.0], [0, 529.5, 265.0], [0, 0, 1.0]])
    depth = 1.0 + np.random.default_rng(1).random((53, 73)) * 7.0
    points, valid = unproject_depth(depth, K)
    v, u = np.mgrid[0:53, 0:73]
    assert valid.all() and np.allclose(points[..., 2], depth)               # depth is the z coordinate
    assert np.abs(project_points(points, K) - np.stack([u, v], -1)).max() < 1e-9


def test_depth_is_not_distance():
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    depth = np.full((480, 640), 2.0)
    points, _ = unproject_depth(depth, K)
    distance = np.linalg.norm(points, axis=-1)
    assert np.isclose(distance[240, 320], 2.0) and distance[0, 0] > 2.0 * 1.2   # equal on the axis, larger at a corner


# ================================================================ 5. KITTI

def test_split_kitti_projection_reproduces_P2():
    K = np.array([[721.5, 0, 609.6], [0, 721.5, 172.9], [0, 0, 1.0]])
    t = np.array([0.06, -0.0004, 0.003])
    P2 = np.c_[K, K @ t]
    K_read, t_read = frames.split_kitti_projection(P2)
    assert np.allclose(K_read, K) and np.allclose(t_read, t)
    p_rect = np.array([1.5, 0.8, 14.0])
    with_P2 = P2 @ np.append(p_rect, 1.0)
    assert np.allclose(project_points(p_rect + t_read, K_read), with_P2[:2] / with_P2[2])


def test_kitti_chain_worked_example():
    e = EXAMPLES["kitti_2011_09_26"]
    Tr, R0, P2 = np.array(e["Tr_velo_to_cam"]), np.array(e["R0_rect"]), np.array(e["P2"])
    K, t_cam2 = frames.split_kitti_projection(P2)
    assert np.allclose(t_cam2, e["expected_t_cam2"], atol=1e-6)
    p_cam2 = frames.velodyne_to_camera2(np.array([e["lidar_point"]]), Tr, R0, t_cam2)[0]
    assert np.allclose(p_cam2, e["expected_p_cam2"], atol=1e-5)
    assert np.allclose(project_points(p_cam2, K), e["expected_pixel"], atol=1e-3)
    with_P2 = P2 @ np.append(p_cam2 - t_cam2, 1.0)                           # the devkit's way: P2 on rectified camera 0
    assert np.allclose(project_points(p_cam2, K), with_P2[:2] / with_P2[2], atol=1e-6)


def test_kitti_calibration_is_close_to_the_ideal_axis_permutation():
    R = np.array(EXAMPLES["kitti_2011_09_26"]["Tr_velo_to_cam"])[:, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(frames.VELO_TO_CAM_IDEAL.T @ R) - 1) / 2, -1, 1)))
    assert angle < 2.0, f"the calibration is {angle:.2f} degrees from the ideal permutation"
    assert np.allclose(frames.VELO_TO_CAM_IDEAL @ [1, 0, 0], [0, 0, 1])     # LiDAR forward -> camera forward
    assert np.allclose(frames.VELO_TO_CAM_IDEAL @ [0, 1, 0], [-1, 0, 0])    # LiDAR left    -> camera minus-right
    assert np.allclose(frames.VELO_TO_CAM_IDEAL @ [0, 0, 1], [0, -1, 0])    # LiDAR up      -> camera minus-down


def test_kitti_label_worked_example():
    e = EXAMPLES["kitti_2011_09_26"]
    K, t_cam2 = frames.split_kitti_projection(np.array(e["P2"]))
    label = e["label"]
    center, size_lwh, heading = frames.kitti_label_to_box(label["dimensions_hwl"], label["location"], label["rotation_y"], t_cam2)
    assert np.allclose(center, e["expected_center_cam2"], atol=1e-5)
    assert size_lwh == [3.69, 1.67, 1.56] and heading == -1.56              # (h, w, l) reordered to (l, w, h)
    assert np.allclose(project_points(center, K), e["expected_center_pixel"], atol=1e-3)


def test_kitti_box_equals_the_devkit_box():
    """The devkit builds corners from the bottom center with y in {0, -h}; CAGE's corners must be the same points."""
    h, w, l, location, rotation_y = 1.5, 1.6, 3.9, np.array([0.5, 1.65, 12.0]), -1.2
    t_cam2 = np.array([0.06, 0.0, 0.003])
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    devkit = (rot_y(rotation_y) @ np.vstack([x, y, z])).T + location        # in rectified camera 0
    center, size_lwh, heading = frames.kitti_label_to_box([h, w, l], location, rotation_y, t_cam2)
    corners = box3d_corners(center, size_lwh, heading)
    assert np.allclose(corners, devkit + t_cam2)                            # same points, same order
    assert np.allclose(corners[:4].mean(axis=0), location + t_cam2)         # the bottom face sits on the labeled location


def test_kitti_label_round_trip():
    t_cam2 = np.array([0.06, 0.0, 0.003])
    dims, location, rotation_y = [1.5, 1.6, 3.9], [0.5, 1.65, 12.0], -1.2
    back = frames.box_to_kitti_label(*frames.kitti_label_to_box(dims, location, rotation_y, t_cam2), t_cam2)
    assert np.allclose(back[0], dims) and np.allclose(back[1], location) and back[2] == rotation_y


# ================================================================ 6. Total3D: camera tilt and gravity

@pytest.mark.parametrize("angle_deg", [-70, -40, -25, -5, 0, 12, 25])
def test_angle_from_bin_inverts_total3d_binning(angle_deg):
    angle = np.deg2rad(angle_deg)
    for bins in (frames.PITCH_BINS, frames.ROLL_BINS):
        assert np.isclose(frames.angle_from_bin(bins, *total3d_bin_cls_reg(bins, angle)), angle)


@pytest.mark.parametrize("pitch_deg, roll_deg", TILTS_DEG)
def test_camera_rotation_equals_total3d_formula(pitch_deg, roll_deg):
    p, r = np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
    cam_R = frames.total3d_camera_rotation(p, r)
    assert np.allclose(cam_R, total3d_R_from_yaw_pitch_roll(0.0, p, r))
    assert np.allclose(cam_R, rot_z(p) @ rot_x(r)) and is_rotation(cam_R)


@pytest.mark.parametrize("pitch_deg, roll_deg", TILTS_DEG)
def test_gravity_rotation(pitch_deg, roll_deg):
    p, r = np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
    R_gravity = frames.gravity_from_total3d(p, r)
    assert is_rotation(R_gravity) and np.allclose(R_gravity, rot_x(p) @ rot_z(r))
    # Total3D's own chain: world down (0, -1, 0) -> its camera axes (cam_R^T) -> OpenCV axes (T).
    gravity_in_camera = T @ frames.total3d_camera_rotation(p, r).T @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(R_gravity @ gravity_in_camera, [0.0, 1.0, 0.0])      # +y is down in the upright frame
    optical_axis = R_gravity @ [0.0, 0.0, 1.0]
    assert np.isclose(optical_axis[1], -np.sin(p)) and np.isclose(optical_axis[0], 0.0)   # pitch < 0 looks down; no yaw
    assert np.allclose(frames.pitch_roll_from_gravity(R_gravity), [p, r])
    if pitch_deg == roll_deg == 0:
        assert np.allclose(R_gravity, np.eye(3))                            # a level camera is already upright


@pytest.mark.parametrize("pitch_deg, roll_deg", TILTS_DEG)
def test_total3d_back_projection_and_projection_agree_with_R_gravity(pitch_deg, roll_deg):
    p, r = np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
    cam_R = total3d_R_from_yaw_pitch_roll(0.0, p, r)
    p_C = np.random.default_rng(2).normal(size=(50, 3)) + [0, 0, 4]
    p_W = p_C @ T.T @ cam_R.T                                               # Total3D rgb_to_world: p_W = cam_R T p_C
    assert np.allclose(p_W @ cam_R @ T.T, p_C)                              # Total3D project_3d_points_to_2d, before K
    assert np.allclose(frames.total3d_world_to_upright(p_W), p_C @ frames.gravity_from_total3d(p, r).T)


def test_floor_point_worked_example():
    e = EXAMPLES["sunrgbd_floor_point"]
    p_C = e["depth_z"] * np.linalg.inv(np.array(e["K"])) @ np.array([*e["pixel"], 1.0])
    R_gravity = frames.gravity_from_total3d(np.deg2rad(e["pitch_deg"]), np.deg2rad(e["roll_deg"]))
    assert np.allclose(R_gravity @ p_C, e["expected_p_upright"], atol=1e-5)


# ================================================================ 7. Total3D: boxes and meshes

def test_total3d_corner_order():
    assert len({tuple(s) for s in frames.TOTAL3D_CORNER_SIGNS}) == 8
    assert all(frames.TOTAL3D_CORNER_SIGNS[i, 1] > 0 for i in REGISTRY["box_corners"]["total3d_order"]["top_face"])


@pytest.mark.parametrize("orientation", [-2.5, -0.7, 0.0, 1.1, 3.0])
def test_box_from_total3d_corners(orientation):
    basis_rows, coeffs, centroid = total3d_box(orientation)
    corners_world = total3d_corners(basis_rows, coeffs, centroid)
    box = frames.box_from_total3d_corners(corners_world)
    assert np.allclose(box["center"], T @ centroid)
    assert np.allclose(box["size_lwh"], 2 * coeffs[[2, 0, 1]])              # length = front, width = side, height = up
    assert np.isclose(np.cos(box["heading"]), np.cos(-orientation)) and np.isclose(np.sin(box["heading"]), np.sin(-orientation))
    rebuilt = box3d_corners(**box)
    assert np.allclose(rebuilt[frames.TOTAL3D_TO_CAGE_CORNER], frames.total3d_world_to_upright(corners_world))


def test_place_model_points_equals_total3d_placement():
    basis_rows, coeffs, centroid = total3d_box(0.9)
    n = np.random.default_rng(3).uniform(-1, 1, (100, 3))
    by_total3d = frames.total3d_world_to_upright((n * coeffs) @ basis_rows + centroid)   # recover_points_to_world_sys
    box = frames.box_from_total3d_corners(total3d_corners(basis_rows, coeffs, centroid))
    assert np.allclose(frames.place_model_points(n, **box), by_total3d)


# ================================================================ 8. frustum frame

@pytest.mark.parametrize("ray", [[0.25, 0.2, 1.0], [-0.6, 0.1, 1.0], [0.0, -0.3, 1.0]])
def test_to_frustum_frame_puts_the_ray_on_the_z_axis(ray):
    ray = np.array(ray)
    angle = frames.frustum_angle(ray)
    rotated = frames.to_frustum_frame(ray[None], angle)[0]
    assert np.isclose(rotated[0], 0.0) and rotated[2] > 0 and np.isclose(rotated[1], ray[1])
    heading = 0.8                                                           # a heading h becomes h - angle
    assert np.isclose(frames.heading_of(frames.to_frustum_frame(frames.length_axis(heading)[None], angle)[0]), heading - angle)


def test_frustum_worked_example():
    """The Atlas car: 2D box center (791.6, 320.8) with KITTI's K gives 14.16 degrees."""
    K, _ = frames.split_kitti_projection(np.array(EXAMPLES["kitti_2011_09_26"]["P2"]))
    ray = np.linalg.inv(K) @ [791.603, 320.829, 1.0]
    assert np.isclose(np.degrees(frames.frustum_angle(ray)), 14.16, atol=0.01)


# ================================================================ 9. horizontal flip

@pytest.mark.parametrize("pitch_deg, roll_deg", TILTS_DEG)
def test_mirror_keeps_pitch_and_negates_roll(pitch_deg, roll_deg):
    p, r = np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
    assert np.allclose(frames.mirror_rotation(rot_x(p) @ rot_z(r)), rot_x(p) @ rot_z(-r))


def test_mirrored_box_lands_on_the_mirrored_image():
    e = EXAMPLES["kitti_flip_offset"]
    width, K = e["W"], np.array([[721.5377, 0, e["cx"]], [0, 721.5377, 172.854], [0, 0, 1.0]])
    center, size_lwh, heading = np.array([1.8, 1.5, 9.0]), [3.9, 1.6, 1.5], -1.2
    u = project_points(box3d_corners(center, size_lwh, heading), K)[:, 0]
    mirrored = box3d_corners(frames.mirror_points(center[None])[0], size_lwh, frames.mirror_heading(heading))
    u_mirrored = project_points(mirrored, frames.mirror_K(K, width))[:, 0]
    assert np.allclose(sorted(u_mirrored), sorted(width - 1 - u))           # every corner lands on its mirrored pixel
    assert frames.mirror_box2d([u.min(), 0, u.max(), 1], width) == pytest.approx([u_mirrored.min(), 0, u_mirrored.max(), 1])
    u_old_K = project_points(mirrored, K)[:, 0]                             # the mistake: mirrored 3D, old intrinsics
    assert np.isclose(u_mirrored.min() - u_old_K.min(), e["expected_offset_px"], atol=1e-3)


def test_mirror_heading_is_wrapped_and_self_inverse():
    for heading in [-3.0, -1.56, 0.0, 0.5, 2.9]:
        mirrored = frames.mirror_heading(heading)
        assert -np.pi <= mirrored < np.pi
        assert np.allclose(frames.length_axis(mirrored), frames.M_MIRROR @ frames.length_axis(heading))
        assert np.isclose(np.cos(frames.mirror_heading(mirrored)), np.cos(heading))


# ================================================================ 10. viewers, encodings, class lists

def test_open3d_flip_puts_up_up():
    up_in_camera = np.array([[0.0, -1.0, 0.0]])                             # one meter above the camera
    assert np.allclose(frames.to_open3d(up_in_camera), [[0.0, 1.0, 0.0]])
    assert np.allclose(frames.to_open3d(np.array([[0.0, 0.0, 5.0]])), [[0.0, 0.0, -5.0]])   # in front = along -z in Open3D


def test_moge_normalized_intrinsics_round_trip():
    offset = REGISTRY["encodings"]["pixel_center_offset_moge"]
    W, H, K = 640, 480, np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]])
    Kn = np.array([[K[0, 0] / W, 0, (K[0, 2] + offset) / W], [0, K[1, 1] / H, (K[1, 2] + offset) / H], [0, 0, 1.0]])
    assert np.isclose(Kn[0, 2], 0.5) and np.isclose(Kn[1, 2], 0.5)          # a centered camera has its center at 0.5
    back = np.array([[Kn[0, 0] * W, 0, Kn[0, 2] * W - offset], [0, Kn[1, 1] * H, Kn[1, 2] * H - offset], [0, 0, 1.0]])
    assert np.allclose(back, K)


def test_class_lists():
    assert len(frames.NYU40_CLASSES) == 41 and frames.NYU40_CLASSES[0] == "void" and frames.NYU40_CLASSES[-1] == "otherprop"
    # Total3D's size_cls is an index into this list (total3d/configs/data_config.py), so the positions matter, not only the names.
    positions = {"bed": 4, "chair": 5, "sofa": 6, "table": 7, "bookshelf": 10, "desk": 14, "dresser": 17,
                 "night_stand": 32, "toilet": 33, "bathtub": 36}
    assert {name: frames.NYU40_CLASSES.index(name) for name in frames.SUNRGBD_CLASSES} == positions
    assert len(frames.PIX3D_CLASSES) == 9 and frames.PIX3D_CLASSES[0] == "misc"
    assert set(frames.SUNRGBD_CLASSES) <= set(frames.NYU40_CLASSES)
    assert set(frames.KINS_TO_KITTI_CLASS) <= set(REGISTRY["classes"]["kins_categories"].values())
    assert "rider" not in frames.KINS_TO_KITTI_CLASS                        # KITTI has no rider class
    assert set(frames.KITTI_CLASSES) <= set(frames.KINS_TO_KITTI_CLASS.values())
    for values in (frames.NYU40_CLASSES, frames.PIX3D_CLASSES, frames.SUNRGBD_CLASSES, frames.KITTI_CLASSES):
        assert len(values) == len(set(values))


# ================================================================ 11. one definition only

FORBIDDEN = {
    r"\[\s*0(\.0)?\s*,\s*0(\.0)?\s*,\s*1(\.0)?\s*\]\s*,\s*\[\s*0(\.0)?\s*,\s*-1(\.0)?\s*,\s*0(\.0)?\s*\]": "the Total3D axis swap T",
    r"diag\(\s*\[?\s*1(\.0)?\s*,\s*-1(\.0)?\s*,\s*-1(\.0)?": "the Open3D flip F",
    r"diag\(\s*\(?\[?\s*-1(\.0)?\s*,\s*1(\.0)?\s*,\s*1(\.0)?": "the mirror matrix M",
    r"deg2rad\(\s*\[\s*\[": "Total3D's angle bins",
    r"[\"']night_stand[\"']": "a SUN RGB-D class list",
    r"[\"']wardrobe[\"']": "the Pix3D class list",
    r"[\"']Cyclist[\"']\s*[,\]\)]": "a KITTI class list",
}


def test_conventions_are_defined_only_in_the_registry():
    """No file under cage/ or scripts/ may spell out a convention that the registry owns."""
    offenders = []
    for folder in ("cage", "scripts"):
        for path in sorted((REPO / folder).rglob("*.py")):
            if path.name == "frames.py":
                continue
            text = path.read_text()
            offenders += [f"{path.relative_to(REPO)} defines {what}" for pattern, what in FORBIDDEN.items() if re.search(pattern, text)]
    assert not offenders, "import these from cage.geometry.frames instead:\n  " + "\n  ".join(offenders)
