"""Tests for cage/data/convert_sunrgbd.py on synthetic Total3D pickles. No dataset is needed.

The synthetic samples are built with Total3D's own formulas, ported below from its repository, so the
tests check the whole converter against the convention that produced the real pickles. The frame math
itself (axis swap, angle bins, gravity rotation, box conversion) is tested in tests/test_frames.py.
"""
import json
import pickle

import numpy as np
import pytest

from cage.data import convert_sunrgbd
from cage.data.sample_io import read_sample, read_split
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points
from cage.geometry.frames import NYU40_CLASSES, PITCH_BINS, ROLL_BINS, T_TOTAL3D, total3d_camera_rotation

WIDTH, HEIGHT = 64, 48
K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
CHAIR = NYU40_CLASSES.index("chair")
TILTS = [(0.0, 0.0), (-0.30, 0.10), (0.20, -0.15)]                # (pitch, roll) in radians


def total3d_corners(basis, coeffs, centroid):
    """total3d/utils/sunrgbd_utils.py::get_corners_of_bb3d_no_index."""
    signs = [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, -1, -1), (-1, -1, 1), (1, -1, 1), (1, -1, -1)]
    return np.array([sum(sign * basis[axis] * coeffs[axis] for axis, sign in enumerate(row)) for row in signs]) + centroid


def total3d_project(points_world, cam_R):
    """total3d/utils/sunrgbd_utils.py::proj_from_point_to_2d: world points -> pixels."""
    points_cam = points_world @ cam_R @ T_TOTAL3D.T
    return (points_cam / points_cam[:, 2:]) @ K.T


def total3d_bin(bins, angle):
    """total3d/libs/tools.py::bin_cls_reg: angle -> (bin index, residual in bin widths)."""
    centers = bins.mean(axis=1)
    index = int(np.abs(centers - angle).argmin())
    return index, float((angle - centers[index]) / (bins[0][1] - bins[0][0]))


def world_box(yaw, half_sizes, centroid):
    """A gravity-aligned box in Total3D's world (x forward, y up, z right): its basis rows are side, up, front."""
    front, up = np.array([np.cos(yaw), 0.0, np.sin(yaw)]), np.array([0.0, 1.0, 0.0])
    return np.stack([np.cross(up, front), up, front]), np.array(half_sizes), np.array(centroid)


def total3d_sample(pitch, roll, sequence_id=5051, boxes=None, masks=None):
    """A dict with the keys and conventions of one Total3D SUN RGB-D pickle."""
    cam_R = total3d_camera_rotation(pitch, roll)
    boxes = boxes or [world_box(0.4, [0.3, 0.4, 0.6], [3.0, -0.2, 0.3])]
    corner_sets = [total3d_corners(*box) for box in boxes]
    boxes_2d = []
    for corners in corner_sets:
        pixels = total3d_project(corners, cam_R)
        boxes_2d.append([max(pixels[:, 0].min(), 0), max(pixels[:, 1].min(), 0),
                         min(pixels[:, 0].max(), WIDTH - 1), min(pixels[:, 1].max(), HEIGHT - 1)])
    principal_point = np.array([K[0, 2], K[1, 2], K[0, 2], K[1, 2]])
    pitch_cls, pitch_reg = total3d_bin(PITCH_BINS, pitch)
    roll_cls, roll_reg = total3d_bin(ROLL_BINS, roll)
    depth = np.full((HEIGHT, WIDTH), 2.5, np.float32)
    depth[:4] = 8.0                                               # rows that Total3D clipped at 8 m
    rng = np.random.default_rng(sequence_id)
    return {"rgb_img": rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8), "depth_map": depth,
            "boxes": {"bdb3D": np.array(corner_sets), "bdb2D_from_3D": np.array(boxes_2d) / principal_point,
                      "bdb2D_pos": np.array(boxes_2d).round(), "size_cls": np.array([CHAIR] * len(boxes)),
                      "mask": np.array(masks or [None] * len(boxes), dtype=object)},
            "camera": {"K": K, "pitch_cls": pitch_cls, "pitch_reg": pitch_reg, "roll_cls": roll_cls, "roll_reg": roll_reg},
            "layout": {"bdb3D": total3d_corners(*world_box(0.1, [2.0, 1.4, 3.0], [2.5, 0.2, 0.0]))},
            "sequence_id": sequence_id}


@pytest.mark.parametrize("pitch, roll", TILTS)
def test_converted_box_projects_onto_total3d_2d_box(tmp_path, pitch, roll):
    data = total3d_sample(pitch, roll)
    convert_sunrgbd.convert_one(data, "05051", "train", tmp_path)
    sample = read_sample(tmp_path, "05051")
    obj, R_gravity = sample["objects"][0], np.array(sample["meta"]["R_gravity"])
    corners_camera = box3d_corners(**obj["box3d"]) @ R_gravity               # upright -> camera, as check_dataset.py does
    assert (corners_camera[:, 2] > 0).all()
    pixels = project_points(corners_camera, np.array(sample["meta"]["K"]))
    reprojected = [max(pixels[:, 0].min(), 0), max(pixels[:, 1].min(), 0),
                   min(pixels[:, 0].max(), WIDTH - 1), min(pixels[:, 1].max(), HEIGHT - 1)]
    stored_by_total3d = data["boxes"]["bdb2D_from_3D"][0] * [K[0, 2], K[1, 2], K[0, 2], K[1, 2]]
    assert np.allclose(reprojected, stored_by_total3d, atol=1e-6)
    assert np.allclose(obj["bbox2d_amodal"], stored_by_total3d)
    assert abs(np.array(obj["box3d"]["center"])[1] - 0.2) < 1e-9             # world y = -0.2 (up) is +0.2 down


def test_convert_one_writes_masks_depth_layout_and_class(tmp_path):
    crop = np.ones((11, 21), bool)                                            # Total3D crops rows y1..y2 and columns x1..x2 inclusive
    mask = {"msk_bdb": [10, 5, 30, 15], "msk": crop, "class_id": CHAIR}
    convert_sunrgbd.convert_one(total3d_sample(-0.2, 0.05, masks=[mask]), "05051", "val", tmp_path)
    sample = read_sample(tmp_path, "05051")
    expected = np.zeros((HEIGHT, WIDTH), bool)
    expected[5:16, 10:31] = True
    assert ((sample["instances"] == 1) == expected).all()
    assert sample["objects"][0]["class"] == "chair" and sample["objects"][0]["has_mask"]
    assert (sample["depth"][:4] == 0).all() and np.allclose(sample["depth"][4:], 2.5)   # clipped rows are invalid
    assert np.allclose(sorted(sample["layout"]["size_lwh"]), [2.8, 4.0, 6.0])
    assert sample["meta"]["split"] == "val" and sample["meta"]["has_layout"] and sample["meta"]["label_type"] == "sensor"


def test_nearer_object_wins_where_masks_overlap(tmp_path):
    boxes = [world_box(0.0, [0.3, 0.4, 0.3], [5.0, 0.0, 0.0]), world_box(0.0, [0.3, 0.4, 0.3], [2.0, 0.0, 0.0])]
    masks = [{"msk_bdb": [10, 5, 30, 15], "msk": np.ones((11, 21), bool)},     # far object (5 m)
             {"msk_bdb": [20, 5, 40, 15], "msk": np.ones((11, 21), bool)}]     # near object (2 m)
    convert_sunrgbd.convert_one(total3d_sample(0.0, 0.0, boxes=boxes, masks=masks), "05051", "train", tmp_path)
    instances = read_sample(tmp_path, "05051")["instances"]
    assert instances[10, 25] == 2 and instances[10, 12] == 1 and instances[10, 38] == 2


def write_split_files(split_dir, train_ids, test_ids):
    split_dir.mkdir(parents=True)
    folder = "./data/sunrgbd/sunrgbd_train_test_data"
    train = [f"{folder}/{i}{suffix}.pkl" for i in train_ids for suffix in ("", "_flip")]
    (split_dir / "train.json").write_text(json.dumps(train))
    (split_dir / "test.json").write_text(json.dumps([f"{folder}/{i}.pkl" for i in test_ids]))


def test_make_splits_holds_out_every_20th_training_id(tmp_path):
    write_split_files(tmp_path / "splits", range(5051, 5151), range(1, 11))
    splits = convert_sunrgbd.make_splits(tmp_path / "splits")
    assert splits["test"] == list(range(1, 11))
    assert splits["val"] == [5051, 5071, 5091, 5111, 5131]
    assert len(splits["train"]) == 95 and not set(splits["train"]) & set(splits["val"])


def test_main_converts_listed_pickles_and_reports_missing_ones(tmp_path, capsys):
    raw, processed, split_dir = tmp_path / "raw", tmp_path / "processed", tmp_path / "splits"
    folder = raw / "sunrgbd/total3d_processed/sunrgbd_train_test_data"        # one nesting level, as the archive may have
    folder.mkdir(parents=True)
    for sequence_id in (7, 5051, 5052):
        with open(folder / f"{sequence_id}.pkl", "wb") as f:
            pickle.dump(total3d_sample(-0.1, 0.0, sequence_id), f)
    (folder / "5052_flip.pkl").write_bytes(b"never opened")
    write_split_files(split_dir, [5051, 5052, 5053], [7])                     # 5053 has no pickle
    convert_sunrgbd.main(raw, processed, split_dir)
    root = processed / "sunrgbd"
    assert read_split(root, "test") == ["00007"] and read_split(root, "val") == ["05051"]
    assert read_split(root, "train") == ["05052"]
    assert read_sample(root, "00007")["meta"]["raw_ref"] == "sunrgbd/7"
    assert "1 ids in the split files have no pickle" in capsys.readouterr().out
