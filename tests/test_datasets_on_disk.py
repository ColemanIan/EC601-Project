"""Tests on the real data under $CAGE_DATA (plan task D8.2). They run only when asked for:

pytest -m data                       # every test in this file
pytest -m data -k sunrgbd            # the tests whose name contains "sunrgbd"

Plain `pytest` skips this file (pyproject.toml deselects the `data` marker), because the laptop has no datasets.
A dataset that is absent FAILS here; it is not skipped. The purpose of the file is to prove the data exists.

What the file establishes, in order:
  1. every download is in raw/ with the expected number of files           (and writes processed/RAW_INVENTORY.md)
  2. every processed dataset is complete and self-consistent                (and writes processed/<dataset>/REPORT.md)
  3. the metadata has the values each dataset must have                     (splits, camera tilt, classes, flags)
  4. processed samples agree with their raw files, recomputed here with the SOURCE's own formulas
     (the KITTI devkit's box corners and P2 projection; Total3D's camera rotation and projection)
"""
import json
import pickle

import numpy as np
import pytest

from cage.config import PROCESSED, RAW
from cage.data import report
from cage.data.convert_kitti import read_calib, read_labels
from cage.data.sample_io import read_sample, read_split
from cage.geometry import frames
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points

pytestmark = pytest.mark.data

# Files that a complete download holds: (folder under raw/, glob pattern, number of files).
RAW_COUNTS = [
    ("kitti/object/training/image_2", "*.png", 7481),
    ("kitti/object/training/label_2", "*.txt", 7481),
    ("kitti/object/training/calib", "*.txt", 7481),
    ("kitti/object/training/velodyne", "*.bin", 7481),
    ("kitti/depth/depth_selection/val_selection_cropped/image", "*.png", 1000),
    ("kitti/depth/depth_selection/val_selection_cropped/groundtruth_depth", "*.png", 1000),
    ("kitti/depth/depth_selection/val_selection_cropped/intrinsics", "*.txt", 1000),
    ("kitti/kins", "update_train_2020.json", 1),
    ("pix3d", "pix3d.json", 1),
]
# Split sizes that follow from the split files CAGE reads (Frustum PointNets' for KITTI, Total3D's for the indoor sets).
SPLITS = {
    "kitti_depth_val": {"train": 0, "val": 0, "test": 1000},
    "kitti_object": {"train": 3712, "val": 3769, "test": 0},
    "pix3d": {"train": 7178, "val": 378, "test": 2513},          # Total3D's 7,556 training images, every 20th held out
}

_reports = {}


def report_of(name):
    """Build and write the report of one dataset once per test run."""
    if name not in _reports:
        _reports[name] = report.write_report(name)
    return _reports[name]


# ---------------------------------------------------------------- 1. the downloads

@pytest.mark.parametrize("folder, pattern, count", RAW_COUNTS)
def test_raw_folder_holds_the_complete_download(folder, pattern, count):
    assert (RAW / folder).is_dir(), f"{RAW / folder} does not exist"
    found = len(list((RAW / folder).glob(pattern)))
    assert found == count, f"{RAW / folder} holds {found} files matching {pattern}, a complete download holds {count}"


@pytest.mark.parametrize("folder, at_least", [("sunrgbd/total3d_processed", 10000), ("pix3d/total3d_processed", 10069)])
def test_raw_pickles_are_total3d_per_image_pickles(folder, at_least):
    """Total3D names each pickle <number>.pkl; its flipped copies carry a suffix and are not counted."""
    pickles = [path for path in (RAW / folder).rglob("*.pkl") if path.stem.isdigit()]
    assert len(pickles) >= at_least, f"{RAW / folder} holds {len(pickles)} numbered pickles, expected at least {at_least}"


def test_raw_inventory_is_written():
    inventory = report.write_raw_inventory()
    assert all(entry["exists"] for entry in inventory.values()), [name for name, entry in inventory.items() if not entry["exists"]]
    assert (PROCESSED / "RAW_INVENTORY.md").exists() and (PROCESSED / "raw_inventory.json").exists()


# ---------------------------------------------------------------- 2. every processed dataset: complete and consistent

@pytest.mark.parametrize("name", report.REQUIRED)
def test_processed_dataset_exists_and_its_report_finds_no_problem(name):
    result = report_of(name)
    assert result["exists"], f"{result['path']} does not exist; run its converter (plan, D9)"
    assert result["problems"] == [], "\n".join(result["problems"])
    assert (PROCESSED / name / "REPORT.md").exists() and (PROCESSED / name / "report.json").exists()


@pytest.mark.parametrize("name", ["kitti_object_occ", "occluders_indoor"])
def test_derived_dataset_report(name):
    """The two occlusion sets are built in D6. Before that this test is skipped; afterwards it must pass."""
    if not (PROCESSED / name).exists():
        pytest.skip(f"{name} is not built yet (D6)")
    result = report_of(name)
    assert result["problems"] == [], "\n".join(result["problems"])
    assert result["summary"]["samples"] > 0


# ---------------------------------------------------------------- 3. the metadata each dataset must have

@pytest.mark.parametrize("name", sorted(SPLITS))
def test_split_sizes(name):
    assert report_of(name)["summary"]["splits"] == SPLITS[name]


def test_sunrgbd_splits_follow_total3d():
    """Total3D tests on sequence ids 1 to 5050 and trains on the rest; CAGE holds out every 20th training image."""
    root = PROCESSED / "sunrgbd"
    train, val, test = (read_split(root, split) for split in ("train", "val", "test"))
    assert max(int(i) for i in test) <= 5050 < min(int(i) for i in train + val)
    assert 4900 <= len(test) <= 5050 and 5000 <= len(train) + len(val) <= 5285
    assert abs(len(val) - (len(train) + len(val)) / 20) <= 1


@pytest.mark.parametrize("name", ["kitti_depth_val", "kitti_object"])
def test_kitti_camera_is_level(name):
    summary = report_of(name)["summary"]
    assert summary["R_gravity"]["identity_in"] == summary["samples"]


def test_kitti_object_metadata():
    summary = report_of("kitti_object")["summary"]
    assert summary["flags"]["has_boxes3d"] == 7481
    assert summary["flags"].get("has_depth", 0) == 7481, "run: python -m cage.data.convert_kitti object_depth"
    assert summary["flags"].get("has_masks", 0) >= 7400, "run: python -m cage.data.convert_kins"
    assert set(frames.KITTI_CLASSES) <= set(summary["objects"]["per_class"])
    # Camera 2 sits about 6 cm to the left of camera 0, so camera 0 is at x = +0.06 m in camera 2: t_cam2 = K^-1 P2[:, 3].
    assert 0.04 < summary["t_cam2"]["min"][0] and summary["t_cam2"]["max"][0] < 0.08
    assert 690 < summary["K"]["fx"]["min"] and summary["K"]["fx"]["max"] < 730


def test_kitti_depth_val_metadata():
    summary = report_of("kitti_depth_val")["summary"]
    assert summary["image_sizes"] == {"1216 x 352": 1000}
    assert summary["depth_source"] == {"kitti_annotated": 1000}
    assert summary["objects"]["total"] == 0


def test_sunrgbd_metadata():
    summary = report_of("sunrgbd")["summary"]
    n = summary["samples"]
    assert all(summary["flags"].get(flag, 0) == n for flag in ("has_depth", "has_masks", "has_boxes3d", "has_layout"))
    assert set(summary["objects"]["per_class"]) <= set(frames.NYU40_CLASSES)
    assert set(frames.SUNRGBD_CLASSES) <= set(summary["objects"]["per_class"])
    # Total3D stores the tilt in bins that span these ranges, so no decoded angle can lie outside them.
    low, high = np.degrees(frames.PITCH_BINS).min(), np.degrees(frames.PITCH_BINS).max()
    assert low - 20 <= summary["R_gravity"]["pitch_deg"]["min"] and summary["R_gravity"]["pitch_deg"]["max"] <= high + 20
    assert summary["R_gravity"]["pitch_deg"]["mean"] < 0, "indoor cameras look down on average; a positive mean means the pitch sign is inverted"
    assert abs(summary["R_gravity"]["roll_deg"]["mean"]) < 5


def test_pix3d_metadata():
    summary = report_of("pix3d")["summary"]
    assert set(summary["per_class"]) == set(frames.PIX3D_CLASSES)
    assert set(summary["occlusion"]) <= {"none", "slight", "occluded"}


# ---------------------------------------------------------------- 4. processed samples against their raw files

def spread(ids, n=12):
    """`n` ids spread evenly over a list."""
    return [ids[i] for i in np.linspace(0, len(ids) - 1, min(n, len(ids))).astype(int)]


def test_kitti_object_boxes_equal_the_devkit_boxes():
    """Rebuild each box from the raw label with the devkit's own corner formula and project it with the raw P2.

    The devkit works in rectified camera 0 and projects with the full P2; CAGE stores the box in camera 2 and
    projects with K. Both must give the same pixels.
    """
    root, raw = PROCESSED / "kitti_object", RAW / "kitti/object/training"
    compared = 0
    for sample_id in spread(read_split(root, "train") + read_split(root, "val")):
        sample = read_sample(root, sample_id)
        calib = read_calib(raw / "calib" / f"{sample_id}.txt")
        labels = [label for label in read_labels(raw / "label_2" / f"{sample_id}.txt") if label["type"] != "DontCare"]
        K = np.array(sample["meta"]["K"])
        assert np.allclose(K, calib["P2"][:, :3])
        for label, obj in zip(labels, sample["objects"]):                     # the converter keeps the label order
            assert obj["class"] == label["type"]
            h, w, l = label["dimensions"]
            x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2                    # devkit, cpp/evaluate_object.cpp and readme.txt
            y = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=float)
            z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
            devkit_corners = (frames.rot_y(label["rotation_y"]) @ np.vstack([x, y, z])).T + label["location"]
            box = obj["box3d"]
            corners = box3d_corners(box["center"], box["size_lwh"], box["heading"])
            assert np.allclose(corners - sample["meta"]["t_cam2"], devkit_corners, atol=1e-6)
            if (devkit_corners[:, 2] > 0.1).all():
                homogeneous = np.hstack([devkit_corners, np.ones((8, 1))]) @ calib["P2"].T
                assert np.allclose(project_points(corners, K), homogeneous[:, :2] / homogeneous[:, 2:], atol=1e-4)
            compared += 1
    assert compared > 0


def test_kitti_depth_val_equals_the_raw_depth_maps():
    root, selection = PROCESSED / "kitti_depth_val", RAW / "kitti/depth/depth_selection/val_selection_cropped"
    for sample_id in spread(read_split(root, "test"), 5):
        from cage.data.io_depth import read_depth_png
        raw_depth = read_depth_png(selection / "groundtruth_depth" / (sample_id.replace("_image_", "_groundtruth_depth_", 1) + ".png"))
        sample = read_sample(root, sample_id)
        assert np.array_equal(sample["depth"], raw_depth)
        assert np.allclose(sample["meta"]["K"], np.loadtxt(selection / "intrinsics" / f"{sample_id}.txt").reshape(3, 3))


def test_sunrgbd_boxes_equal_total3d_boxes_in_the_camera_frame():
    """Move Total3D's raw corners into the camera with Total3D's own formulas, and compare with the processed box.

    Total3D: p_camera(toward, up, right) = cam_R^T p_world, then the axis swap T gives OpenCV axes.
    CAGE:    p_camera = R_gravity^T p_upright.
    The two routes share no code after the raw pickle except T and the bin tables, which tests/test_frames.py checks.
    """
    root = PROCESSED / "sunrgbd"
    pickle_path = {int(path.stem): path for path in (RAW / "sunrgbd/total3d_processed").rglob("*.pkl") if path.stem.isdigit()}
    compared = 0
    for sample_id in spread(read_split(root, "train") + read_split(root, "test")):
        with open(pickle_path[int(sample_id)], "rb") as f:
            data = pickle.load(f)
        sample = read_sample(root, sample_id)
        camera = data["camera"]
        assert np.array_equal(sample["image"], np.asarray(data["rgb_img"])[..., :3])
        assert np.allclose(sample["meta"]["K"], camera["K"])
        raw_depth = np.array(data["depth_map"], dtype=np.float32)
        measured = (raw_depth > 0) & (raw_depth < frames.SUNRGBD_DEPTH_CAP_M)
        assert np.allclose(sample["depth"][measured], raw_depth[measured], atol=1 / frames.DEPTH_PNG_SCALE)
        assert (sample["depth"][~measured] == 0).all()
        pitch = frames.angle_from_bin(frames.PITCH_BINS, camera["pitch_cls"], camera["pitch_reg"])
        roll = frames.angle_from_bin(frames.ROLL_BINS, camera["roll_cls"], camera["roll_reg"])
        cam_R = frames.total3d_camera_rotation(pitch, roll)
        R_gravity = np.array(sample["meta"]["R_gravity"])
        assert len(sample["objects"]) == len(data["boxes"]["bdb3D"])
        for index, obj in enumerate(sample["objects"]):
            assert obj["class"] == frames.NYU40_CLASSES[int(data["boxes"]["size_cls"][index])]
            total3d_camera = np.asarray(data["boxes"]["bdb3D"][index]) @ cam_R @ frames.T_TOTAL3D.T     # [8, 3], OpenCV axes
            box = obj["box3d"]
            cage_camera = box3d_corners(box["center"], box["size_lwh"], box["heading"]) @ R_gravity     # [8, 3], CAGE order
            assert np.allclose(cage_camera[frames.TOTAL3D_TO_CAGE_CORNER], total3d_camera, atol=0.02), (sample_id, obj["id"])
            compared += 1
    assert compared > 0


def test_pix3d_samples_equal_their_pickles():
    root = PROCESSED / "pix3d"
    pickle_path = {int(path.stem): path for path in (RAW / "pix3d/total3d_processed").rglob("*.pkl") if path.stem.isdigit()}
    entries = json.load(open(RAW / "pix3d/pix3d.json"))
    for sample_id in spread(read_split(root, "train") + read_split(root, "test")):
        with open(pickle_path[int(sample_id)], "rb") as f:
            data = pickle.load(f)
        folder = root / "samples" / sample_id
        meta = json.load(open(folder / "meta.json"))
        assert int(data["sample_id"]) == int(sample_id) and meta["class_id"] == int(data["class_id"])
        assert np.array_equal(np.load(folder / "points.npy"), np.asarray(data["gt_3dpoints"], dtype=np.float32))
        assert meta["class"] == entries[int(sample_id)]["category"]            # Total3D's class index names Pix3D's own category
