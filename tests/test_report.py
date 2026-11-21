"""Tests for cage/data/report.py on small synthetic datasets written into a temporary folder. No dataset is needed.

Each test builds a dataset whose contents are known, so the numbers in the report can be asserted exactly,
and then damages it in one way to confirm that the report names the damage.
"""
import json

import numpy as np
import pytest

from cage.data import report
from cage.data.sample_io import write_sample, write_splits
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points
from cage.geometry.frames import PIX3D_CLASSES, T_TOTAL3D, gravity_from_total3d

WIDTH, HEIGHT = 192, 128
K = np.array([[200.0, 0.0, 96.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]])
CENTER_CAMERA = np.array([0.2, 0.1, 4.0])                             # every synthetic box sits here in the camera frame


def scene_sample(root, sample_id, split, dataset, R_gravity=np.eye(3)):
    """Write one scene sample: a single box 4 m ahead of the camera, and a patch of depth at the box's center.

    The box is stored the way every converter stores it, in the upright frame: center_U = R_gravity center_C.
    """
    box = {"center": (R_gravity @ CENTER_CAMERA).tolist(), "size_lwh": [1.0, 0.8, 0.6], "heading": 0.3}
    corners_camera = box3d_corners(box["center"], box["size_lwh"], box["heading"]) @ R_gravity     # p_C = R_gravity^T p_U
    pixels = project_points(corners_camera, K)
    box2d = [float(max(pixels[:, 0].min(), 0)), float(max(pixels[:, 1].min(), 0)),
             float(min(pixels[:, 0].max(), WIDTH - 1)), float(min(pixels[:, 1].max(), HEIGHT - 1))]
    depth = np.zeros((HEIGHT, WIDTH), np.float32)
    u, v = project_points(CENTER_CAMERA[None], K)[0].round().astype(int)
    depth[v - 2:v + 3, u - 2:u + 3] = CENTER_CAMERA[2]
    instances = np.zeros((HEIGHT, WIDTH), np.uint16)
    instances[v - 2:v + 3, u - 2:u + 3] = 1
    meta = {"dataset": dataset, "sample_id": sample_id, "split": split, "width": WIDTH, "height": HEIGHT,
            "K": K.tolist(), "R_gravity": np.asarray(R_gravity).tolist(), "depth_source": "synthetic", "label_type": "synthetic",
            "has_depth": True, "has_masks": True, "has_amodal_masks": False, "has_boxes3d": True, "has_layout": False,
            "raw_ref": sample_id}
    if dataset.startswith("kitti"):
        meta["t_cam2"] = [0.06, 0.0, 0.003]
    objects = [{"id": 1, "class": "Car" if dataset.startswith("kitti") else "chair", "bbox2d_visible": box2d,
                "bbox2d_amodal": box2d, "box3d": box, "truncated": None, "occluded": None, "has_mask": True, "mask_amodal": None}]
    write_sample(root, sample_id, np.full((HEIGHT, WIDTH, 3), 90, np.uint8), meta, objects, depth=depth, instances=instances)


@pytest.fixture
def kitti(tmp_path):
    """processed/kitti_object with three samples: two for training, one for validation."""
    root = tmp_path / "processed" / "kitti_object"
    for sample_id, split in (("000000", "train"), ("000001", "train"), ("000002", "val")):
        scene_sample(root, sample_id, split, "kitti_object")
    write_splits(root, {"train": ["000000", "000001"], "val": ["000002"]})
    return tmp_path / "processed"


@pytest.fixture
def pix3d(tmp_path):
    """processed/pix3d with two samples whose points fill the cube [-1, 1]^3 up to its faces."""
    root = tmp_path / "processed" / "pix3d"
    rng = np.random.default_rng(0)
    for sample_id, class_id in (("7", 3), ("9", 5)):
        folder = root / "samples" / sample_id
        folder.mkdir(parents=True)
        points = rng.uniform(-1, 1, (10000, 3)).astype(np.float32)
        points[:6] = np.concatenate([np.eye(3), -np.eye(3)])           # one point on each face, so the extent is exactly 1
        np.save(folder / "points.npy", points)
        (folder / "crop.png").write_bytes(b"not read by the report")
        (folder / "meta.json").write_text(json.dumps({"dataset": "pix3d", "sample_id": sample_id, "class": PIX3D_CLASSES[class_id],
                                                       "class_id": class_id, "split": "train", "occlusion": "none", "truncated": False}))
    write_splits(root, {"train": ["7", "9"]})
    return tmp_path / "processed"


def problems_of(name, processed):
    return report.build_report(name, processed, check_count=False)["problems"]


# ---------------------------------------------------------------- counting files

def test_count_files_groups_by_extension_and_by_folder(tmp_path):
    (tmp_path / "a" / "deep" / "deeper").mkdir(parents=True)
    for relative in ("a/one.png", "a/two.png", "a/deep/three.txt", "a/deep/deeper/four.txt", "top"):
        (tmp_path / relative).write_text("x")
    counted = report.count_files(tmp_path, depth=2)
    assert counted["files"] == 5
    assert counted["by_extension"] == {"(none)": 1, ".png": 2, ".txt": 2}
    assert counted["by_folder"] == {".": {"(none)": 1}, "a": {".png": 2}, "a/deep": {".txt": 2}}   # deeper/ is cut to a/deep


def test_count_files_leaves_out_the_report_files(tmp_path):
    for name in ("REPORT.md", "report.json", "train.txt"):
        (tmp_path / name).write_text("x")
    assert report.count_files(tmp_path)["by_extension"] == {".txt": 1}


def test_raw_inventory_reports_missing_and_present_downloads(tmp_path):
    (tmp_path / "raw/kitti/kins").mkdir(parents=True)
    (tmp_path / "raw/kitti/kins/update_train_2020.json").write_text("{}")
    inventory = report.write_raw_inventory(tmp_path / "raw", tmp_path / "processed")
    assert inventory["kins"]["exists"] and inventory["kins"]["by_extension"] == {".json": 1}
    assert not inventory["kitti_object"]["exists"]
    text = (tmp_path / "processed/RAW_INVENTORY.md").read_text()
    assert "**missing**" in text and "## kins" in text
    assert json.loads((tmp_path / "processed/raw_inventory.json").read_text()) == inventory


# ---------------------------------------------------------------- scene datasets

def test_a_dataset_that_is_not_there_is_reported_not_raised(tmp_path):
    result = report.write_report("sunrgbd", tmp_path)
    assert not result["exists"] and "does not exist" in result["problems"][0]
    assert not (tmp_path / "sunrgbd").exists()                         # the report did not create the folder


def test_scene_report_counts_what_was_written(kitti):
    result = report.build_report("kitti_object", kitti, check_count=False)
    summary = result["summary"]
    assert result["problems"] == []
    assert summary["samples"] == 3 and summary["splits"] == {"train": 2, "val": 1, "test": 0}
    assert summary["files_per_name"] == {"depth.png": 3, "image.png": 3, "instances.png": 3, "meta.json": 3, "objects.json": 3}
    assert result["file_types"]["by_extension"] == {".json": 6, ".png": 9, ".txt": 3}
    assert summary["image_sizes"] == {"192 x 128": 3}
    assert summary["K"]["fx"] == {"min": 200.0, "mean": 200.0, "max": 200.0}
    assert summary["R_gravity"]["identity_in"] == 3
    assert summary["objects"] == {"total": 3, "with_box3d": 3, "with_visible_mask": 3, "with_amodal_mask": 0, "per_class": {"Car": 3}}
    assert summary["t_cam2"]["mean"] == pytest.approx([0.06, 0.0, 0.003])


def test_geometry_checks_pass_on_a_consistent_dataset(kitti):
    geometry = report.build_report("kitti_object", kitti, check_count=False)["geometry"]
    assert geometry["samples_read"] == 3 and geometry["boxes_projected"] == 3
    assert geometry["median_box_iou"] > 0.999
    assert geometry["tall_boxes"] == 3 and geometry["tall_boxes_without_points"] == 0
    assert geometry["median_depth_m"] == pytest.approx(4.0, abs=1 / 256)


def test_a_complete_conversion_has_a_known_size(kitti):
    problems = report.build_report("kitti_object", kitti)["problems"]
    assert problems == ["3 samples, but a complete conversion gives 7481"]


def test_a_missing_file_is_named(kitti):
    (kitti / "kitti_object/samples/000001/depth.png").unlink()
    assert "000001: depth.png is missing" in problems_of("kitti_object", kitti)


def test_a_file_that_meta_does_not_promise_is_named(kitti):
    (kitti / "kitti_object/samples/000001/layout.json").write_text("{}")
    assert "000001: layout.json exists but meta.json says has_layout = false" in problems_of("kitti_object", kitti)


def test_split_mistakes_are_named(kitti):
    root = kitti / "kitti_object"
    write_splits(root, {"train": ["000000", "000001", "000009"], "val": ["000001"]})
    problems = problems_of("kitti_object", kitti)
    assert "1 ids of train.txt have no sample folder, for example 000009" in problems
    assert "1 ids are listed in more than one split, for example 000001" in problems
    assert "1 sample folders are in no split, for example 000002" in problems


def edit_meta(processed, dataset, sample_id, **changes):
    path = processed / dataset / "samples" / sample_id / "meta.json"
    meta = json.loads(path.read_text())
    meta.update(changes)
    path.write_text(json.dumps(meta))


def test_a_matrix_that_is_not_a_rotation_is_named(kitti):
    edit_meta(kitti, "kitti_object", "000000", R_gravity=np.diag([1.0, 1.0, -1.0]).tolist())     # a reflection: determinant -1
    assert "000000: R_gravity is not a rotation matrix" in problems_of("kitti_object", kitti)


def test_a_tilted_camera_on_kitti_is_named(kitti):
    edit_meta(kitti, "kitti_object", "000000", R_gravity=gravity_from_total3d(-0.2, 0.0).tolist())
    assert "000000: R_gravity must be the identity on KITTI" in problems_of("kitti_object", kitti)


def edit_boxes(processed, change):
    """Apply change(box3d) to the box of every kitti_object sample."""
    for sample_id in ("000000", "000001", "000002"):
        path = processed / "kitti_object/samples" / sample_id / "objects.json"
        objects = json.loads(path.read_text())
        change(objects[0]["box3d"])
        path.write_text(json.dumps(objects))


def test_swapped_length_and_width_fail_the_reprojection_check(kitti):
    def swap(box):
        length, width, height = box["size_lwh"]
        box["size_lwh"] = [width, length, height]
    edit_boxes(kitti, swap)
    assert any(problem.startswith("box reprojection") for problem in problems_of("kitti_object", kitti))


def test_a_negated_heading_fails_the_reprojection_check(kitti):
    def negate(box):
        box["heading"] = -box["heading"]
    edit_boxes(kitti, negate)
    assert any(problem.startswith("box reprojection") for problem in problems_of("kitti_object", kitti))


def test_boxes_left_in_total3d_axes_fail_the_points_in_boxes_check(kitti):
    """A center left as (forward, up, right) lies beside the camera: it cannot be projected, and it holds no depth point."""
    def leave_in_total3d_axes(box):
        box["center"] = (T_TOTAL3D @ np.array(box["center"])).tolist()
    edit_boxes(kitti, leave_in_total3d_axes)
    result = report.build_report("kitti_object", kitti, check_count=False)
    assert result["geometry"]["boxes_projected"] == 0 and result["geometry"]["boxes_not_in_front"] == 3
    assert any(problem.startswith("points in boxes") for problem in result["problems"])


def test_camera_tilt_is_reported_in_degrees_with_total3d_signs(tmp_path):
    root = tmp_path / "sunrgbd"
    scene_sample(root, "00001", "train", "sunrgbd", R_gravity=gravity_from_total3d(np.radians(-25.0), np.radians(4.0)))
    write_splits(root, {"train": ["00001"]})
    result = report.build_report("sunrgbd", tmp_path, check_count=False)
    assert result["problems"] == []
    tilt = result["summary"]["R_gravity"]
    assert tilt["identity_in"] == 0
    assert tilt["pitch_deg"]["mean"] == pytest.approx(-25.0) and tilt["roll_deg"]["mean"] == pytest.approx(4.0)


# ---------------------------------------------------------------- pix3d

def test_pix3d_report_counts_classes_and_point_ranges(pix3d):
    result = report.build_report("pix3d", pix3d, check_count=False)
    assert result["problems"] == []
    summary = result["summary"]
    assert summary["samples"] == 2 and summary["per_class"] == {"chair": 1, "sofa": 1}
    assert summary["occlusion"] == {"none": 2} and summary["truncated"] == 0
    assert summary["points_min_xyz"] == [-1.0, -1.0, -1.0] and summary["points_max_xyz"] == [1.0, 1.0, 1.0]


def test_pix3d_points_that_are_not_normalized_are_named(pix3d):
    path = pix3d / "pix3d/samples/7/points.npy"
    np.save(path, np.load(path) * 2.0)
    assert "7: points are not centered with longest half-extent 1" in problems_of("pix3d", pix3d)


def test_pix3d_class_id_must_name_the_class(pix3d):
    edit_meta(pix3d, "pix3d", "9", class_id=1)
    assert "9: class_id 1 does not name class sofa" in problems_of("pix3d", pix3d)


# ---------------------------------------------------------------- the written files

@pytest.mark.parametrize("name", sorted(report.DATASETS))
def test_every_matrix_in_a_report_is_the_registry_matrix(name):
    """A report never restates a matrix by hand: each one it prints is looked up in configs/frames.json."""
    registry = report.frames.REGISTRY["axis_changes"]
    for step in report.transforms_for(name):
        assert {"step", "formula", "source"} <= step.keys()
        for key in ("matrix", "ideal_rotation"):
            if key in step:
                assert step[key]["matrix"] == registry[step[key]["name"]]["matrix"]


def test_write_report_writes_both_files_into_the_dataset_folder(kitti):
    result = report.write_report("kitti_object", kitti, check_count=False)
    root = kitti / "kitti_object"
    assert json.loads((root / "report.json").read_text())["summary"] == result["summary"]
    text = (root / "REPORT.md").read_text()
    for expected in ("# Dataset report: kitti_object", "no problems found", "## Layout", "## Files by type", "## Metadata",
                     "## Objects", "## Geometry checks", "## Transformations that produced this dataset",
                     "p_C = R0_rect (Tr_velo_to_cam [p_V; 1]) + t_cam2", "## Problems"):
        assert expected in text, expected


def test_a_second_run_reports_the_same_file_counts(kitti):
    first = report.write_report("kitti_object", kitti, check_count=False)["file_types"]
    second = report.write_report("kitti_object", kitti, check_count=False)["file_types"]
    assert first == second


def test_sunrgbd_report_prints_the_axis_change_and_the_bins(tmp_path):
    root = tmp_path / "sunrgbd"
    scene_sample(root, "00001", "train", "sunrgbd", R_gravity=gravity_from_total3d(-0.3, 0.05))
    write_splits(root, {"train": ["00001"]})
    report.write_report("sunrgbd", tmp_path, check_count=False)
    text = (root / "REPORT.md").read_text()
    assert "`T` maps total3d_camera to opencv_camera" in text
    assert "[  0,   0,   1]" in text and "[  0,  -1,   0]" in text
    assert "Pitch bins (degrees): [[-60.0, -20.0], [-20.0, 20.0]]" in text
    assert "Total3D corner i is CAGE corner [6, 5, 4, 7, 2, 1, 0, 3][i]." in text
