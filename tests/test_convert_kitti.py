"""Tests for cage/data/convert_kitti.py on a miniature KITTI tree built in a temporary folder. No dataset is needed."""
import numpy as np
import pytest
from PIL import Image

from cage.data import convert_kitti
from cage.data.io_depth import write_depth_png
from cage.data.sample_io import read_sample, read_split, write_sample
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points

WIDTH, HEIGHT = 200, 60
K = np.array([[120.0, 0.0, 100.0], [0.0, 120.0, 30.0], [0.0, 0.0, 1.0]])
T_CAM2 = np.array([0.06, 0.0, 0.003])
P2 = np.c_[K, K @ T_CAM2]                                   # P2 = K [I | t_cam2]
VELO_TO_CAM = np.array([[0.0, -1.0, 0.0, 0.0],                   # Velodyne (x forward, y left, z up)
                        [0.0, 0.0, -1.0, 0.0],                   # -> camera (x right, y down, z forward)
                        [1.0, 0.0, 0.0, 0.0]])
CAR_LABEL = "Car 0.00 0 -1.58 80.0 20.0 120.0 40.0 1.50 1.60 3.90 0.50 1.65 12.00 -1.59"
DONTCARE_LABEL = "DontCare -1 -1 -10 10.0 10.0 20.0 20.0 -1 -1 -1 -1000 -1000 -1000 -10"


def write_calib(path):
    row = lambda name, matrix: f"{name}: " + " ".join(f"{x:.12e}" for x in np.asarray(matrix).ravel())
    path.write_text("\n".join([row("P0", np.c_[K, np.zeros(3)]), row("P1", P2), row("P2", P2), row("P3", P2),
                               row("R0_rect", np.eye(3)), row("Tr_velo_to_cam", VELO_TO_CAM),
                               row("Tr_imu_to_velo", VELO_TO_CAM)]) + "\n")


def gray_image(path, value=128):
    Image.fromarray(np.full((HEIGHT, WIDTH, 3), value, np.uint8)).save(path)


@pytest.fixture
def tree(tmp_path):
    """raw/ with 2 depth-evaluation images and 3 object frames; returns (raw, processed, split_dir)."""
    raw, processed, split_dir = tmp_path / "raw", tmp_path / "processed", tmp_path / "image_sets"
    selection = raw / "kitti/depth/depth_selection/val_selection_cropped"
    for folder in ("image", "groundtruth_depth", "intrinsics"):
        (selection / folder).mkdir(parents=True)
    for frame in ("0000000005", "0000000010"):
        stem = f"2011_09_26_drive_0002_sync_image_{frame}_image_02"
        gray_image(selection / "image" / f"{stem}.png")
        write_depth_png(selection / "groundtruth_depth" / f"2011_09_26_drive_0002_sync_groundtruth_depth_{frame}_image_02.png",
                        np.full((HEIGHT, WIDTH), 7.5))
        np.savetxt(selection / "intrinsics" / f"{stem}.txt", K.reshape(1, 9))

    training = raw / "kitti/object/training"
    for folder in ("image_2", "calib", "label_2", "velodyne"):
        (training / folder).mkdir(parents=True)
    scan = np.array([[10.0, 0.0, 0.0, 0.3], [20.0, 1.0, 0.0, 0.3]], np.float32)     # x, y, z, reflectance
    for sample_id in ("000000", "000001", "000002"):
        gray_image(training / "image_2" / f"{sample_id}.png")
        write_calib(training / "calib" / f"{sample_id}.txt")
        (training / "label_2" / f"{sample_id}.txt").write_text(CAR_LABEL + "\n" + DONTCARE_LABEL + "\n")
        scan.tofile(training / "velodyne" / f"{sample_id}.bin")

    mapping = raw / "kitti/object/devkit_object/mapping"
    mapping.mkdir(parents=True)
    (mapping / "train_mapping.txt").write_text("2011_09_26 2011_09_26_drive_0001_sync 0000000007\n"      # line 1
                                               "2011_09_26 2011_09_26_drive_0002_sync 0000000005\n"      # line 2
                                               "2011_09_28 2011_09_28_drive_0003_sync 0000000000\n")     # line 3
    (mapping / "train_rand.txt").write_text("2,3,1\n")           # image 0 <- line 2, image 1 <- line 3, image 2 <- line 1

    # Depth benchmark: drive_0002 is a training drive with ground truth for frame 5;
    # drive_0001 is a validation drive without ground truth for frame 7; drive_0003 is absent.
    annotated = raw / "kitti/depth/train/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02"
    annotated.mkdir(parents=True)
    write_depth_png(annotated / "0000000005.png", np.full((HEIGHT, WIDTH), 33.0))
    (raw / "kitti/depth/val/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02").mkdir(parents=True)

    split_dir.mkdir()
    (split_dir / "train.txt").write_text("000000\n000001\n")
    (split_dir / "val.txt").write_text("000002")
    return raw, processed, split_dir


def test_read_calib_returns_the_three_matrices_under_kitti_names(tmp_path):
    write_calib(tmp_path / "calib.txt")
    calib = convert_kitti.read_calib(tmp_path / "calib.txt")
    assert np.allclose(calib["P2"], P2) and calib["P2"].shape == (3, 4)
    assert np.allclose(calib["Tr_velo_to_cam"], VELO_TO_CAM) and np.allclose(calib["R0_rect"], np.eye(3))


def test_read_labels_follows_the_devkit_columns(tmp_path):
    (tmp_path / "label.txt").write_text(CAR_LABEL + "\n")
    label = convert_kitti.read_labels(tmp_path / "label.txt")[0]
    assert label == {"type": "Car", "truncated": 0.0, "occluded": 0, "bbox": [80.0, 20.0, 120.0, 40.0],
                     "dimensions": [1.5, 1.6, 3.9], "location": [0.5, 1.65, 12.0], "rotation_y": -1.59}


def test_kitti_objects_moves_bottom_center_to_center_and_drops_dontcare(tmp_path):
    (tmp_path / "label.txt").write_text(CAR_LABEL + "\n" + DONTCARE_LABEL + "\n")
    labels = convert_kitti.read_labels(tmp_path / "label.txt")
    objects = convert_kitti.kitti_objects(labels, K, T_CAM2, WIDTH, HEIGHT)
    assert [o["class"] for o in objects] == ["Car"] and objects[0]["id"] == 1
    box = objects[0]["box3d"]
    assert box["size_lwh"] == [3.9, 1.6, 1.5]                               # length, width, height
    assert np.allclose(box["center"], [0.5 + 0.06, 1.65 - 0.75, 12.0 + 0.003])
    corners = box3d_corners(box["center"], box["size_lwh"], box["heading"])
    # Corners 0-3 are the bottom face; their mean must be KITTI's labeled location, moved into camera 2.
    assert np.allclose(corners[:4].mean(axis=0), np.array([0.5, 1.65, 12.0]) + T_CAM2)
    x1, y1, x2, y2 = objects[0]["bbox2d_amodal"]
    u, v = project_points(np.array(box["center"]), K)
    assert 0 <= x1 < u < x2 <= WIDTH - 1 and 0 <= y1 < v < y2 <= HEIGHT - 1
    assert objects[0]["bbox2d_visible"] == [80.0, 20.0, 120.0, 40.0]


def test_load_raw_frame_lookup_reads_one_based_line_numbers(tree):
    raw, _, _ = tree
    lookup = convert_kitti.load_raw_frame_lookup(raw / "kitti/object/devkit_object/mapping")
    assert lookup[0] == ("2011_09_26", "2011_09_26_drive_0002_sync", "0000000005")
    assert lookup[1] == ("2011_09_28", "2011_09_28_drive_0003_sync", "0000000000")
    assert lookup[2] == ("2011_09_26", "2011_09_26_drive_0001_sync", "0000000007")


def test_velodyne_depth_projects_into_camera_2_and_keeps_the_nearest_point():
    calib = {"Tr_velo_to_cam": VELO_TO_CAM, "R0_rect": np.eye(3)}
    near_cam2 = np.array([0.0, 0.0, 10.0]) + T_CAM2                   # Velodyne (10, 0, 0) seen from camera 2
    far_cam0 = 2.0 * near_cam2 - T_CAM2                               # twice as far along the same camera-2 ray
    far_velo = [far_cam0[2], -far_cam0[0], -far_cam0[1]]                   # camera (x, y, z) -> Velodyne (z, -x, -y)
    points = np.array([[10.0, 0.0, 0.0], far_velo, [-5.0, 0.0, 0.0]])      # the last point is behind the camera
    depth = convert_kitti.velodyne_depth(points, calib, T_CAM2, K, HEIGHT, WIDTH)
    u, v = np.round(project_points(near_cam2, K)).astype(int)
    assert (depth > 0).sum() == 1
    assert depth[v, u] == pytest.approx(near_cam2[2])


def test_convert_depth_val(tree):
    raw, processed, _ = tree
    convert_kitti.convert_depth_val(raw, processed)
    root = processed / "kitti_depth_val"
    ids = read_split(root, "test")
    assert len(ids) == 2 and read_split(root, "train") == [] and read_split(root, "val") == []
    sample = read_sample(root, ids[0])
    assert sample["meta"]["split"] == "test" and sample["meta"]["depth_bench_split"] == "val"
    assert np.allclose(sample["meta"]["K"], K)
    assert sample["depth"].shape == (HEIGHT, WIDTH) and np.allclose(sample["depth"], 7.5)
    assert sample["objects"] == [] and sample["instances"] is None


def test_convert_object_then_object_depth(tree):
    raw, processed, split_dir = tree
    convert_kitti.convert_object(raw, processed, split_dir)
    root = processed / "kitti_object"
    assert read_split(root, "train") == ["000000", "000001"] and read_split(root, "val") == ["000002"]
    before = read_sample(root, "000000")
    assert before["meta"]["has_boxes3d"] and not before["meta"]["has_depth"] and before["depth"] is None
    assert np.allclose(before["meta"]["t_cam2"], T_CAM2) and len(before["objects"]) == 1

    convert_kitti.convert_object_depth(raw, processed)
    annotated, outside, val_drive = (read_sample(root, i) for i in ("000000", "000001", "000002"))
    assert annotated["meta"]["depth_source"] == "kitti_annotated" and np.allclose(annotated["depth"], 33.0)
    assert annotated["meta"]["depth_bench_split"] == "train"
    assert annotated["meta"]["raw_ref"] == "2011_09_26/2011_09_26_drive_0002_sync/0000000005"
    assert outside["meta"]["depth_source"] == "velodyne_projected" and outside["meta"]["depth_bench_split"] is None
    assert val_drive["meta"]["depth_source"] == "velodyne_projected" and val_drive["meta"]["depth_bench_split"] == "val"
    assert (outside["depth"] > 0).sum() == 2                                # the two LiDAR points of the test scan
    assert annotated["objects"] == before["objects"]                        # labels are untouched


def test_object_depth_keeps_masks_written_earlier(tree):
    raw, processed, split_dir = tree
    convert_kitti.convert_object(raw, processed, split_dir)
    root = processed / "kitti_object"
    sample = read_sample(root, "000000")
    instances = np.zeros((HEIGHT, WIDTH), np.uint16)
    instances[20:40, 80:120] = 1
    meta = dict(sample["meta"], has_masks=True)
    write_sample(root, "000000", sample["image"], meta, sample["objects"], instances=instances)   # as convert_kins does
    convert_kitti.convert_object_depth(raw, processed)
    after = read_sample(root, "000000")
    assert after["meta"]["has_masks"] and after["meta"]["has_depth"]
    assert (after["instances"] == instances).all()


def test_object_depth_rejects_a_depth_map_of_the_wrong_size(tree):
    raw, processed, split_dir = tree
    convert_kitti.convert_object(raw, processed, split_dir)
    wrong = raw / "kitti/depth/train/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000005.png"
    write_depth_png(wrong, np.full((HEIGHT + 2, WIDTH), 33.0))
    with pytest.raises(AssertionError, match="frame mapping"):
        convert_kitti.convert_object_depth(raw, processed)
