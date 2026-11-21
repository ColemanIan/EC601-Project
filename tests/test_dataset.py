"""Tests for cage/data/dataset.py: SceneDataset, RandomFlip, collate_scenes, ObjectShapeDataset. No dataset is needed.

RandomFlip is the one place where a frame convention is applied to tensors instead of numpy arrays, so its
output is compared, field by field, with the numpy mirror functions of cage/geometry/frames.py.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cage.data.dataset import ObjectShapeDataset, RandomFlip, SceneDataset, collate_scenes
from cage.data.sample_io import write_sample, write_splits
from cage.geometry import frames
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points

WIDTH, HEIGHT = 96, 64
K = np.array([[80.0, 0.0, 50.0], [0.0, 80.0, 30.0], [0.0, 0.0, 1.0]])       # cx is off-center on purpose
R_GRAVITY = frames.gravity_from_total3d(-0.3, 0.1)


def write_scene(root, sample_id, width=WIDTH, height=HEIGHT):
    box = {"center": [0.4, 0.2, 4.0], "size_lwh": [1.0, 0.8, 0.6], "heading": 0.5}
    image = np.zeros((height, width, 3), np.uint8)
    image[:, : width // 2] = 200                                             # left half bright, so a flip is visible
    depth = np.tile(np.linspace(1.0, 5.0, width, dtype=np.float32), (height, 1))
    instances = np.zeros((height, width), np.uint16)
    instances[10:20, 5:25] = 1
    meta = {"dataset": "sunrgbd", "sample_id": sample_id, "split": "train", "width": width, "height": height,
            "K": K.tolist(), "R_gravity": R_GRAVITY.tolist(), "depth_source": "synthetic", "label_type": "synthetic",
            "has_depth": True, "has_masks": True, "has_amodal_masks": False, "has_boxes3d": True, "has_layout": False,
            "raw_ref": sample_id}
    objects = [{"id": 1, "class": "chair", "bbox2d_visible": [5.0, 10.0, 24.0, 19.0], "bbox2d_amodal": [4.0, 8.0, 30.0, 22.0],
                "box3d": box, "truncated": None, "occluded": None, "has_mask": True, "mask_amodal": None}]
    write_sample(root, sample_id, image, meta, objects, depth=depth, instances=instances)


@pytest.fixture
def scene_root(tmp_path):
    write_scene(tmp_path, "a")
    write_scene(tmp_path, "b", width=80, height=48)
    write_splits(tmp_path, {"train": ["a", "b"]})
    return tmp_path


def test_scene_dataset_returns_the_documented_shapes(scene_root):
    sample = SceneDataset("sunrgbd", "train", classes=frames.SUNRGBD_CLASSES, root=scene_root)[0]
    assert sample["image"].shape == (3, HEIGHT, WIDTH) and sample["image"].max() <= 1.0
    assert sample["depth"].shape == (HEIGHT, WIDTH) and sample["instances"].shape == (HEIGHT, WIDTH)
    assert sample["boxes3d"].shape == (1, 7) and sample["boxes2d"].shape == (1, 4)
    assert sample["labels"].tolist() == [frames.SUNRGBD_CLASSES.index("chair")]
    assert sample["has_box3d"].tolist() == [True] and sample["layout"] is None


def test_class_filter_drops_other_classes(scene_root):
    sample = SceneDataset("sunrgbd", "train", classes=["bed"], root=scene_root)[0]
    assert sample["boxes3d"].shape == (0, 7) and sample["labels"].shape == (0,)


def test_random_flip_equals_the_numpy_mirror_functions(scene_root):
    plain = SceneDataset("sunrgbd", "train", root=scene_root)[0]
    flipped = SceneDataset("sunrgbd", "train", root=scene_root, transform=RandomFlip(p=1.0))[0]
    assert torch.equal(flipped["image"], plain["image"].flip(-1))
    assert torch.equal(flipped["depth"], plain["depth"].flip(-1)) and torch.equal(flipped["instances"], plain["instances"].flip(-1))
    assert np.allclose(flipped["K"].numpy(), frames.mirror_K(plain["K"].numpy(), WIDTH))
    assert np.allclose(flipped["R_gravity"].numpy(), frames.mirror_rotation(plain["R_gravity"].numpy()), atol=1e-6)
    assert np.allclose(flipped["boxes2d"].numpy()[0], frames.mirror_box2d(plain["boxes2d"].numpy()[0], WIDTH))
    center, heading = plain["boxes3d"][0, :3].numpy(), float(plain["boxes3d"][0, 6])
    assert np.allclose(flipped["boxes3d"][0, :3].numpy(), frames.mirror_points(center))
    assert np.isclose(float(flipped["boxes3d"][0, 6]), frames.mirror_heading(heading), atol=1e-6)
    assert torch.equal(flipped["boxes3d"][0, 3:6], plain["boxes3d"][0, 3:6])        # sizes do not change


def test_flipped_box_projects_onto_the_mirrored_pixels(scene_root):
    """The geometric meaning of the flip: project the flipped box with the flipped camera, and get the mirrored pixels."""
    def pixels_of(sample):
        box = sample["boxes3d"][0].numpy().astype(np.float64)
        corners_camera = box3d_corners(box[:3], box[3:6], box[6]) @ sample["R_gravity"].numpy().astype(np.float64)
        return project_points(corners_camera, sample["K"].numpy().astype(np.float64))
    plain = pixels_of(SceneDataset("sunrgbd", "train", root=scene_root)[0])
    flipped = pixels_of(SceneDataset("sunrgbd", "train", root=scene_root, transform=RandomFlip(p=1.0))[0])
    mirrored = np.sort(np.column_stack([WIDTH - 1 - plain[:, 0], plain[:, 1]]), axis=0)
    assert np.allclose(np.sort(flipped, axis=0), mirrored, atol=1e-3)


def test_flipping_twice_restores_the_sample(scene_root):
    plain = SceneDataset("sunrgbd", "train", root=scene_root)[0]
    twice = RandomFlip(p=1.0)(RandomFlip(p=1.0)(SceneDataset("sunrgbd", "train", root=scene_root)[0]))
    for key in ("image", "depth", "K", "R_gravity", "boxes2d", "boxes3d"):
        assert torch.allclose(twice[key], plain[key], atol=1e-5), key


def test_collate_pads_bottom_and_right_and_marks_real_pixels(scene_root):
    dataset = SceneDataset("sunrgbd", "train", root=scene_root)
    batch = collate_scenes([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, HEIGHT, WIDTH) and batch["depth"].shape == (2, HEIGHT, WIDTH)
    assert batch["pixel_valid"][1, :48, :80].all() and not batch["pixel_valid"][1, 48:].any() and not batch["pixel_valid"][1, :, 80:].any()
    assert (batch["depth"][1, 48:] == 0).all()                                # padding is "no measurement"
    assert torch.equal(batch["K"][1], dataset[1]["K"])                        # padding at the bottom and right leaves K valid
    assert len(batch["boxes3d"]) == 2 and batch["boxes3d"][0].shape == (1, 7)


def test_object_shape_dataset(tmp_path):
    from PIL import Image
    folder = tmp_path / "samples" / "7"
    folder.mkdir(parents=True)
    Image.fromarray(np.full((30, 20, 3), 120, np.uint8)).save(folder / "crop.png")
    np.save(folder / "points.npy", np.zeros((10000, 3), np.float32))
    (folder / "meta.json").write_text(json.dumps({"dataset": "pix3d", "sample_id": "7", "class": "chair", "class_id": 3,
                                                   "split": "train", "occlusion": "none", "truncated": False}))
    write_splits(tmp_path, {"train": ["7"]})
    sample = ObjectShapeDataset("train", root=tmp_path)[0]
    assert sample["crop"].shape == (3, 256, 256) and sample["points"].shape == (10000, 3) and int(sample["label"]) == 3
    assert len(ObjectShapeDataset("train", classes=["bed"], root=tmp_path)) == 0
