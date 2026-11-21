"""SUN RGB-D converter (plan task D4.3). Run on the SCC:

python -m cage.data.convert_sunrgbd

Reads Total3D's per-image pickles from raw/sunrgbd/total3d_processed and writes processed/sunrgbd.
Every frame conversion is a function of cage/geometry/frames.py; this file only moves data.
"""
import json
import pickle
from pathlib import Path

import numpy as np

from cage.config import PROCESSED, RAW
from cage.data.sample_io import write_sample, write_splits
from cage.geometry.frames import (NYU40_CLASSES, PITCH_BINS, ROLL_BINS, SUNRGBD_DEPTH_CAP_M, angle_from_bin,
                                  box_from_total3d_corners, gravity_from_total3d)

REPO = Path(__file__).resolve().parents[2]                       # the EC601 folder
SPLIT_DIR = REPO / "total3d/data/sunrgbd/splits"                 # train.json, test.json: lists of pickle paths
VAL_EVERY = 20                                                   # every 20th training image is held out for validation


def paste_mask(full_mask, crop, left, top):
    """Write a bool crop into a full-size bool mask with its top-left pixel at (left, top), clipped at the border."""
    crop_height = min(crop.shape[0], full_mask.shape[0] - top)
    crop_width = min(crop.shape[1], full_mask.shape[1] - left)
    full_mask[top:top + crop_height, left:left + crop_width] |= crop[:crop_height, :crop_width].astype(bool)


def convert_one(data, sample_id, split, out_root):
    """One unpickled Total3D sample -> one processed/sunrgbd sample folder."""
    image = np.asarray(data["rgb_img"])
    depth = np.array(data["depth_map"], dtype=np.float32)
    depth[depth >= SUNRGBD_DEPTH_CAP_M] = 0.0                           # Total3D clipped depth at 8 m; a clipped pixel is not a measurement
    height, width = image.shape[:2]
    camera = data["camera"]
    K = np.asarray(camera["K"], dtype=np.float64)
    pitch = angle_from_bin(PITCH_BINS, camera["pitch_cls"], camera["pitch_reg"])
    roll = angle_from_bin(ROLL_BINS, camera["roll_cls"], camera["roll_reg"])
    R_gravity = gravity_from_total3d(pitch, roll)                       # p_upright = R_gravity @ p_camera
    principal_point = [K[0, 2], K[1, 2], K[0, 2], K[1, 2]]              # cx, cy, cx, cy

    boxes = data["boxes"]                                               # a dict of per-object lists
    objects, layers = [], []                                            # layers: (camera depth, object id, visible mask)
    for index in range(len(boxes["bdb3D"])):
        object_id = index + 1
        box3d = box_from_total3d_corners(boxes["bdb3D"][index])         # Total3D world corners -> CAGE box in the upright frame
        # Total3D stores this box divided by the principal point; multiplying restores pixels.
        bbox2d_amodal = (np.asarray(boxes["bdb2D_from_3D"][index]) * principal_point).tolist()
        mask = boxes["mask"][index]                                     # {'msk_bdb': [x1, y1, x2, y2], 'msk': bool crop} or None
        if mask is not None:
            full_mask = np.zeros((height, width), bool)
            paste_mask(full_mask, np.asarray(mask["msk"]), int(mask["msk_bdb"][0]), int(mask["msk_bdb"][1]))
            center_depth = float((R_gravity.T @ np.array(box3d["center"]))[2])   # z of the center in the camera frame
            layers.append((center_depth, object_id, full_mask))
        objects.append({"id": object_id, "class": NYU40_CLASSES[int(boxes["size_cls"][index])],
                        "bbox2d_visible": [float(v) for v in boxes["bdb2D_pos"][index]],
                        "bbox2d_amodal": bbox2d_amodal, "box3d": box3d,
                        "truncated": None, "occluded": None, "has_mask": False, "mask_amodal": None})

    instances = np.zeros((height, width), np.uint16)
    for center_depth, object_id, full_mask in sorted(layers, key=lambda layer: -layer[0]):
        instances[full_mask] = object_id                                # painted far first, so a nearer object overwrites
    for obj in objects:
        obj["has_mask"] = bool((instances == obj["id"]).any())

    layout = box_from_total3d_corners(data["layout"]["bdb3D"])
    meta = {"dataset": "sunrgbd", "sample_id": sample_id, "split": split, "width": width, "height": height,
            "K": K.tolist(), "R_gravity": R_gravity.tolist(), "depth_source": "sensor", "depth_bench_split": None,
            "label_type": "sensor", "has_depth": True, "has_masks": True, "has_amodal_masks": False,
            "has_boxes3d": True, "has_layout": True, "raw_ref": f"sunrgbd/{int(data['sequence_id'])}"}
    write_sample(out_root, sample_id, image, meta, objects, depth=depth, instances=instances, layout=layout)


def split_sequence_ids(split_dir, name):
    """Total3D's split file -> the sorted SUN RGB-D sequence ids in it, flipped copies left out."""
    stems = [Path(path).stem for path in json.load(open(split_dir / f"{name}.json"))]
    return sorted(int(stem) for stem in stems if not stem.endswith("_flip"))


def make_splits(split_dir):
    """Total3D's test split is kept; every VAL_EVERY-th id of its train split becomes validation."""
    train_ids = split_sequence_ids(split_dir, "train")
    val_ids = train_ids[::VAL_EVERY]
    held_out = set(val_ids)
    return {"train": [i for i in train_ids if i not in held_out], "val": val_ids,
            "test": split_sequence_ids(split_dir, "test")}


def main(raw=RAW, processed=PROCESSED, split_dir=SPLIT_DIR):
    pickle_root = raw / "sunrgbd/total3d_processed"
    out_root = processed / "sunrgbd"
    pickle_path = {int(path.stem): path for path in pickle_root.rglob("*.pkl") if path.stem.isdigit()}   # flips excluded
    written = {"train": [], "val": [], "test": []}
    missing = []
    for split, sequence_ids in make_splits(split_dir).items():
        for sequence_id in sequence_ids:
            if sequence_id not in pickle_path:
                missing.append(sequence_id)
                continue
            with open(pickle_path[sequence_id], "rb") as f:
                data = pickle.load(f)
            assert int(data["sequence_id"]) == sequence_id, f"{pickle_path[sequence_id]} holds sequence {data['sequence_id']}"
            sample_id = f"{sequence_id:05d}"
            convert_one(data, sample_id, split, out_root)
            written[split].append(sample_id)
    write_splits(out_root, written)
    print({split: len(ids) for split, ids in written.items()},
          f"; {len(missing)} ids in the split files have no pickle", missing[:5])


if __name__ == "__main__":
    main()
