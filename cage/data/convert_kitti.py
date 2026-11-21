"""KITTI converters (plan task D3.3). Run on the SCC:

python -m cage.data.convert_kitti depth_val      # processed/kitti_depth_val: the 1,000-image depth evaluation set
python -m cage.data.convert_kitti object         # processed/kitti_object: images, calibration, 3D boxes
python -m cage.data.convert_kitti object_depth   # adds depth.png to every kitti_object sample

`object` creates the samples and must run first. `object_depth` and convert_kins.py each add to
them and can run, or be run again, in either order.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from cage.config import PROCESSED, RAW
from cage.data.io_depth import read_depth_png
from cage.data.sample_io import read_sample, read_split, write_sample, write_splits
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points
from cage.geometry.frames import kitti_label_to_box, split_kitti_projection, velodyne_to_camera2

REPO = Path(__file__).resolve().parents[2]                       # the EC601 folder
SPLIT_DIR = REPO / "frustum-pointnets/kitti/image_sets"          # train.txt, val.txt: one image id per line


# ---------------------------------------------------------------- part 1: the depth evaluation set

def convert_depth_val(raw=RAW, processed=PROCESSED):
    selection = raw / "kitti/depth/depth_selection/val_selection_cropped"
    out_root = processed / "kitti_depth_val"
    sample_ids = []
    for image_path in sorted((selection / "image").glob("*.png")):
        sample_id = image_path.stem          # 2011_09_26_drive_0002_sync_image_0000000005_image_02
        # The depth file has the same name with the FIRST "_image_" swapped; the trailing "_image_02" stays.
        depth_name = sample_id.replace("_image_", "_groundtruth_depth_", 1) + ".png"
        K = np.loadtxt(selection / "intrinsics" / f"{sample_id}.txt").reshape(3, 3)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        depth = read_depth_png(selection / "groundtruth_depth" / depth_name)
        height, width = depth.shape
        meta = {"dataset": "kitti_depth_val", "sample_id": sample_id, "split": "test",
                "width": width, "height": height, "K": K.tolist(), "R_gravity": np.eye(3).tolist(),
                "depth_source": "kitti_annotated", "depth_bench_split": "val", "label_type": "lidar",
                "has_depth": True, "has_masks": False, "has_amodal_masks": False, "has_boxes3d": False,
                "has_layout": False, "raw_ref": sample_id}
        write_sample(out_root, sample_id, image, meta, [], depth=depth)
        sample_ids.append(sample_id)
    write_splits(out_root, {"test": sample_ids})
    print(f"kitti_depth_val: {len(sample_ids)} samples")


# ---------------------------------------------------------------- part 2: the 3D object benchmark

def read_calib(path):
    """One KITTI calibration file -> the three matrices CAGE uses, under KITTI's own names."""
    rows = {}
    for line in open(path):
        if ":" in line:
            name, numbers = line.split(":", 1)
            rows[name.strip()] = np.array([float(x) for x in numbers.split()])
    return {"P2": rows["P2"].reshape(3, 4),                           # projection matrix of camera 2 (left color)
            "R0_rect": rows["R0_rect"].reshape(3, 3),                 # rectifying rotation of camera 0
            "Tr_velo_to_cam": rows["Tr_velo_to_cam"].reshape(3, 4)}   # Velodyne frame -> camera 0


def read_labels(path):
    """One KITTI label file -> a list of dicts, one per line, keyed by the devkit readme's field names."""
    labels = []
    for line in open(path):
        fields = line.split()                                         # 15 columns; column 3 (alpha) is not used
        labels.append({"type": fields[0],
                       "truncated": float(fields[1]),
                       "occluded": int(float(fields[2])),
                       "bbox": [float(x) for x in fields[4:8]],           # left, top, right, bottom (pixels)
                       "dimensions": [float(x) for x in fields[8:11]],    # height, width, length (meters)
                       "location": [float(x) for x in fields[11:14]],     # bottom center, rectified camera 0
                       "rotation_y": float(fields[14])})
    return labels


def kitti_objects(labels, K, t_cam2, width, height):
    """KITTI label rows -> objects.json entries (D2.1), with boxes in the camera 2 frame."""
    objects = []
    for label in labels:
        if label["type"] == "DontCare":                               # an unlabeled region, not an object
            continue
        # Bottom center in camera 0, sizes as (h, w, l)  ->  geometric center in camera 2, sizes as (l, w, h).
        center, size_lwh, heading = kitti_label_to_box(label["dimensions"], label["location"], label["rotation_y"], t_cam2)
        corners = box3d_corners(center, size_lwh, heading)
        if (corners[:, 2] > 0.1).all():                               # every corner is in front of the camera
            pixels = project_points(corners, K)
            bbox2d_amodal = [max(pixels[:, 0].min(), 0.0), max(pixels[:, 1].min(), 0.0),
                             min(pixels[:, 0].max(), width - 1.0), min(pixels[:, 1].max(), height - 1.0)]
        else:                                                         # box partly behind the camera
            bbox2d_amodal = label["bbox"]
        objects.append({"id": len(objects) + 1, "class": label["type"],
                        "bbox2d_visible": label["bbox"],
                        "bbox2d_amodal": [float(x) for x in bbox2d_amodal],
                        "box3d": {"center": center.tolist(), "size_lwh": size_lwh, "heading": heading},
                        "truncated": label["truncated"], "occluded": label["occluded"],
                        "has_mask": False, "mask_amodal": None})
    return objects


def convert_object(raw=RAW, processed=PROCESSED, split_dir=SPLIT_DIR):
    training = raw / "kitti/object/training"
    out_root = processed / "kitti_object"
    splits = {name: [line.strip() for line in open(split_dir / f"{name}.txt") if line.strip()]
              for name in ("train", "val")}
    for split, sample_ids in splits.items():
        for sample_id in sample_ids:                                  # "000000" ... "007480"
            calib = read_calib(training / "calib" / f"{sample_id}.txt")
            K, t_cam2 = split_kitti_projection(calib["P2"])
            image = np.asarray(Image.open(training / "image_2" / f"{sample_id}.png").convert("RGB"))
            height, width = image.shape[:2]
            labels = read_labels(training / "label_2" / f"{sample_id}.txt")
            objects = kitti_objects(labels, K, t_cam2, width, height)
            meta = {"dataset": "kitti_object", "sample_id": sample_id, "split": split,
                    "width": width, "height": height, "K": K.tolist(), "R_gravity": np.eye(3).tolist(),
                    "t_cam2": t_cam2.tolist(),                        # camera 0 -> camera 2 offset; E3 needs it to write KITTI label files
                    "depth_source": None, "depth_bench_split": None, "label_type": None,
                    "has_depth": False, "has_masks": False, "has_amodal_masks": False, "has_boxes3d": True,
                    "has_layout": False, "raw_ref": f"object/training/{sample_id}"}
            write_sample(out_root, sample_id, image, meta, objects)
    write_splits(out_root, splits)
    print(f"kitti_object: {len(splits['train'])} train, {len(splits['val'])} val")


# ---------------------------------------------------------------- part 3: depth for every object frame

def load_raw_frame_lookup(mapping_dir):
    """-> a list: entry n is (date, drive, frame), the raw recording frame of object image n.

    KITTI built the object benchmark by taking 7,481 frames from its raw drives and shuffling them.
    train_mapping.txt lists those frames in their original order, one "date drive frame" per line.
    train_rand.txt is the shuffle: one line of 7,481 comma-separated numbers, where number n
    (counting from 0) is the 1-based line of train_mapping.txt that became object image n.
    """
    mapping_line_of_image = [int(x) for x in (mapping_dir / "train_rand.txt").read_text().strip().split(",")]
    raw_frames = [tuple(line.split()) for line in open(mapping_dir / "train_mapping.txt") if line.strip()]
    return [raw_frames[line_number - 1] for line_number in mapping_line_of_image]


def depth_benchmark_split(depth_root, drive):
    """'train' or 'val' when the depth benchmark contains this drive, otherwise None."""
    for split in ("train", "val"):
        if (depth_root / split / drive).exists():
            return split
    return None


def velodyne_depth(points_velo, calib, t_cam2, K, height, width):
    """One LiDAR scan [N, 3] (Velodyne frame, meters) -> a sparse depth map [height, width] for camera 2."""
    points_cam2 = velodyne_to_camera2(points_velo, calib["Tr_velo_to_cam"], calib["R0_rect"], t_cam2)
    points_cam2 = points_cam2[points_cam2[:, 2] > 0.5]                # keep points in front of the camera
    pixels = np.round(project_points(points_cam2, K)).astype(int)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    depth = np.full((height, width), np.inf)
    np.minimum.at(depth, (pixels[inside, 1], pixels[inside, 0]), points_cam2[inside, 2])   # nearest point per pixel
    depth[np.isinf(depth)] = 0.0                                      # 0 = no measurement
    return depth


def convert_object_depth(raw=RAW, processed=PROCESSED):
    training = raw / "kitti/object/training"
    depth_root = raw / "kitti/depth"
    out_root = processed / "kitti_object"
    raw_frame_of_image = load_raw_frame_lookup(raw / "kitti/object/devkit_object/mapping")
    n_annotated = n_total = 0
    for split in ("train", "val"):
        for sample_id in read_split(out_root, split):
            sample = read_sample(out_root, sample_id)
            meta = sample["meta"]
            date, drive, frame = raw_frame_of_image[int(sample_id)]
            bench_split = depth_benchmark_split(depth_root, drive)
            annotated_path = None
            if bench_split is not None:
                annotated_path = depth_root / bench_split / drive / "proj_depth/groundtruth/image_02" / f"{frame}.png"
            if annotated_path is not None and annotated_path.exists():
                depth = read_depth_png(annotated_path)
                assert depth.shape == (meta["height"], meta["width"]), \
                    f"{sample_id}: depth {depth.shape} differs from the image; the frame mapping is wrong"
                meta["depth_source"] = "kitti_annotated"
                n_annotated += 1
            else:        # no ground truth: first or last 5 frames of a drive, or a drive outside the benchmark
                calib = read_calib(training / "calib" / f"{sample_id}.txt")
                scan = np.fromfile(training / "velodyne" / f"{sample_id}.bin", dtype=np.float32).reshape(-1, 4)
                depth = velodyne_depth(scan[:, :3].astype(np.float64), calib, np.array(meta["t_cam2"]),
                                       np.array(meta["K"]), meta["height"], meta["width"])
                meta["depth_source"] = "velodyne_projected"
            meta.update(has_depth=True, label_type="lidar", depth_bench_split=bench_split,
                        raw_ref=f"{date}/{drive}/{frame}")
            # instances is passed through, so masks already added by convert_kins.py are kept.
            write_sample(out_root, sample_id, sample["image"], meta, sample["objects"],
                         depth=depth, instances=sample["instances"])
            n_total += 1
    print(f"kitti_object: {n_annotated} of {n_total} frames have annotated depth")


CONVERTERS = {"depth_val": convert_depth_val, "object": convert_object, "object_depth": convert_object_depth}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=sorted(CONVERTERS))
    CONVERTERS[parser.parse_args().part]()
