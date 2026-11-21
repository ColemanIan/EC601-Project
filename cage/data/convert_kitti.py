# cage/data/convert_kitti.py, part 1: the depth evaluation set
import argparse
import numpy as np
from PIL import Image
from cage.config import RAW, PROCESSED
from cage.data.io_depth import read_depth_png
from cage.data.sample_io import write_sample, write_splits

def convert_depth_val():
    src = RAW / "kitti/depth/depth_selection/val_selection_cropped"
    dst = PROCESSED / "kitti_depth_val"
    ids = []
    for img_path in sorted((src / "image").glob("*.png")):
        name = img_path.stem
        gt_path = src / "groundtruth_depth" / (name.replace("_image_", "_groundtruth_depth_", 1) + ".png")
        K = np.loadtxt(src / "intrinsics" / f"{name}.txt").reshape(3, 3)
        image = np.asarray(Image.open(img_path).convert("RGB"))
        depth = read_depth_png(gt_path)              # already in the KITTI encoding; re-written unchanged
        H, W = depth.shape
        meta = {"dataset": "kitti_depth_val", "sample_id": name, "split": "test", "width": W, "height": H,
                "K": K.tolist(), "R_gravity": np.eye(3).tolist(), "depth_source": "kitti_annotated",
                "depth_bench_split": "val", "label_type": "lidar", "has_depth": True, "has_masks": False,
                "has_amodal_masks": False, "has_boxes3d": False, "has_layout": False, "raw_ref": name}
        write_sample(dst, name, image, meta, [], depth=depth)
        ids.append(name)
    write_splits(dst, {"test": ids})
    print(f"kitti_depth_val: {len(ids)} samples")


# cage/data/convert_kitti.py, part 2: the 3D object benchmark
from pathlib import Path
from cage.geometry.camera import project_points
from cage.geometry.boxes import box3d_corners

REPO = Path(__file__).resolve().parents[2]
OBJ = RAW / "kitti/object/training"

def read_calib(path):
    """KITTI calibration file -> P2 [3, 4], R0 [3, 3], V2C [3, 4]; same logic as Calibration in frustum-pointnets/kitti/kitti_util.py."""
    vals = {}
    for line in open(path):
        if ":" in line:
            key, value = line.split(":", 1)
            vals[key.strip()] = np.array([float(x) for x in value.split()])
    return {"P2": vals["P2"].reshape(3, 4), "R0": vals["R0_rect"].reshape(3, 3), "V2C": vals["Tr_velo_to_cam"].reshape(3, 4)}

def read_labels(path):
    out = []
    for line in open(path):
        f = line.split()
        out.append({"type": f[0], "truncated": float(f[1]), "occluded": int(f[2]), "bbox": [float(x) for x in f[4:8]],
                    "hwl": [float(x) for x in f[8:11]], "xyz": [float(x) for x in f[11:14]], "ry": float(f[14])})
    return out

def kitti_objects(labels, K, t, W, H):
    objects = []
    for o in labels:
        if o["type"] == "DontCare":
            continue
        h, w, l = o["hwl"]
        center = np.array(o["xyz"]) + t - np.array([0.0, h / 2.0, 0.0])     # bottom center -> geometric center, camera 2
        corners = box3d_corners(center, [l, w, h], o["ry"])
        if (corners[:, 2] > 0.1).all():
            uv = project_points(corners, K)
            amodal = [max(uv[:, 0].min(), 0.0), max(uv[:, 1].min(), 0.0), min(uv[:, 0].max(), W - 1.0), min(uv[:, 1].max(), H - 1.0)]
        else:
            amodal = o["bbox"]                     # box partly behind the camera: keep the labeled box
        objects.append({"id": len(objects) + 1, "class": o["type"], "bbox2d_visible": o["bbox"],
                        "bbox2d_amodal": [float(x) for x in amodal],
                        "box3d": {"center": center.tolist(), "size_lwh": [l, w, h], "heading": o["ry"]},
                        "truncated": o["truncated"], "occluded": o["occluded"], "has_mask": False, "mask_amodal": None})
    return objects

def convert_object():
    dst = PROCESSED / "kitti_object"
    splits = {s: [line.strip() for line in open(REPO / f"frustum-pointnets/kitti/image_sets/{s}.txt")] for s in ("train", "val")}
    for split, ids in splits.items():
        for i in ids:
            calib = read_calib(OBJ / "calib" / f"{i}.txt")
            K = calib["P2"][:, :3]
            t = np.linalg.solve(K, calib["P2"][:, 3])                   # camera 0 rect -> camera 2, about [0.06, 0, 0] m
            image = np.asarray(Image.open(OBJ / "image_2" / f"{i}.png").convert("RGB"))
            H, W = image.shape[:2]
            objects = kitti_objects(read_labels(OBJ / "label_2" / f"{i}.txt"), K, t, W, H)
            meta = {"dataset": "kitti_object", "sample_id": i, "split": split, "width": W, "height": H,
                    "K": K.tolist(), "R_gravity": np.eye(3).tolist(), "t_cam2": t.tolist(),
                    "depth_source": None, "depth_bench_split": None, "label_type": None,
                    "has_depth": False, "has_masks": False, "has_amodal_masks": False, "has_boxes3d": True,
                    "has_layout": False, "raw_ref": f"object/training/{i}"}
            write_sample(dst, i, image, meta, objects)
    write_splits(dst, splits)
    
    
CONVERTERS = {"depth_val": convert_depth_val, "object": convert_object}     # D3.4 and D3.5 add "object" and "object_depth"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=sorted(CONVERTERS))
    CONVERTERS[p.parse_args().part]()