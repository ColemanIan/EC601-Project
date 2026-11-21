# cage/data/convert_sunrgbd.py
import pickle, random
import numpy as np
from cage.config import RAW, PROCESSED
from cage.data.sample_io import write_sample, write_splits

T = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64)    # T == T.T == inv(T)
PITCH_BINS = np.deg2rad([[-60, -20], [-20, 20]])                       # total3d/configs/data_config.py
ROLL_BINS = np.deg2rad([[-20, 0], [0, 20]])
NYU40CLASSES = ['void',
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'blinds', 'desk', 'shelves',
                'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
                'person', 'night_stand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

def from_bins(bins, cls, reg):                                          # total3d/net_utils/libs.py::num_from_bins
    width = bins[0][1] - bins[0][0]
    return bins[cls].mean() + reg * width

def R_from_pitch_roll(pitch, roll):                                     # libs.py::R_from_yaw_pitch_roll with yaw = 0
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cp, -cr * sp, sp * sr],
                     [sp, cp * cr, -cp * sr],
                     [0.0, sr, cr]])

def box_from_corners(c):
    """8x3 corners in the upright frame, Total3D's order -> center, size_lwh, heading."""
    front, side, up = (c[1] - c[0]) / 2, (c[2] - c[1]) / 2, (c[0] - c[4]) / 2
    f = front / np.linalg.norm(front)
    return {"center": c.mean(axis=0).tolist(),
            "size_lwh": [2 * float(np.linalg.norm(front)), 2 * float(np.linalg.norm(side)), 2 * float(np.linalg.norm(up))],
            "heading": float(np.arctan2(-f[2], f[0]))}

def paste(full, crop, x1, y1):
    """Paste a bool crop into a full-size bool mask at (x1, y1), clipping at the image border."""
    h = min(crop.shape[0], full.shape[0] - y1)
    w = min(crop.shape[1], full.shape[1] - x1)
    full[y1:y1 + h, x1:x1 + w] |= crop[:h, :w].astype(bool)

def convert_one(data, sid, split, dst):
    image, depth = np.asarray(data["rgb_img"]), np.asarray(data["depth_map"], dtype=np.float32)
    H, W = image.shape[:2]
    cam = data["camera"]
    K = np.asarray(cam["K"], dtype=np.float64)
    cam_R = R_from_pitch_roll(from_bins(PITCH_BINS, cam["pitch_cls"], cam["pitch_reg"]),
                              from_bins(ROLL_BINS, cam["roll_cls"], cam["roll_reg"]))
    R_gravity = T @ cam_R @ T                                           # OpenCV camera -> upright
    boxes, objects, paint = data["boxes"], [], []
    for j in range(len(boxes["bdb3D"])):
        box3d = box_from_corners(np.asarray(boxes["bdb3D"][j]) @ T.T)   # world -> upright
        amodal = (np.asarray(boxes["bdb2D_from_3D"][j]) * [K[0, 2], K[1, 2], K[0, 2], K[1, 2]]).tolist()
        oid = j + 1
        mask = boxes["mask"][j]
        if mask is not None:
            full = np.zeros((H, W), bool)
            x1, y1 = int(mask["msk_bdb"][0]), int(mask["msk_bdb"][1])
            paste(full, np.asarray(mask["msk"]), x1, y1)
            z_cam = (R_gravity.T @ np.array(box3d["center"]))[2]       # center depth in the camera frame
            paint.append((z_cam, oid, full))
        objects.append({"id": oid, "class": NYU40CLASSES[boxes["size_cls"][j]],
                        "bbox2d_visible": [float(v) for v in boxes["bdb2D_pos"][j]], "bbox2d_amodal": amodal,
                        "box3d": box3d, "truncated": None, "occluded": None, "has_mask": False, "mask_amodal": None})
    inst = np.zeros((H, W), np.uint16)
    for z, oid, full in sorted(paint, key=lambda p: -p[0]):             # far first; the nearer object wins overlaps
        inst[full] = oid
    for o in objects:
        o["has_mask"] = bool((inst == o["id"]).any())
    layout = box_from_corners(np.asarray(data["layout"]["bdb3D"]) @ T.T)
    meta = {"dataset": "sunrgbd", "sample_id": sid, "split": split, "width": W, "height": H,
            "K": K.tolist(), "R_gravity": R_gravity.tolist(), "depth_source": "sensor", "depth_bench_split": None,
            "label_type": "sensor", "has_depth": True, "has_masks": True, "has_amodal_masks": False,
            "has_boxes3d": True, "has_layout": True, "raw_ref": f"sunrgbd/{int(data['sequence_id'])}"}
    write_sample(dst, sid, image, meta, objects, depth=depth, instances=inst, layout=layout)

def main():
    src, dst = RAW / "sunrgbd/total3d_processed", PROCESSED / "sunrgbd"
    files = sorted(p for p in src.rglob("*.pkl") if not p.stem.endswith("_flip"))
    random.seed(0)
    splits = {"train": [], "val": [], "test": []}
    for p in files:
        data = pickle.load(open(p, "rb"))
        seq = int(data["sequence_id"])
        split = "test" if seq <= 5050 else ("val" if random.random() < 0.05 else "train")
        sid = f"{seq:05d}"
        convert_one(data, sid, split, dst)
        splits[split].append(sid)
    write_splits(dst, splits)
    print({k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    main()