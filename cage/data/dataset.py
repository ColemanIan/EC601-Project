"""PyTorch Dataset classes over the processed datasets (plan task D7.1).

SceneDataset reads one split of a scene dataset (kitti_depth_val, kitti_object, kitti_object_occ, sunrgbd).
ObjectShapeDataset reads one split of pix3d. RandomFlip and OccludeCrop are their training-time transforms.
Class lists and the mirror matrix come from the frame registry.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils

from cage.config import PROCESSED
from cage.data.occlusion import occlude_array
from cage.data.sample_io import read_sample, read_split
from cage.geometry.frames import KITTI_CLASSES, M_MIRROR, SUNRGBD_CLASSES   # the class lists are re-exported from here


def box7(b):
    """box3d dict (or None) -> [cx, cy, cz, l, w, h, heading]; zeros for None."""
    return [*b["center"], *b["size_lwh"], b["heading"]] if b else [0.0] * 7

class SceneDataset(torch.utils.data.Dataset):
    """One split of a scene dataset in the D2.1 format. `root` replaces processed/<name>, for tests."""
    def __init__(self, name, split, classes=None, transform=None, root=None):
        self.root = Path(root) if root is not None else PROCESSED / name
        self.ids = read_split(self.root, split)
        self.classes, self.transform = classes, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        s = read_sample(self.root, self.ids[i])
        m = s["meta"]
        H, W = m["height"], m["width"]
        objs = [o for o in s["objects"] if self.classes is None or o["class"] in self.classes]
        amodal = None
        if m["has_amodal_masks"]:
            amodal = torch.from_numpy(np.stack(
                [mask_utils.decode(o["mask_amodal"]) if o["mask_amodal"] else np.zeros((H, W), np.uint8) for o in objs]
                or [np.zeros((0, H, W), np.uint8)]).reshape(-1, H, W))
        out = {
            "image": torch.from_numpy(s["image"].copy()).permute(2, 0, 1).float() / 255.0,     # [3, H, W] in [0, 1]
            "depth": torch.from_numpy(s["depth"]) if s["depth"] is not None else torch.zeros(H, W),   # 0 = invalid
            "K": torch.tensor(m["K"], dtype=torch.float32),
            "R_gravity": torch.tensor(m["R_gravity"], dtype=torch.float32),
            "instances": torch.from_numpy(s["instances"]) if s["instances"] is not None else None,  # [H, W] int64
            "boxes2d": torch.tensor([o["bbox2d_amodal"] for o in objs], dtype=torch.float32).reshape(-1, 4),
            "boxes2d_vis": torch.tensor([o["bbox2d_visible"] for o in objs], dtype=torch.float32).reshape(-1, 4),
            "boxes3d": torch.tensor([box7(o["box3d"]) for o in objs], dtype=torch.float32).reshape(-1, 7),
            "has_box3d": torch.tensor([o["box3d"] is not None for o in objs], dtype=torch.bool),
            "labels": torch.tensor([self.classes.index(o["class"]) if self.classes else 0 for o in objs], dtype=torch.int64),
            "instance_ids": torch.tensor([o["id"] for o in objs], dtype=torch.int64),
            "masks_amodal": amodal,                                                         # [N, H, W] uint8 or None
            "layout": torch.tensor(box7(s["layout"]), dtype=torch.float32) if s["layout"] else None,
            "meta": m,
        }
        return self.transform(out) if self.transform else out

class RandomFlip:
    """Horizontal flip of a SceneDataset sample with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, s):
        if torch.rand(()) >= self.p:
            return s
        W = s["image"].shape[-1]
        M = torch.tensor(M_MIRROR, dtype=s["R_gravity"].dtype)      # mirrors x; from the frame registry
        s["image"], s["depth"] = s["image"].flip(-1), s["depth"].flip(-1)
        for k in ("instances", "masks_amodal"):
            if s[k] is not None:
                s[k] = s[k].flip(-1)
        for k in ("boxes2d", "boxes2d_vis"):
            b = s[k].clone()
            s[k][:, 0], s[k][:, 2] = W - 1 - b[:, 2], W - 1 - b[:, 0]
        s["K"] = s["K"].clone()
        s["K"][0, 2] = W - 1 - s["K"][0, 2]                         # cx' = W - 1 - cx
        s["R_gravity"] = M @ s["R_gravity"] @ M                    # mirrored camera, mirrored upright frame
        for k in ("boxes3d",) + (("layout",) if s["layout"] is not None else ()):
            b = s[k]
            b[..., 0] = -b[..., 0]                                  # center x mirrored
            b[..., 6] = torch.remainder(2 * torch.pi - b[..., 6], 2 * torch.pi) - torch.pi   # heading' = pi - heading, wrapped to [-pi, pi)
        return s

def collate_scenes(samples):
    """Pad images and depth (bottom and right, so K stays valid); keep per-object fields as lists."""
    Hm = max(s["image"].shape[1] for s in samples)
    Wm = max(s["image"].shape[2] for s in samples)
    pad = lambda t: F.pad(t, (0, Wm - t.shape[-1], 0, Hm - t.shape[-2]))
    batch = {
        "image": torch.stack([pad(s["image"]) for s in samples]),
        "depth": torch.stack([pad(s["depth"]) for s in samples]),                       # padding = 0 = invalid
        "pixel_valid": torch.stack([pad(torch.ones(s["image"].shape[1:], dtype=torch.bool)) for s in samples]),
        "K": torch.stack([s["K"] for s in samples]),
        "R_gravity": torch.stack([s["R_gravity"] for s in samples]),
    }
    for k in ("instances", "boxes2d", "boxes2d_vis", "boxes3d", "has_box3d", "labels", "instance_ids", "masks_amodal", "layout", "meta"):
        batch[k] = [s[k] for s in samples]
    return batch

class ObjectShapeDataset(torch.utils.data.Dataset):
    """One split of processed/pix3d (D5.2). `root` replaces processed/pix3d, for tests."""
    def __init__(self, split, classes=None, transform=None, size=256, root=None):
        self.root = Path(root) if root is not None else PROCESSED / "pix3d"
        ids = read_split(self.root, split)
        if classes is not None:
            ids = [i for i in ids if json.load(open(self.root / "samples" / i / "meta.json"))["class"] in classes]
        self.ids, self.transform, self.size = ids, transform, size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = self.root / "samples" / self.ids[i]
        meta = json.load(open(d / "meta.json"))
        crop = Image.open(d / "crop.png").convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        out = {"crop": torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1).float() / 255.0,   # [3, 256, 256]
               "points": torch.from_numpy(np.load(d / "points.npy")),                               # [10000, 3]
               "label": torch.tensor(meta["class_id"]),
               "meta": meta}
        return self.transform(out) if self.transform else out

class OccludeCrop:
    """Paste one indoor cut-out over a Pix3D crop; the shape target is left unchanged (D6).

    With fixed=True every sample always gets the same occluder, chosen from its id, so an evaluation repeats.
    """
    def __init__(self, p=0.5, cover=(0.1, 0.5), fixed=False, root=None):
        self.files = sorted((Path(root) if root is not None else PROCESSED / "occluders_indoor").glob("*.png"))
        self.p, self.cover, self.fixed = p, cover, fixed

    def __call__(self, s):
        rng = np.random.default_rng(int(s["meta"]["sample_id"]) if self.fixed else None)
        if not self.fixed and rng.random() >= self.p:
            return s
        rgba = np.asarray(Image.open(self.files[int(rng.integers(len(self.files)))]).convert("RGBA"))
        crop = (s["crop"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        s["crop"] = torch.from_numpy(occlude_array(crop, rgba, rng, self.cover).copy()).permute(2, 0, 1).float() / 255.0
        return s
