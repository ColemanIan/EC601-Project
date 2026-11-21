"""Occlusion data (plan task D6). Run once on the SCC:
python -m cage.data.occlusion kitti    # processed/kitti_object_occ: copy-paste samples for the outdoor track
python -m cage.data.occlusion indoor   # processed/occluders_indoor: cut-outs for OccludeCrop on the indoor track
"""
import argparse
import copy
import json

import numpy as np
from PIL import Image, ImageFilter
from pycocotools import mask as mask_utils

from cage.config import PROCESSED
from cage.data.sample_io import read_sample, read_split, write_sample, write_splits
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points
from cage.geometry.frames import KITTI_CLASSES, wrap_angle

PASTE_CLASSES = tuple(KITTI_CLASSES)


def encode_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def extent(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def radius(box):
    """Half the diagonal of a box's footprint on the ground, meters."""
    return 0.5 * float(np.hypot(box["size_lwh"][0], box["size_lwh"][1]))


def amodal_box(center, size_lwh, heading, K, W, H):
    uv = project_points(box3d_corners(np.array(center), size_lwh, heading), K)
    return [float(max(uv[:, 0].min(), 0.0)), float(max(uv[:, 1].min(), 0.0)),
            float(min(uv[:, 0].max(), W - 1.0)), float(min(uv[:, 1].max(), H - 1.0))]


def soft_alpha(mask):
    """Soft compositing weight: the mask blurred by about 5 x 5 pixels, so no sharp paste edge is learned."""
    im = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(radius=1))
    return np.asarray(im, dtype=np.float32)[..., None] / 255.0


def kitti_sources(root, ids):
    """Fully visible, labeled objects from frames allowed to supply depth: list of (sample_id, object id)."""
    out = []
    for i in ids:
        d = root / "samples" / i
        meta = json.load(open(d / "meta.json"))
        if not (meta["has_masks"] and meta["has_depth"]) or meta.get("depth_bench_split") == "val":
            continue
        for o in json.load(open(d / "objects.json")):
            if (o["class"] in PASTE_CLASSES and o["box3d"] and o["has_mask"] and o["occluded"] == 0 and o["truncated"] < 0.15
                    and o["bbox2d_visible"][3] - o["bbox2d_visible"][1] >= 40):
                out.append((i, o["id"]))
    return out


def paste_object(t, s, so, rng, min_occ=0.2, max_occ=0.7, min_z=4.0):
    """Paste source object so from sample s in front of one object of sample t. Returns a new sample dict, or None
    when this placement breaks a geometry rule. Every array and label of the result describes the same pasted scene."""
    Kt, Ks = np.array(t["meta"]["K"]), np.array(s["meta"]["K"])
    H, W = t["image"].shape[:2]
    inst_t = t["instances"]
    targets = [o for o in t["objects"] if o["class"] in PASTE_CLASSES and o["box3d"] and o["has_mask"]]
    if not targets:
        return None
    to = targets[int(rng.integers(len(targets)))]
    tb, sb = to["box3d"], so["box3d"]

    # 1. Depth of the pasted object: in front of the target, with no 3D overlap
    far = tb["center"][2] - radius(tb) - radius(sb) - 0.5
    if far < min_z:
        return None
    z_new = float(rng.uniform(max(min_z, far - 8.0), far))
    z_src = sb["center"][2]
    f = float(Kt[0, 0] / Ks[0, 0] * z_src / z_new)                       # pinhole scale from source to target image
    if not 0.5 <= f <= 2.0:
        return None

    # 2. Scale the source patch
    src_mask = s["instances"] == so["id"]
    ys, xs = np.nonzero(src_mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    nw, nh = int(round((x1 - x0) * f)), int(round((y1 - y0) * f))
    if nw < 8 or nh < 8:
        return None
    size = (nw, nh)
    rgb = np.asarray(Image.fromarray(s["image"][y0:y1, x0:x1]).resize(size, Image.BILINEAR))
    mask = np.asarray(Image.fromarray(src_mask[y0:y1, x0:x1].astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127
    on_object = src_mask & (np.abs(s["depth"] - z_src) <= radius(sb) + 0.5)   # LiDAR points on the object itself
    dep = np.asarray(Image.fromarray((s["depth"] * on_object)[y0:y1, x0:x1]).resize(size, Image.NEAREST))
    if ((dep > 0) & mask).sum() < 20:
        return None                                                      # too little LiDAR to supervise the pasted surface

    # 3. Place it: same ground height as the target, in a column that crosses the target's visible pixels
    target_mask = inst_t == to["id"]
    a, _, b, _ = extent(target_mask)
    u_new = float(rng.uniform(a, b))
    y_new = tb["center"][1] + tb["size_lwh"][2] / 2 - sb["size_lwh"][2] / 2      # bottom faces at the same height
    x_new = (u_new - Kt[0, 2]) * z_new / Kt[0, 0]
    v_new = Kt[1, 1] * y_new / z_new + Kt[1, 2]
    us, vs = project_points(np.array(sb["center"]), Ks)
    left, y_top = int(round(u_new + f * (x0 - us))), int(round(v_new + f * (y0 - vs)))
    if left < 0 or y_top < 0 or left + nw > W or y_top + nh > H:
        return None                                                      # keep the whole pasted object in the image

    # 4. Reject placements that collide in 3D or sit behind measured geometry
    r_s = radius(sb)
    for o in t["objects"]:
        if o["box3d"]:
            c = o["box3d"]["center"]
            if np.hypot(c[0] - x_new, c[2] - z_new) < radius(o["box3d"]) + r_s + 0.2:
                return None
    mask_full = np.zeros((H, W), bool)
    mask_full[y_top:y_top + nh, left:left + nw] = mask
    measured = t["depth"][mask_full]
    if ((measured > 0) & (measured < z_new - r_s - 1.0)).mean() > 0.05:
        return None

    # 5. Compose far to near: objects nearer than the paste, and objects of unknown depth, stay in front of it
    front = np.zeros((H, W), bool)
    for o in t["objects"]:
        if o["has_mask"] and (o["box3d"] is None or o["box3d"]["center"][2] < z_new):
            front |= inst_t == o["id"]
    visible = mask_full & ~front
    if visible.sum() < 64:
        return None

    # 6. Occlusion amounts: the target loses min_occ to max_occ of its visible pixels, and no other object loses more
    objects = copy.deepcopy(t["objects"])
    lost = {}
    for o in objects:
        if o["has_mask"]:
            m = inst_t == o["id"]
            lost[o["id"]] = float((m & visible).sum() / max(m.sum(), 1))
    if not min_occ <= lost[to["id"]] <= max_occ or max(lost.values()) > max_occ:
        return None

    # 7. Image, depth and instances
    canvas = t["image"].copy()
    canvas[y_top:y_top + nh, left:left + nw][mask] = rgb[mask]
    canvas[~visible] = t["image"][~visible]
    alpha = soft_alpha(visible)
    image = (alpha * canvas + (1 - alpha) * t["image"]).round().astype(np.uint8)
    dep_full = np.zeros((H, W), np.float32)
    dep_full[y_top:y_top + nh, left:left + nw] = dep
    depth = t["depth"].copy()
    depth[visible] = 0.0                                                 # no measurement unless the source has one
    measured = visible & (dep_full > 0)
    depth[measured] = np.maximum(dep_full[measured] - z_src + z_new, 0.0)  # the surface moves with its object
    new_id = max(o["id"] for o in objects) + 1
    inst = inst_t.copy()
    inst[visible] = new_id

    # 8. Labels: the 3D box and amodal mask of every existing object stay; only what is visible changes
    for o in objects:
        if o["has_mask"] and lost[o["id"]] > 0:
            m = inst == o["id"]
            o["has_mask"] = bool(m.any())
            if o["has_mask"]:
                o["bbox2d_visible"] = extent(m)
            o["occluded"] = max(o["occluded"] or 0, 2 if lost[o["id"]] >= 0.5 else 1)
    heading = sb["heading"] - np.arctan2(sb["center"][0], z_src) + np.arctan2(x_new, z_new)   # same view of the object
    heading = wrap_angle(heading)
    center = [x_new, y_new, z_new]
    objects.append({"id": new_id, "class": so["class"], "bbox2d_visible": extent(visible),
                    "bbox2d_amodal": amodal_box(center, sb["size_lwh"], heading, Kt, W, H),
                    "box3d": {"center": center, "size_lwh": sb["size_lwh"], "heading": heading},
                    "truncated": 0.0, "occluded": 0 if visible.sum() == mask_full.sum() else 1,
                    "has_mask": True, "mask_amodal": encode_rle(mask_full)})
    return {"image": image, "depth": depth, "instances": inst, "objects": objects,
            "target_lost": lost[to["id"]], "z_new": z_new}


def make_kitti(limit=None):
    src, dst = PROCESSED / "kitti_object", PROCESSED / "kitti_object_occ"
    ids = read_split(src, "train")
    bank = kitti_sources(src, ids)
    print(f"{len(bank)} source objects")
    out, lost = [], []
    for tid in ids[:limit]:
        t = read_sample(src, tid)
        if not (t["meta"]["has_masks"] and t["meta"]["has_depth"]):
            continue
        rng = np.random.default_rng(int(tid))
        for _ in range(30):
            sid, oid = bank[int(rng.integers(len(bank)))]
            if sid == tid:
                continue
            s = read_sample(src, sid)
            r = paste_object(t, s, next(o for o in s["objects"] if o["id"] == oid), rng)
            if r:
                new_id = f"{tid}_occ"
                meta = dict(t["meta"], dataset="kitti_object_occ", sample_id=new_id, split="train",
                            raw_ref=f"paste target={tid} source={sid} object={oid} z={r['z_new']:.2f}")
                write_sample(dst, new_id, r["image"], meta, r["objects"], depth=r["depth"], instances=r["instances"])
                out.append(new_id)
                lost.append(r["target_lost"])
                break
    write_splits(dst, {"train": out})
    print(f"kitti_object_occ: {len(out)} samples from {len(ids[:limit])} frames, mean target occlusion {np.mean(lost):.2f}")


def make_indoor(max_per_image=5, min_side=64):
    src, dst = PROCESSED / "sunrgbd", PROCESSED / "occluders_indoor"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in read_split(src, "train"):
        s = read_sample(src, i)
        if not s["meta"]["has_masks"]:
            continue
        kept = 0
        for o in s["objects"]:
            m = s["instances"] == o["id"]
            if not o["has_mask"] or not m.any() or kept == max_per_image:
                continue
            x1, y1, x2, y2 = [int(v) for v in extent(m)]
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            rgba = np.dstack([s["image"], m.astype(np.uint8) * 255])[y1:y2 + 1, x1:x2 + 1]
            Image.fromarray(rgba).save(dst / f"{n:06d}.png")
            n, kept = n + 1, kept + 1
    print(f"occluders_indoor: {n} cut-outs")


def occlude_array(crop, rgba, rng, cover=(0.1, 0.5)):
    """Paste one RGBA cut-out over a crop. crop uint8 [H, W, 3]; the cut-out covers the given share of the crop's area."""
    H, W = crop.shape[:2]
    mask = rgba[..., 3] > 0
    scale = np.sqrt(rng.uniform(*cover) * H * W / max(mask.sum(), 1))
    h, w = rgba.shape[:2]
    scale = min(scale, W / w, H / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    patch = np.asarray(Image.fromarray(rgba).resize((nw, nh), Image.BILINEAR))
    x, y = int(rng.integers(0, W - nw + 1)), int(rng.integers(0, H - nh + 1))
    full = np.zeros((H, W), bool)
    full[y:y + nh, x:x + nw] = patch[..., 3] > 127
    canvas = crop.copy()
    canvas[y:y + nh, x:x + nw][patch[..., 3] > 127] = patch[..., :3][patch[..., 3] > 127]
    alpha = soft_alpha(full)
    return (alpha * canvas + (1 - alpha) * crop).round().astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=["kitti", "indoor"])
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    make_kitti(a.limit) if a.part == "kitti" else make_indoor()
