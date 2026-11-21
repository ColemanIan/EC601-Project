# cage/data/convert_kins.py
import json
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from cage.config import RAW, PROCESSED
from cage.data.sample_io import read_sample, read_split, write_sample

KINS_TO_KITTI = {"car": "Car", "van": "Van", "truck": "Truck", "tram": "Tram", "pedestrian": "Pedestrian", "cyclist": "Cyclist", "misc": "Misc"}
# KINS ids 1-8 checked against update_train_2020.json. "rider" (id 3) has no KITTI class: it is left out on
# purpose, so it never matches a KITTI label and is kept as a KINS-only object with class "rider" and no 3D box.

def decode(segm, h, w):
    """COCO polygons, uncompressed RLE or compressed RLE -> bool [h, w]."""
    if isinstance(segm, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segm, h, w))
    elif isinstance(segm["counts"], list):
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        rle = segm
    return mask_utils.decode(rle).astype(bool)

def encode(mask):
    """bool [h, w] -> compressed RLE that json.dump accepts."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / (area(a) + area(b) - w * h + 1e-9)

def extent(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

def main():
    kins = json.load(open(RAW / "kitti/kins/update_train_2020.json"))
    cats = {c["id"]: c["name"] for c in kins["categories"]}
    image_of = {im["file_name"].split("/")[-1].split(".")[0]: im for im in kins["images"]}   # "000123" -> image record
    anns = {}
    for a in kins["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    dst = PROCESSED / "kitti_object"
    covered = matched = eligible = 0
    for split in ("train", "val"):
        for i in read_split(dst, split):
            if i not in image_of:
                continue
            covered += 1
            s = read_sample(dst, i)
            m, objs = s["meta"], s["objects"]
            H, W = m["height"], m["width"]
            ks = anns.get(image_of[i]["id"], [])
            cost = np.ones((len(objs), len(ks)))
            for r, o in enumerate(objs):
                for c, a in enumerate(ks):
                    if KINS_TO_KITTI.get(cats[a["category_id"]]) == o["class"]:
                        x, y, w, h = a["a_bbox"]
                        cost[r, c] = 1.0 - iou(o["bbox2d_amodal"], [x, y, x + w, y + h])
            pairs = {}
            if objs and ks:
                rows, cols = linear_sum_assignment(cost)
                pairs = {c: r for r, c in zip(rows, cols) if cost[r, c] <= 0.5}
            paint = []                                            # (depth key, object id, visible mask)
            for c, a in enumerate(ks):
                visible = decode(a["i_segm"], H, W)
                amodal = decode(a["a_segm"], H, W)
                if c in pairs:
                    o = objs[pairs[c]]
                else:                                             # KINS-only object: no 3D box
                    x, y, w, h = a["a_bbox"]
                    o = {"id": len(objs) + 1, "class": KINS_TO_KITTI.get(cats[a["category_id"]], cats[a["category_id"]]),
                         "bbox2d_visible": extent(visible) if visible.any() else [x, y, x + w, y + h],
                         "bbox2d_amodal": [x, y, x + w, y + h], "box3d": None, "truncated": None, "occluded": None}
                    objs.append(o)
                o["mask_amodal"] = encode(amodal)
                key = o["box3d"]["center"][2] if o["box3d"] else float("inf")
                paint.append((key, o["id"], visible))
            inst = np.zeros((H, W), np.uint16)
            for key, oid, visible in sorted(paint, key=lambda p: -p[0]):   # far first, so nearer objects overwrite
                inst[visible] = oid
            for o in objs:
                o["has_mask"] = bool((inst == o["id"]).any())
                o.setdefault("mask_amodal", None)
                if o["box3d"] and o["class"] in ("Car", "Pedestrian", "Cyclist") and o["bbox2d_visible"][3] - o["bbox2d_visible"][1] >= 25:
                    eligible += 1
                    matched += o["has_mask"]
            m.update(has_masks=True, has_amodal_masks=True)
            write_sample(dst, i, s["image"], m, objs, depth=s["depth"], instances=inst)
    print(f"KINS covers {covered} kitti_object samples; {matched} of {eligible} Car/Pedestrian/Cyclist labels at least 25 px tall got a mask")

if __name__ == "__main__":
    main()