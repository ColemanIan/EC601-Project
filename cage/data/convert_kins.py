"""KINS masks for kitti_object (plan task D3.4). Run on the SCC after `convert_kitti object`:

python -m cage.data.convert_kins

Adds instances.png (visible masks) and mask_amodal (full silhouettes) to every kitti_object sample
that KINS covers. Running it again gives the same result.
"""
import json

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from cage.config import PROCESSED, RAW
from cage.data.sample_io import read_sample, read_split, write_sample
from cage.geometry.frames import KINS_TO_KITTI_CLASS, KITTI_CLASSES

MIN_MATCH_IOU = 0.5                                # a KINS instance and a KITTI label must overlap this much
REPORT_CLASSES = tuple(KITTI_CLASSES)              # the classes counted in the printed coverage


def decode_mask(segmentation, height, width):
    """A COCO segmentation (polygon list, uncompressed RLE or compressed RLE) -> bool [height, width]."""
    if isinstance(segmentation, list):                                    # polygons
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):                        # uncompressed RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:                                                                 # compressed RLE
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


def encode_mask(mask):
    """bool [height, width] -> compressed COCO RLE whose 'counts' is a str, so json.dump accepts it."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def xywh_to_xyxy(box):
    """COCO box [x, y, width, height] -> CAGE box [x1, y1, x2, y2]."""
    x, y, box_width, box_height = box
    return [float(x), float(y), float(x + box_width), float(y + box_height)]


def box_iou(a, b):
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    overlap_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return overlap_w * overlap_h / (area(a) + area(b) - overlap_w * overlap_h + 1e-9)


def mask_extent(mask):
    """bool mask -> the [x1, y1, x2, y2] box around its true pixels."""
    rows, cols = np.nonzero(mask)
    return [float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())]


def match_to_labels(objects, kins_boxes, kins_classes):
    """One-to-one matching of KINS instances to KITTI objects.

    objects: objects.json entries. kins_boxes: amodal [x1, y1, x2, y2] per KINS instance.
    kins_classes: KITTI class name per KINS instance.
    Returns {index into kins_boxes: index into objects} for pairs of the same class with IoU >= MIN_MATCH_IOU.
    """
    if not objects or not kins_boxes:
        return {}
    cost = np.ones((len(objects), len(kins_boxes)))                   # 1 = cannot match
    for row, obj in enumerate(objects):
        for col, (box, kitti_class) in enumerate(zip(kins_boxes, kins_classes)):
            if kitti_class == obj["class"]:
                cost[row, col] = 1.0 - box_iou(obj["bbox2d_amodal"], box)
    rows, cols = linear_sum_assignment(cost)                          # Hungarian method: lowest total cost
    return {col: row for row, col in zip(rows, cols) if cost[row, col] <= 1.0 - MIN_MATCH_IOU}


def add_kins_masks(objects, annotations, category_name, height, width):
    """Merge one image's KINS annotations into its objects.

    objects: the sample's objects.json list. annotations: KINS annotation dicts for this image.
    category_name: {KINS category id: KINS category name}.
    Returns (objects, instances): the updated list and the uint16 visible-instance id image.
    """
    # Start from the KITTI labels only. A KITTI label always has a 3D box, so an entry without one
    # was appended by an earlier run of this function; dropping it makes a second run harmless.
    objects = [dict(obj, has_mask=False, mask_amodal=None) for obj in objects if obj["box3d"] is not None]
    kins_names = [category_name.get(ann["category_id"], f"kins_{ann['category_id']}") for ann in annotations]
    # A KINS name with no KITTI equivalent ("rider") keeps its own name, so it can never match a KITTI label.
    kins_classes = [KINS_TO_KITTI_CLASS.get(name, name) for name in kins_names]
    kins_boxes = [xywh_to_xyxy(ann["a_bbox"]) for ann in annotations]            # a_ = amodal
    object_of_annotation = match_to_labels(objects, kins_boxes, kins_classes)

    next_id = max((obj["id"] for obj in objects), default=0) + 1
    layers = []                                    # (camera depth, object id, visible mask), painted far to near
    for index, ann in enumerate(annotations):
        visible = decode_mask(ann["i_segm"], height, width)                      # i_ = inmodal = visible
        amodal = decode_mask(ann["a_segm"], height, width)
        if index in object_of_annotation:
            obj = objects[object_of_annotation[index]]
        else:                                      # KINS drew an object that KITTI did not label: no 3D box
            obj = {"id": next_id, "class": kins_classes[index],
                   "bbox2d_visible": mask_extent(visible) if visible.any() else kins_boxes[index],
                   "bbox2d_amodal": kins_boxes[index], "box3d": None, "truncated": None, "occluded": None,
                   "has_mask": False, "mask_amodal": None}
            next_id += 1
            objects.append(obj)
        obj["mask_amodal"] = encode_mask(amodal)
        depth_key = obj["box3d"]["center"][2] if obj["box3d"] else float("inf")   # unknown depth = farthest
        layers.append((depth_key, obj["id"], visible))

    instances = np.zeros((height, width), np.uint16)
    for depth_key, object_id, visible in sorted(layers, key=lambda layer: -layer[0]):
        instances[visible] = object_id             # painted far first, so a nearer object overwrites
    for obj in objects:
        obj["has_mask"] = bool((instances == obj["id"]).any())
    return objects, instances


def main(raw=RAW, processed=PROCESSED):
    kins = json.load(open(raw / "kitti/kins/update_train_2020.json"))
    category_name = {category["id"]: category["name"] for category in kins["categories"]}
    image_record = {record["file_name"].split("/")[-1].split(".")[0]: record for record in kins["images"]}
    annotations_of_image = {}                                         # KINS image id -> its annotations
    for ann in kins["annotations"]:
        annotations_of_image.setdefault(ann["image_id"], []).append(ann)

    out_root = processed / "kitti_object"
    n_covered = n_eligible = n_with_mask = 0
    for split in ("train", "val"):
        for sample_id in read_split(out_root, split):
            if sample_id not in image_record:                         # KINS skips a few KITTI images
                continue
            n_covered += 1
            sample = read_sample(out_root, sample_id)
            meta = sample["meta"]
            annotations = annotations_of_image.get(image_record[sample_id]["id"], [])
            objects, instances = add_kins_masks(sample["objects"], annotations, category_name,
                                                meta["height"], meta["width"])
            for obj in objects:
                box_height_px = obj["bbox2d_visible"][3] - obj["bbox2d_visible"][1]
                if obj["box3d"] and obj["class"] in REPORT_CLASSES and box_height_px >= 25:
                    n_eligible += 1
                    n_with_mask += obj["has_mask"]
            meta.update(has_masks=True, has_amodal_masks=True)
            write_sample(out_root, sample_id, sample["image"], meta, objects,
                         depth=sample["depth"], instances=instances)
    print(f"KINS covers {n_covered} kitti_object samples; {n_with_mask} of {n_eligible} "
          f"Car/Pedestrian/Cyclist labels at least 25 px tall got a mask")


if __name__ == "__main__":
    main()
