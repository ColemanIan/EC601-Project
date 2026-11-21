# tests/test_convert_kins.py
"""Tests for cage/data/convert_kins.py on hand-built annotations (rectangles as COCO polygons)."""
import json

import numpy as np

from cage.data import convert_kins
from cage.data.sample_io import read_sample, write_sample, write_splits

WIDTH, HEIGHT = 200, 60
CATEGORY_NAME = {1: "cyclist", 2: "pedestrian", 4: "car", 7: "van"}


def kitti_object(object_id, kitti_class, box, depth):
    """A minimal objects.json entry: amodal box [x1, y1, x2, y2] and the camera depth of its 3D center."""
    return {"id": object_id, "class": kitti_class, "bbox2d_visible": list(box), "bbox2d_amodal": list(box),
            "box3d": {"center": [0.0, 1.0, depth], "size_lwh": [3.9, 1.6, 1.5], "heading": 0.0},
            "truncated": 0.0, "occluded": 0, "has_mask": False, "mask_amodal": None}


def rectangle(x1, y1, x2, y2):
    """A rectangle as a COCO polygon list."""
    return [[x1, y1, x2, y1, x2, y2, x1, y2]]


def kins_annotation(category_id, amodal, visible, image_id=0):
    """A KINS annotation whose amodal and visible shapes are the rectangles [x1, y1, x2, y2]."""
    xywh = lambda b: [b[0], b[1], b[2] - b[0], b[3] - b[1]]
    return {"image_id": image_id, "category_id": category_id,
            "a_bbox": xywh(amodal), "a_segm": rectangle(*amodal), "i_bbox": xywh(visible), "i_segm": rectangle(*visible)}


def test_decode_mask_reads_polygons_and_both_rle_forms():
    polygon = convert_kins.decode_mask(rectangle(20, 10, 60, 40), HEIGHT, WIDTH)
    assert polygon.shape == (HEIGHT, WIDTH) and polygon[25, 40] and not polygon[5, 5] and not polygon[25, 100]
    assert abs(int(polygon.sum()) - 40 * 30) <= 0.15 * 40 * 30                # edge pixels depend on the rasterizer
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[10:40, 20:60] = True
    assert (convert_kins.decode_mask(convert_kins.encode_mask(mask), HEIGHT, WIDTH) == mask).all()
    # Uncompressed RLE counts pixels column by column, starting with a run of zeros.
    column_major = mask.T.ravel()
    changes = np.flatnonzero(np.diff(column_major.astype(int))) + 1
    counts = np.diff(np.r_[0, changes, column_major.size]).tolist()
    assert (convert_kins.decode_mask({"size": [HEIGHT, WIDTH], "counts": counts}, HEIGHT, WIDTH) == mask).all()


def test_encode_mask_can_be_written_to_json():
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[5:9, 7:30] = True
    restored = json.loads(json.dumps(convert_kins.encode_mask(mask)))
    assert (convert_kins.decode_mask(restored, HEIGHT, WIDTH) == mask).all()


def test_xywh_to_xyxy_and_box_iou():
    assert convert_kins.xywh_to_xyxy([10, 20, 30, 5]) == [10.0, 20.0, 40.0, 25.0]
    assert convert_kins.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) > 0.999
    assert abs(convert_kins.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6      # overlap 50, union 150
    assert convert_kins.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_match_needs_the_same_class_and_enough_overlap():
    objects = [kitti_object(1, "Car", [20, 10, 60, 40], 10.0)]
    same = convert_kins.match_to_labels(objects, [[21, 10, 61, 40]], ["Car"])
    other_class = convert_kins.match_to_labels(objects, [[21, 10, 61, 40]], ["Van"])
    small_overlap = convert_kins.match_to_labels(objects, [[50, 10, 90, 40]], ["Car"])
    assert same == {0: 0} and other_class == {} and small_overlap == {}
    assert convert_kins.match_to_labels([], [[0, 0, 5, 5]], ["Car"]) == {}


def test_match_is_one_to_one():
    objects = [kitti_object(1, "Car", [20, 10, 60, 40], 10.0)]
    boxes = [[24, 10, 64, 40], [20, 10, 60, 40]]                              # the second fits better
    assert convert_kins.match_to_labels(objects, boxes, ["Car", "Car"]) == {1: 0}


def test_add_kins_masks_attaches_matches_and_appends_unlabeled_instances():
    objects = [kitti_object(1, "Car", [20, 10, 60, 40], 10.0)]
    annotations = [kins_annotation(4, [20, 10, 60, 40], [20, 10, 50, 40]),     # the labeled car, right side hidden
                   kins_annotation(2, [120, 10, 135, 50], [120, 10, 135, 50])]  # a pedestrian KITTI did not label
    merged, instances = convert_kins.add_kins_masks(objects, annotations, CATEGORY_NAME, HEIGHT, WIDTH)
    car, pedestrian = merged
    assert car["id"] == 1 and car["has_mask"] and car["box3d"] is not None
    amodal = convert_kins.decode_mask(car["mask_amodal"], HEIGHT, WIDTH)
    visible = instances == 1
    assert visible.any() and not (visible & ~amodal).any() and amodal.sum() > visible.sum()
    assert pedestrian["id"] == 2 and pedestrian["class"] == "Pedestrian" and pedestrian["box3d"] is None
    assert pedestrian["has_mask"] and pedestrian["bbox2d_amodal"] == [120.0, 10.0, 135.0, 50.0]
    assert set(np.unique(instances)) == {0, 1, 2}
    assert objects[0]["mask_amodal"] is None                                   # the input list is not modified


def test_nearer_object_wins_where_visible_masks_overlap():
    objects = [kitti_object(1, "Car", [20, 10, 80, 40], 25.0), kitti_object(2, "Car", [60, 10, 120, 40], 9.0)]
    annotations = [kins_annotation(4, [20, 10, 80, 40], [20, 10, 80, 40]),
                   kins_annotation(4, [60, 10, 120, 40], [60, 10, 120, 40]),
                   kins_annotation(7, [100, 10, 160, 40], [100, 10, 160, 40])]   # an unlabeled van: depth unknown
    _, instances = convert_kins.add_kins_masks(objects, annotations, CATEGORY_NAME, HEIGHT, WIDTH)
    assert instances[25, 70] == 2                                              # overlap of the two cars: the near one
    assert instances[25, 110] == 2                                             # overlap with the van: the known depth wins
    assert instances[25, 30] == 1 and instances[25, 150] == 3


def test_second_run_gives_the_same_result():
    objects = [kitti_object(1, "Car", [20, 10, 60, 40], 10.0)]
    annotations = [kins_annotation(4, [20, 10, 60, 40], [20, 10, 50, 40]),
                   kins_annotation(2, [120, 10, 135, 50], [120, 10, 135, 50])]
    first, first_instances = convert_kins.add_kins_masks(objects, annotations, CATEGORY_NAME, HEIGHT, WIDTH)
    second, second_instances = convert_kins.add_kins_masks(first, annotations, CATEGORY_NAME, HEIGHT, WIDTH)
    assert json.dumps(first) == json.dumps(second) and (first_instances == second_instances).all()
    # The result depends only on the KITTI labels and the annotations given: objects that an earlier
    # run appended do not survive a run in which KINS no longer lists them.
    emptied, emptied_instances = convert_kins.add_kins_masks(first, [], CATEGORY_NAME, HEIGHT, WIDTH)
    assert [o["id"] for o in emptied] == [1] and emptied[0]["mask_amodal"] is None and not emptied_instances.any()


def test_unknown_category_id_becomes_an_unmatched_object():
    annotations = [kins_annotation(3, [20, 10, 60, 40], [20, 10, 60, 40])]     # id 3 is not in CATEGORY_NAME
    merged, _ = convert_kins.add_kins_masks([kitti_object(1, "Car", [20, 10, 60, 40], 10.0)], annotations,
                                            CATEGORY_NAME, HEIGHT, WIDTH)
    assert [o["class"] for o in merged] == ["Car", "kins_3"] and merged[1]["box3d"] is None
    assert not merged[0]["has_mask"]


def test_main_updates_covered_samples_and_keeps_depth(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    root = processed / "kitti_object"
    meta = {"dataset": "kitti_object", "split": "train", "width": WIDTH, "height": HEIGHT,
            "K": np.eye(3).tolist(), "R_gravity": np.eye(3).tolist(), "depth_source": "kitti_annotated",
            "label_type": "lidar", "has_depth": True, "has_masks": False, "has_amodal_masks": False,
            "has_boxes3d": True, "has_layout": False, "raw_ref": ""}
    image = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    depth = np.full((HEIGHT, WIDTH), 12.0)
    for sample_id in ("000000", "000001"):                                     # KINS will cover only 000000
        write_sample(root, sample_id, image, dict(meta, sample_id=sample_id),
                     [kitti_object(1, "Car", [20, 10, 60, 40], 10.0)], depth=depth)
    write_splits(root, {"train": ["000000", "000001"]})
    (raw / "kitti/kins").mkdir(parents=True)
    kins = {"categories": [{"id": i, "name": n} for i, n in CATEGORY_NAME.items()],
            "images": [{"id": 0, "file_name": "000000.png", "width": WIDTH, "height": HEIGHT}],
            "annotations": [kins_annotation(4, [20, 10, 60, 40], [20, 10, 50, 40])]}
    (raw / "kitti/kins/update_train_2020.json").write_text(json.dumps(kins))

    convert_kins.main(raw, processed)
    covered, skipped = read_sample(root, "000000"), read_sample(root, "000001")
    assert covered["meta"]["has_masks"] and covered["meta"]["has_amodal_masks"]
    assert covered["objects"][0]["has_mask"] and (covered["instances"] == 1).any()
    assert np.allclose(covered["depth"], 12.0)                                 # depth written earlier is kept
    assert not skipped["meta"]["has_masks"] and skipped["instances"] is None
