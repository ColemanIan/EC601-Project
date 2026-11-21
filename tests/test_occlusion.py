# tests/test_occlusion.py
import numpy as np
from pycocotools import mask as mask_utils
from cage.data.occlusion import amodal_box, encode_rle, extent, occlude_array, paste_object
from cage.geometry.boxes import box3d_corners
from cage.geometry.camera import project_points

K = np.array([[721.5, 0, 609.6], [0, 721.5, 172.9], [0, 0, 1.0]])
W, H = 1242, 375
SIZE = [3.9, 1.6, 1.5]                                   # l, w, h of a car, meters


def rect_mask(c, heading):
    uv = project_points(box3d_corners(np.array(c), SIZE, heading), K)
    m = np.zeros((H, W), bool)
    m[int(uv[:, 1].min()):int(uv[:, 1].max()) + 1, int(uv[:, 0].min()):int(uv[:, 0].max()) + 1] = True
    return m


def scene(positions, seed):
    """A flat gray image with one painted car per (x, z), sparse depth, instance masks and labels."""
    rng = np.random.default_rng(seed)
    image = np.full((H, W, 3), 128, np.uint8)
    depth = np.zeros((H, W), np.float32)
    inst = np.zeros((H, W), np.int64)
    objects = []
    for n, (x, z) in enumerate(sorted(positions, key=lambda p: -p[1])):       # far to near
        c, heading = [x, 1.65 - SIZE[2] / 2, z], float(rng.uniform(-3, 3))
        m = rect_mask(c, heading)
        image[m], inst[m] = rng.integers(30, 255, 3), n + 1
        depth[m] = 0
        depth[m & (np.arange(H)[:, None] % 3 == 0)] = z
        objects.append({"id": n + 1, "class": "Car", "bbox2d_visible": extent(m), "bbox2d_amodal": amodal_box(c, SIZE, heading, K, W, H),
                        "box3d": {"center": c, "size_lwh": SIZE, "heading": heading}, "truncated": 0.0, "occluded": 0,
                        "has_mask": True, "mask_amodal": encode_rle(m)})
    return {"image": image, "depth": depth, "instances": inst, "objects": objects, "meta": {"K": K.tolist()}}


def pasted():
    t, s = scene([(0.0, 30.0)], 1), scene([(2.0, 15.0)], 2)
    rng = np.random.default_rng(0)
    for _ in range(500):
        r = paste_object(t, s, s["objects"][0], rng)
        if r:
            return t, r
    raise AssertionError("no valid placement found")


def test_labels_describe_the_pasted_scene():
    t, r = pasted()
    old, new = r["objects"][0], r["objects"][-1]
    assert old["box3d"] == t["objects"][0]["box3d"] and old["mask_amodal"] == t["objects"][0]["mask_amodal"]   # hidden part stays labeled
    full, visible = mask_utils.decode(new["mask_amodal"]).astype(bool), r["instances"] == new["id"]
    assert visible.any() and not (visible & ~full).any()
    lost = 1 - (r["instances"] == old["id"]).sum() / (t["instances"] == old["id"]).sum()
    assert 0.2 <= lost <= 0.7 and old["occluded"] >= 1
    assert new["box3d"]["center"][2] < old["box3d"]["center"][2]                       # the paste is nearer
    assert (r["instances"][~visible] == t["instances"][~visible]).all()                 # nothing else moved
    assert np.abs(r["depth"][~visible] - t["depth"][~visible]).max() < 1e-3


def test_pasted_box_projects_onto_pasted_pixels():
    _, r = pasted()
    new = r["objects"][-1]
    a, b = new["bbox2d_amodal"], extent(mask_utils.decode(new["mask_amodal"]).astype(bool))
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda q: (q[2] - q[0]) * (q[3] - q[1])
    assert w * h / (area(a) + area(b) - w * h) > 0.8


def test_depth_on_pasted_pixels_belongs_to_the_pasted_object():
    _, r = pasted()
    new = r["objects"][-1]
    d = r["depth"][r["instances"] == new["id"]]
    assert (d > 0).any() and np.abs(d[d > 0] - new["box3d"]["center"][2]).max() < 1e-3   # the test scene's surface sits at the center's z


def test_occlude_array_is_repeatable_and_covers_the_requested_share():
    rng = np.random.default_rng(1)
    crop, rgba = (rng.random((256, 256, 3)) * 255).astype(np.uint8), np.full((100, 150, 4), 200, np.uint8)
    a = occlude_array(crop, rgba, np.random.default_rng(7))
    assert (a == occlude_array(crop, rgba, np.random.default_rng(7))).all()
    assert 0.05 < (np.abs(a.astype(int) - crop).sum(-1) > 0).mean() < 0.65
