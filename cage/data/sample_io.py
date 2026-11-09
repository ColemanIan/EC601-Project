import json
from pathlib import Path
import numpy as np
from PIL import Image
from cage.data.io_depth import read_depth_png, write_depth_png

META_KEYS = {"dataset", "sample_id", "split", "width", "height", "K", "R_gravity", "depth_source", "label_type",
             "has_depth", "has_masks", "has_amodal_masks", "has_boxes3d", "has_layout", "raw_ref"}

def write_sample(root, sample_id, image, meta, objects, depth=None, instances=None, layout=None):
    """root = processed/<dataset>. image uint8 [H, W, 3]; depth float meters [H, W]; instances int [H, W]."""
    missing = META_KEYS - meta.keys()
    assert not missing, f"meta.json lacks {missing}"
    H, W = image.shape[:2]
    assert (meta["width"], meta["height"]) == (W, H), "meta.json size differs from the image"
    assert meta["has_depth"] == (depth is not None) and meta["has_masks"] == (instances is not None)
    d = Path(root) / "samples" / sample_id
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(d / "image.png")
    if depth is not None:
        assert depth.shape == (H, W), "depth size differs from the image"
        write_depth_png(d / "depth.png", depth)
    if instances is not None:
        assert instances.shape == (H, W), "instances size differs from the image"
        Image.fromarray(np.asarray(instances, dtype=np.uint16)).save(d / "instances.png")
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(d / "objects.json", "w") as f:
        json.dump(objects, f, indent=1)
    if layout is not None:
        with open(d / "layout.json", "w") as f:
            json.dump(layout, f, indent=1)

def read_sample(root, sample_id):
    d = Path(root) / "samples" / sample_id
    meta = json.load(open(d / "meta.json"))
    return {
        "meta": meta,
        "objects": json.load(open(d / "objects.json")),
        "image": np.asarray(Image.open(d / "image.png").convert("RGB")),
        "depth": read_depth_png(d / "depth.png") if meta["has_depth"] else None,
        "instances": np.asarray(Image.open(d / "instances.png")).astype(np.int64) if meta["has_masks"] else None,
        "layout": json.load(open(d / "layout.json")) if meta["has_layout"] else None,
    }

def write_splits(root, splits):
    """splits = {"train": [ids], "val": [ids], "test": [ids]}; a missing split becomes an empty file."""
    for name in ("train", "val", "test"):
        (Path(root) / f"{name}.txt").write_text("".join(f"{i}\n" for i in splits.get(name, [])))

def read_split(root, split):
    return [line.strip() for line in open(Path(root) / f"{split}.txt") if line.strip()]