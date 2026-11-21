"""Pix3D converter (plan task D5.2). Run on the SCC:

python -m cage.data.convert_pix3d

Reads Total3D's per-image pickles from raw/pix3d/total3d_processed and the occlusion flags from
raw/pix3d/pix3d.json, and writes processed/pix3d.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from cage.config import PROCESSED, RAW
from cage.data.sample_io import write_splits
from cage.geometry.frames import PIX3D_CLASSES

REPO = Path(__file__).resolve().parents[2]                       # the EC601 folder
SPLIT_DIR = REPO / "total3d/data/pix3d/splits"                   # train.json, test.json: lists of pickle paths
VAL_EVERY = 20                                                   # every 20th training image is held out for validation


def split_sample_ids(split_dir, name):
    """Total3D's split file -> the sorted Pix3D sample ids in it, flipped copies left out.

    A sample id is the image's index in pix3d.json; Total3D names each pickle <id>.pkl.
    """
    stems = [Path(path).stem for path in json.load(open(split_dir / f"{name}.json"))]
    return sorted(int(stem) for stem in stems if "flipped" not in stem)


def make_splits(split_dir):
    """Total3D's test split is kept; every VAL_EVERY-th id of its train split becomes validation."""
    train_ids = split_sample_ids(split_dir, "train")
    val_ids = train_ids[::VAL_EVERY]
    held_out = set(val_ids)
    return {"train": [i for i in train_ids if i not in held_out], "val": val_ids,
            "test": split_sample_ids(split_dir, "test")}


def is_true(value):
    """pix3d.json writes its flags as JSON booleans or as the strings 'true' / 'false'."""
    return str(value).lower() == "true"


def occlusion_of(pix3d_entry):
    """One pix3d.json entry -> 'occluded', 'slight' or 'none'."""
    if is_true(pix3d_entry["occluded"]):
        return "occluded"
    return "slight" if is_true(pix3d_entry["slightly_occluded"]) else "none"


def main(raw=RAW, processed=PROCESSED, split_dir=SPLIT_DIR):
    pickle_root = raw / "pix3d/total3d_processed"
    out_root = processed / "pix3d"
    pix3d_entries = json.load(open(raw / "pix3d/pix3d.json"))            # one entry per image; list index = sample id
    pickle_path = {int(path.stem): path for path in pickle_root.rglob("*.pkl") if path.stem.isdigit()}   # flips excluded
    assert pickle_path, f"no <id>.pkl files under {pickle_root}; it must hold Total3D's pickles, not the raw pix3d.zip"

    written = {"train": [], "val": [], "test": []}
    missing = []
    for split, sample_ids in make_splits(split_dir).items():
        for sample_number in sample_ids:
            if sample_number not in pickle_path:
                missing.append(sample_number)
                continue
            with open(pickle_path[sample_number], "rb") as f:
                data = pickle.load(f)                                    # keys: sample_id, img, gt_3dpoints, class_id
            sample_id = str(sample_number)
            sample_dir = out_root / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(data["img"], dtype=np.uint8)).convert("RGB").save(sample_dir / "crop.png")
            np.save(sample_dir / "points.npy", np.asarray(data["gt_3dpoints"], dtype=np.float32))
            class_id = int(data["class_id"])
            entry = pix3d_entries[sample_number]
            meta = {"dataset": "pix3d", "sample_id": sample_id, "class": PIX3D_CLASSES[class_id],
                    "class_id": class_id, "split": split,
                    "occlusion": occlusion_of(entry), "truncated": is_true(entry["truncated"])}
            with open(sample_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=1)
            written[split].append(sample_id)
    write_splits(out_root, written)
    print({split: len(ids) for split, ids in written.items()},
          f"; {len(missing)} ids in the split files have no pickle", missing[:5])


if __name__ == "__main__":
    main()
