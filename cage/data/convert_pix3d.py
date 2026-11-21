# cage/data/convert_pix3d.py
import json, pickle, random
from pathlib import Path
import numpy as np
from PIL import Image
from cage.config import RAW, PROCESSED
from cage.data.sample_io import write_splits

REPO = Path(__file__).resolve().parents[2]
PIX3D_CLASSES = ["misc", "bed", "bookcase", "chair", "desk", "sofa", "table", "tool", "wardrobe"]   # total3d/configs/pix3d_config.py order

def split_ids(name):
    return [Path(p).stem for p in json.load(open(REPO / f"total3d/data/pix3d/splits/{name}.json")) if "flipped" not in p]

def main():
    src, dst = RAW / "pix3d/total3d_processed", PROCESSED / "pix3d"
    random.seed(0)
    train = split_ids("train")
    val = set(random.sample(train, len(train) // 20))                 # 5% of Total3D's train split
    splits = {"train": [i for i in train if i not in val], "val": sorted(val), "test": split_ids("test")}
    for split, ids in splits.items():
        for i in ids:
            data = pickle.load(open(next(src.rglob(f"{i}.pkl")), "rb"))
            d = dst / "samples" / i
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(data["img"].convert("RGB") if hasattr(data["img"], "convert") else data["img"], dtype=np.uint8)).save(d / "crop.png")
            np.save(d / "points.npy", np.asarray(data["gt_3dpoints"], dtype=np.float32))
            cid = int(data["class_id"])
            json.dump({"dataset": "pix3d", "sample_id": i, "class": PIX3D_CLASSES[cid], "class_id": cid, "split": split},
                      open(d / "meta.json", "w"), indent=1)
    write_splits(dst, splits)
    print({k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    main()