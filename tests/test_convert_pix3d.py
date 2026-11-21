# tests/test_convert_pix3d.py
"""Tests for cage/data/convert_pix3d.py on a miniature Total3D Pix3D folder built in a temporary folder."""
import json
import pickle

import numpy as np
import pytest
from PIL import Image

from cage.data import convert_pix3d
from cage.data.sample_io import read_split

FOLDER = "./data/pix3d/train_test_data"


def write_split_files(split_dir, train_ids, test_ids):
    split_dir.mkdir(parents=True)
    train = [f"{FOLDER}/{i}{suffix}.pkl" for i in train_ids for suffix in ("", "_flipped")]
    (split_dir / "train.json").write_text(json.dumps(train))
    (split_dir / "test.json").write_text(json.dumps([f"{FOLDER}/{i}.pkl" for i in test_ids]))


def write_pickle(folder, sample_id, class_id, crop_shape=(30, 20, 3)):
    """One file with the keys Total3D's utils/preprocess_pix3d.py writes."""
    rng = np.random.default_rng(sample_id)
    data = {"sample_id": sample_id, "img": rng.integers(0, 255, crop_shape, dtype=np.uint8),
            "gt_3dpoints": rng.uniform(-1, 1, (10000, 3)), "class_id": class_id}
    with open(folder / f"{sample_id}.pkl", "wb") as f:
        pickle.dump(data, f)
    return data


@pytest.fixture
def tree(tmp_path):
    raw, processed, split_dir = tmp_path / "raw", tmp_path / "processed", tmp_path / "splits"
    folder = raw / "pix3d/total3d_processed/train_test_data"              # one nesting level, as the archive may have
    folder.mkdir(parents=True)
    written = {i: write_pickle(folder, i, class_id) for i, class_id in ((0, 1), (2, 3), (3, 5))}
    (folder / "2_flipped.pkl").write_bytes(b"never opened")
    entries = [{"occluded": False, "slightly_occluded": False, "truncated": False},      # id 0
               {"occluded": False, "slightly_occluded": False, "truncated": False},      # id 1 (in no split)
               {"occluded": "true", "slightly_occluded": "false", "truncated": "true"},  # id 2, flags as strings
               {"occluded": False, "slightly_occluded": True, "truncated": False}]       # id 3
    (raw / "pix3d/pix3d.json").write_text(json.dumps(entries))
    write_split_files(split_dir, [3, 2], [0])
    return raw, processed, split_dir, written


def test_split_sample_ids_sorts_and_ignores_flipped_copies(tmp_path):
    write_split_files(tmp_path / "splits", [10003, 10001], [0, 10000])
    assert convert_pix3d.split_sample_ids(tmp_path / "splits", "train") == [10001, 10003]
    assert convert_pix3d.split_sample_ids(tmp_path / "splits", "test") == [0, 10000]


def test_make_splits_holds_out_every_20th_training_id(tmp_path):
    write_split_files(tmp_path / "splits", range(100, 160), range(5))
    splits = convert_pix3d.make_splits(tmp_path / "splits")
    assert splits["val"] == [100, 120, 140] and len(splits["train"]) == 57
    assert not set(splits["train"]) & set(splits["val"]) and splits["test"] == [0, 1, 2, 3, 4]


def test_occlusion_of_reads_booleans_and_strings():
    assert convert_pix3d.occlusion_of({"occluded": True, "slightly_occluded": False}) == "occluded"
    assert convert_pix3d.occlusion_of({"occluded": "True", "slightly_occluded": "True"}) == "occluded"
    assert convert_pix3d.occlusion_of({"occluded": "false", "slightly_occluded": "true"}) == "slight"
    assert convert_pix3d.occlusion_of({"occluded": False, "slightly_occluded": False}) == "none"


def test_main_writes_crop_points_meta_and_splits(tree):
    raw, processed, split_dir, written = tree
    convert_pix3d.main(raw, processed, split_dir)
    root = processed / "pix3d"
    assert read_split(root, "val") == ["2"] and read_split(root, "train") == ["3"] and read_split(root, "test") == ["0"]
    sample_dir = root / "samples" / "2"
    assert (np.asarray(Image.open(sample_dir / "crop.png")) == written[2]["img"]).all()
    points = np.load(sample_dir / "points.npy")
    assert points.dtype == np.float32 and points.shape == (10000, 3)
    assert np.allclose(points, written[2]["gt_3dpoints"], atol=1e-6)
    meta = json.load(open(sample_dir / "meta.json"))
    assert meta == {"dataset": "pix3d", "sample_id": "2", "class": "chair", "class_id": 3, "split": "val",
                    "occlusion": "occluded", "truncated": True}
    assert json.load(open(root / "samples/3/meta.json"))["occlusion"] == "slight"
    assert not (root / "samples" / "2_flipped").exists()


def test_main_reports_ids_without_a_pickle(tree, capsys):
    raw, processed, split_dir, _ = tree
    (raw / "pix3d/total3d_processed/train_test_data/3.pkl").unlink()
    convert_pix3d.main(raw, processed, split_dir)
    assert read_split(processed / "pix3d", "train") == []
    assert "1 ids in the split files have no pickle" in capsys.readouterr().out


def test_main_stops_when_the_folder_holds_the_raw_pix3d_download(tmp_path):
    raw = tmp_path / "raw"
    (raw / "pix3d/total3d_processed/img/bed").mkdir(parents=True)             # the layout of pix3d.zip, not of Total3D's pickles
    (raw / "pix3d/pix3d.json").write_text("[]")
    write_split_files(tmp_path / "splits", [1], [0])
    with pytest.raises(AssertionError, match="not the raw pix3d.zip"):
        convert_pix3d.main(raw, tmp_path / "processed", tmp_path / "splits")
