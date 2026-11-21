"""Dataset reports (plan task D8). Run on the SCC after any converter:

python -m cage.data.report                      # the raw inventory, and a report for every processed dataset that exists
python -m cage.data.report --dataset sunrgbd    # one dataset

A report answers five questions about a dataset:
  1. Is it there, and how is it laid out?         folder tree, split sizes, number of files of each type
  2. Is it complete and self-consistent?          every sample has the files its meta.json promises
  3. What metadata does it carry?                 image sizes, intrinsics, camera tilt, flags, classes
  4. Which transformations produced it?           the formulas and matrices, quoted from the frame registry
  5. Do depth, boxes and camera agree?            geometry checks on a fixed random draw of samples

Questions 1 to 4 read only the JSON files of every sample. Question 5 reads the images of N_GEOMETRY samples.

It writes  processed/<dataset>/REPORT.md  (for reading) and  processed/<dataset>/report.json  (for tests),
and the inventory of the downloads to  processed/RAW_INVENTORY.md  and  processed/raw_inventory.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from cage.config import PROCESSED, RAW
from cage.data.sample_io import read_sample
from cage.geometry import frames
from cage.geometry.boxes import box3d_corners, points_in_box
from cage.geometry.camera import project_points, unproject_depth

N_GEOMETRY = 50                 # samples whose images are read for the geometry checks
MIN_BOX_IOU = 0.99              # median overlap of the projected 3D box with the stored amodal 2D box
MAX_EMPTY_BOXES = 0.05          # share of boxes at least MIN_BOX_PX tall that may hold no depth point
MIN_BOX_PX = 25

# The downloads, relative to raw/. The inventory counts the files of each type in each of them.
RAW_SOURCES = {                       # name: (path under raw/, folder levels listed in the inventory)
    "kitti_depth": ("kitti/depth", 1),
    "kitti_object": ("kitti/object", 2),
    "kins": ("kitti/kins", 1),
    "sunrgbd_total3d": ("sunrgbd/total3d_processed", 1),
    "pix3d_total3d": ("pix3d/total3d_processed", 1),
    "pix3d_json": ("pix3d/pix3d.json", 1),
}

# The processed datasets. `expected` is the number of samples a complete conversion gives, or a (low, high) range.
DATASETS = {
    "kitti_depth_val": {"kind": "scene", "built_from": ["raw/kitti/depth"], "expected": 1000},
    "kitti_object": {"kind": "scene", "built_from": ["raw/kitti/object", "raw/kitti/depth", "raw/kitti/kins"], "expected": 7481},
    "sunrgbd": {"kind": "scene", "built_from": ["raw/sunrgbd/total3d_processed"], "expected": (10000, 10335)},
    "pix3d": {"kind": "object", "built_from": ["raw/pix3d/total3d_processed", "raw/pix3d/pix3d.json"], "expected": 10069},
    "kitti_object_occ": {"kind": "scene", "built_from": ["processed/kitti_object"], "expected": None},
    "occluders_indoor": {"kind": "cutouts", "built_from": ["processed/sunrgbd"], "expected": None},
}
REQUIRED = ["kitti_depth_val", "kitti_object", "sunrgbd", "pix3d"]      # the other two are built later, in D6

# Which files a scene sample must hold, by the flag in meta.json that promises each one.
FILE_OF_FLAG = {"has_depth": "depth.png", "has_masks": "instances.png", "has_layout": "layout.json"}
ALWAYS = ["image.png", "meta.json", "objects.json"]


# ---------------------------------------------------------------- transformations, quoted from the registry

def _matrix(name):
    entry = frames.REGISTRY["axis_changes"][name]
    return {"name": name, "from": entry["from"], "to": entry["to"], "matrix": entry["matrix"]}


def transforms_for(name):
    """The conversions a dataset went through, as a list of {"step", "formula", ...} dicts."""
    depth = {"step": "depth encoding", "formula": f"depth_m = png_value / {frames.DEPTH_PNG_SCALE:g}; 0 = no measurement",
             "source": "configs/frames.json: encodings"}
    if name == "kitti_depth_val":
        return [depth,
                {"step": "frame", "formula": "none: the depth maps are already in camera 2, CAGE's camera frame. R_gravity = I.",
                 "source": "KITTI depth devkit"},
                {"step": "intrinsics", "formula": "K is read from each frame's intrinsics file", "source": "per sample: meta.json K"}]
    if name in ("kitti_object", "kitti_object_occ"):
        steps = [depth,
                 {"step": "camera 2 intrinsics", "formula": "P2 = K [I | t_cam2];  K = P2[:, :3];  t_cam2 = K^-1 P2[:, 3]",
                  "source": "per sample: meta.json K and t_cam2 (cage.geometry.frames.split_kitti_projection)"},
                 {"step": "LiDAR to camera 2", "formula": "p_C = R0_rect (Tr_velo_to_cam [p_V; 1]) + t_cam2",
                  "source": "per sample calibration file (frames.velodyne_to_camera2)",
                  "ideal_rotation": _matrix("VELO_TO_CAM_IDEAL")},
                 {"step": "label to box", "formula": "center = location + t_cam2 - (0, h/2, 0);  (l, w, h) = (dim[2], dim[1], dim[0]);  heading = rotation_y",
                  "source": "frames.kitti_label_to_box"},
                 {"step": "KINS masks", "formula": "pixel masks; boxes [x, y, w, h] -> [x, y, x + w, y + h]; no 3D transform",
                  "source": "cage/data/convert_kins.py"},
                 {"step": "gravity", "formula": "R_gravity = I: KITTI's rectified camera is treated as level, so the upright frame is the camera frame",
                  "source": "cage/data/convert_kitti.py"}]
        if name == "kitti_object_occ":
            steps.append({"step": "pasted object", "formula": "image scale = (f_target / f_source) (z_source / z_pasted);  "
                          "heading' = heading - atan2(x_source, z_source) + atan2(x_pasted, z_pasted)", "source": "cage/data/occlusion.py"})
        return steps
    if name == "sunrgbd":
        return [depth,
                {"step": "depth cap", "formula": f"depth >= {frames.SUNRGBD_DEPTH_CAP_M:g} m becomes 0 (Total3D clipped depth there)",
                 "source": "configs/frames.json: encodings"},
                {"step": "axis change", "formula": "p_opencv = T p_total3d, and p_upright = T p_world. T is its own inverse.",
                 "source": "configs/frames.json: axis_changes.T", "matrix": _matrix("T")},
                {"step": "camera tilt", "formula": "angle = mean(bins[cls]) + reg * bin_width",
                 "source": "configs/frames.json: total3d_bins_deg", "bins_deg": frames.REGISTRY["total3d_bins_deg"]},
                {"step": "gravity", "formula": "R_gravity = T cam_R T = R_x(pitch) R_z(roll);  p_upright = R_gravity p_camera",
                 "source": "per sample: meta.json R_gravity (frames.gravity_from_total3d)"},
                {"step": "boxes", "formula": "corners_U = T corners_W;  length, width, height = front, side, up extents;  heading = atan2(-front_z, front_x)",
                 "source": "frames.box_from_total3d_corners",
                 "total3d_to_cage_corner": frames.TOTAL3D_TO_CAGE_CORNER},
                {"step": "amodal 2D box", "formula": "bbox2d_amodal = bdb2D_from_3D * (cx, cy, cx, cy)", "source": "cage/data/convert_sunrgbd.py"}]
    if name == "pix3d":
        return [{"step": "points", "formula": "none: points.npy keeps Total3D's normalized model frame (bounding box centered, longest half-extent 1)",
                 "source": "total3d/libs/tools.py::normalize_to_unit_square"},
                {"step": "placing a shape in a box (used later, by the mesh head)",
                 "formula": "p_U = center + R_y(heading) diag(l/2, h/2, w/2) MODEL_TO_BOX n", "source": "frames.place_model_points",
                 "matrix": _matrix("MODEL_TO_BOX")}]
    return []


# ---------------------------------------------------------------- counting files

OWN_FILES = ("REPORT.md", "report.json")       # the report's own files are not counted, so a second run counts the same


def count_files(path, depth=2):
    """Every file under `path` -> {"files", "by_extension", "by_folder"}; folders are cut to `depth` levels."""
    path = Path(path)
    if path.is_file():
        return {"files": 1, "by_extension": {path.suffix or "(none)": 1}, "by_folder": {".": {path.suffix or "(none)": 1}}}
    by_extension, by_folder = Counter(), {}
    for folder, _, names in os.walk(path):
        relative = Path(folder).relative_to(path).parts[:depth]
        key = "/".join(relative) or "."
        for name in names:
            if name in OWN_FILES:
                continue
            extension = Path(name).suffix or "(none)"
            by_extension[extension] += 1
            by_folder.setdefault(key, Counter())[extension] += 1
    return {"files": sum(by_extension.values()), "by_extension": dict(sorted(by_extension.items())),
            "by_folder": {k: dict(sorted(v.items())) for k, v in sorted(by_folder.items())}}


def raw_inventory(raw=RAW):
    """What is in each download folder. Missing folders are reported, not raised."""
    inventory = {}
    for name, (relative, depth) in RAW_SOURCES.items():
        path = raw / relative
        inventory[name] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            inventory[name].update(count_files(path, depth))
    return inventory


# ---------------------------------------------------------------- reading one processed dataset

def read_ids(root, split):
    path = root / f"{split}.txt"
    return [line.strip() for line in open(path) if line.strip()] if path.exists() else None


def span(values):
    """A list of numbers -> {"min", "mean", "max"}, or None when the list is empty."""
    return {"min": float(np.min(values)), "mean": float(np.mean(values)), "max": float(np.max(values))} if len(values) else None


def split_summary(root, sample_ids, problems):
    """Split sizes, plus problems: ids without a folder, folders in no split, ids in two splits."""
    splits, seen = {}, Counter()
    for split in ("train", "val", "test"):
        ids = read_ids(root, split)
        if ids is None:
            problems.append(f"{split}.txt is missing")
            continue
        splits[split] = len(ids)
        seen.update(ids)
        absent = sorted(set(ids) - sample_ids)
        if absent:
            problems.append(f"{len(absent)} ids of {split}.txt have no sample folder, for example {absent[0]}")
    doubled = [i for i, n in seen.items() if n > 1]
    if doubled:
        problems.append(f"{len(doubled)} ids are listed in more than one split, for example {doubled[0]}")
    orphans = sorted(sample_ids - set(seen))
    if orphans:
        problems.append(f"{len(orphans)} sample folders are in no split, for example {orphans[0]}")
    return splits


def summarize_scene(root, name, problems):
    """Read every meta.json and objects.json of a scene dataset (no images) and summarize them."""
    sample_dirs = sorted(p for p in (root / "samples").iterdir() if p.is_dir()) if (root / "samples").exists() else []
    files, sizes, flags, depth_sources, classes = Counter(), Counter(), Counter(), Counter(), Counter()
    fx, fy, cx, cy, pitch, roll, t_cam2 = [], [], [], [], [], [], []
    n_objects = n_box3d = n_visible_mask = n_amodal_mask = n_identity = 0
    for sample_dir in sample_dirs:
        present = {p.name for p in sample_dir.iterdir()}
        files.update(present)
        if not {"meta.json", "objects.json"} <= present:
            problems.append(f"{sample_dir.name}: meta.json or objects.json is missing")
            continue
        meta = json.load(open(sample_dir / "meta.json"))
        objects = json.load(open(sample_dir / "objects.json"))
        wanted = ALWAYS + [file for flag, file in FILE_OF_FLAG.items() if meta.get(flag)]
        for file in wanted:
            if file not in present:
                problems.append(f"{sample_dir.name}: {file} is missing")
        for flag, file in FILE_OF_FLAG.items():
            if not meta.get(flag) and file in present:
                problems.append(f"{sample_dir.name}: {file} exists but meta.json says {flag} = false")
        sizes[f"{meta['width']} x {meta['height']}"] += 1
        flags.update(flag for flag in ("has_depth", "has_masks", "has_amodal_masks", "has_boxes3d", "has_layout") if meta.get(flag))
        depth_sources[str(meta.get("depth_source"))] += 1
        K, R = np.array(meta["K"], dtype=np.float64), np.array(meta["R_gravity"], dtype=np.float64)
        fx.append(K[0, 0]); fy.append(K[1, 1]); cx.append(K[0, 2]); cy.append(K[1, 2])
        if not (K[0, 0] > 0 and K[1, 1] > 0 and 0 < K[0, 2] < meta["width"] and 0 < K[1, 2] < meta["height"]):
            problems.append(f"{sample_dir.name}: K is not a plausible intrinsics matrix")
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)):
            problems.append(f"{sample_dir.name}: R_gravity is not a rotation matrix")
        identity = np.allclose(R, np.eye(3), atol=1e-9)
        n_identity += identity
        if name.startswith("kitti") and not identity:
            problems.append(f"{sample_dir.name}: R_gravity must be the identity on KITTI")
        p, r = frames.pitch_roll_from_gravity(R)
        pitch.append(np.degrees(p)); roll.append(np.degrees(r))
        if "t_cam2" in meta:
            t_cam2.append(meta["t_cam2"])
        ids = [o["id"] for o in objects]
        if len(ids) != len(set(ids)) or any(i < 1 for i in ids):
            problems.append(f"{sample_dir.name}: object ids must be unique and at least 1")
        for o in objects:
            n_objects += 1
            classes[o["class"]] += 1
            n_visible_mask += bool(o.get("has_mask"))
            n_amodal_mask += o.get("mask_amodal") is not None
            if o.get("box3d") is not None:
                n_box3d += 1
                if min(o["box3d"]["size_lwh"]) <= 0:
                    problems.append(f"{sample_dir.name}: object {o['id']} has a box with a non-positive size")
    summary = {"samples": len(sample_dirs), "files_per_name": dict(sorted(files.items())),
               "splits": split_summary(root, {p.name for p in sample_dirs}, problems),
               "image_sizes": dict(sizes.most_common(8)), "flags": dict(flags), "depth_source": dict(depth_sources),
               "K": {"fx": span(fx), "fy": span(fy), "cx": span(cx), "cy": span(cy)},
               "R_gravity": {"identity_in": int(n_identity), "pitch_deg": span(pitch), "roll_deg": span(roll)},
               "objects": {"total": n_objects, "with_box3d": n_box3d, "with_visible_mask": n_visible_mask,
                           "with_amodal_mask": n_amodal_mask, "per_class": dict(classes.most_common())}}
    if t_cam2:
        summary["t_cam2"] = {"mean": np.mean(t_cam2, axis=0).tolist(), "min": np.min(t_cam2, axis=0).tolist(),
                             "max": np.max(t_cam2, axis=0).tolist()}
    return summary


def summarize_pix3d(root, problems, n_points_checked=200):
    """Read every meta.json of processed/pix3d, and the points of a spread of samples."""
    sample_dirs = sorted(p for p in (root / "samples").iterdir() if p.is_dir()) if (root / "samples").exists() else []
    files, classes, occlusion = Counter(), Counter(), Counter()
    truncated = 0
    for sample_dir in sample_dirs:
        present = {p.name for p in sample_dir.iterdir()}
        files.update(present)
        for file in ("crop.png", "points.npy", "meta.json"):
            if file not in present:
                problems.append(f"{sample_dir.name}: {file} is missing")
        if "meta.json" in present:
            meta = json.load(open(sample_dir / "meta.json"))
            classes[meta["class"]] += 1
            occlusion[meta["occlusion"]] += 1
            truncated += bool(meta["truncated"])
            if frames.PIX3D_CLASSES[meta["class_id"]] != meta["class"]:
                problems.append(f"{sample_dir.name}: class_id {meta['class_id']} does not name class {meta['class']}")
    lows, highs = [], []
    for sample_dir in sample_dirs[::max(1, len(sample_dirs) // n_points_checked)]:
        if not (sample_dir / "points.npy").exists():
            continue
        points = np.load(sample_dir / "points.npy")
        lows.append(points.min(axis=0)); highs.append(points.max(axis=0))
        if points.shape != (10000, 3) or not np.isfinite(points).all():
            problems.append(f"{sample_dir.name}: points.npy has shape {points.shape} or holds non-finite values")
        elif not (np.allclose(points.max(axis=0) + points.min(axis=0), 0.0, atol=1e-4) and np.isclose(np.abs(points).max(), 1.0, atol=1e-4)):
            problems.append(f"{sample_dir.name}: points are not centered with longest half-extent 1")
    summary = {"samples": len(sample_dirs), "files_per_name": dict(sorted(files.items())),
               "splits": split_summary(root, {p.name for p in sample_dirs}, problems),
               "per_class": dict(classes.most_common()), "occlusion": dict(occlusion), "truncated": truncated,
               "points_checked": len(lows)}
    if lows:
        summary["points_min_xyz"] = np.min(lows, axis=0).tolist()
        summary["points_max_xyz"] = np.max(highs, axis=0).tolist()
    return summary


# ---------------------------------------------------------------- geometry checks on a draw of samples

def box_iou(a, b):
    """Overlap of two [x1, y1, x2, y2] boxes: intersection area over union area."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return width * height / (area(a) + area(b) - width * height + 1e-9)


def geometry_checks(root, problems, n=N_GEOMETRY, seed=0):
    """Read `n` samples in full and test that their arrays, boxes, depth and camera describe one scene.

    array sizes       depth.png and instances.png have the image's size
    box reprojection  box3d, moved from the upright frame to the camera with R_gravity and projected with K,
                      lands on the stored bbox2d_amodal. A wrong axis, sign, heading or corner order fails here.
    points in boxes   the depth points of the sample, moved to the upright frame, fall inside the 3D boxes.
                      Depth and boxes that are in different frames or units fail here.
    depth statistics  share of pixels with a measurement and the median depth, for comparison with D8.3
    """
    ids = sorted(p.name for p in (root / "samples").iterdir() if p.is_dir())
    picked = random.Random(seed).sample(ids, min(n, len(ids)))
    ious, valid_share, median_depth = [], [], []
    tall_boxes = empty_boxes = not_in_front = 0
    for sample_id in picked:
        try:
            sample = read_sample(root, sample_id)
        except Exception as error:                              # a missing or unreadable file was already reported
            problems.append(f"{sample_id}: cannot be read ({error!r})")
            continue
        meta = sample["meta"]
        size = (meta["height"], meta["width"])
        for key in ("depth", "instances"):
            if sample[key] is not None and sample[key].shape != size:
                problems.append(f"{sample_id}: {key} has shape {sample[key].shape}, the image has {size}")
        K, R_gravity = np.array(meta["K"]), np.array(meta["R_gravity"])
        points_upright = None
        if sample["depth"] is not None:
            points, valid = unproject_depth(sample["depth"], K)
            valid_share.append(float(valid.mean()))
            if valid.any():
                median_depth.append(float(np.median(sample["depth"][valid])))
            points_upright = points[valid] @ R_gravity.T        # p_U = R_gravity p_C, for row vectors
        for obj in sample["objects"]:
            if obj["box3d"] is None:
                continue
            box = obj["box3d"]
            corners_camera = box3d_corners(box["center"], box["size_lwh"], box["heading"]) @ R_gravity   # p_C = R_gravity^T p_U
            if (corners_camera[:, 2] > 0.1).all():              # a box that crosses the camera plane has no 2D box
                pixels = project_points(corners_camera, K)
                projected = [max(pixels[:, 0].min(), 0), max(pixels[:, 1].min(), 0),
                             min(pixels[:, 0].max(), meta["width"] - 1), min(pixels[:, 1].max(), meta["height"] - 1)]
                ious.append(box_iou(projected, obj["bbox2d_amodal"]))
            else:
                not_in_front += 1
            if points_upright is not None and obj["bbox2d_amodal"][3] - obj["bbox2d_amodal"][1] >= MIN_BOX_PX:
                tall_boxes += 1
                empty_boxes += not points_in_box(points_upright, box["center"], box["size_lwh"], box["heading"]).any()
    result = {"samples_read": len(picked), "boxes_projected": len(ious), "boxes_not_in_front": not_in_front,
              "median_box_iou": float(np.median(ious)) if ious else None,
              "tall_boxes": tall_boxes, "tall_boxes_without_points": int(empty_boxes),
              "valid_pixel_share": float(np.mean(valid_share)) if valid_share else None,
              "median_depth_m": float(np.median(median_depth)) if median_depth else None}
    if ious and result["median_box_iou"] <= MIN_BOX_IOU:
        problems.append(f"box reprojection: median IoU {result['median_box_iou']:.3f}, needs more than {MIN_BOX_IOU}")
    if tall_boxes and empty_boxes / tall_boxes >= MAX_EMPTY_BOXES:
        problems.append(f"points in boxes: {empty_boxes} of {tall_boxes} boxes hold no depth point, needs under {MAX_EMPTY_BOXES:.0%}")
    return result


# ---------------------------------------------------------------- building and writing a report

def build_report(name, processed=PROCESSED, check_count=True):
    """Everything known about one processed dataset, as a JSON-serializable dict. report["problems"] lists what is wrong.

    check_count=False skips the comparison with the size of a complete conversion; the tests use it on small synthetic datasets.
    """
    spec, root, problems = DATASETS[name], processed / name, []
    report = {"dataset": name, "path": str(root), "exists": root.exists(), "kind": spec["kind"],
              "built_from": spec["built_from"], "expected_samples": spec["expected"],
              "generated": datetime.datetime.now().isoformat(timespec="seconds"),
              "registry": {"path": str(frames.REGISTRY_PATH), "version": frames.REGISTRY.get("version"),
                           "sha256": hashlib.sha256(frames.REGISTRY_PATH.read_bytes()).hexdigest()[:16]},
              "transforms": transforms_for(name), "problems": problems}
    if not root.exists():
        problems.append(f"{root} does not exist")
        return report
    report["file_types"] = count_files(root, depth=1)
    if spec["kind"] == "scene":
        report["summary"] = summarize_scene(root, name, problems)
        report["geometry"] = geometry_checks(root, problems)
    elif spec["kind"] == "object":
        report["summary"] = summarize_pix3d(root, problems)
    else:
        report["summary"] = {"samples": len(list(root.glob("*.png")))}
    n, expected = report["summary"]["samples"], spec["expected"]
    low, high = expected if isinstance(expected, tuple) else (expected, expected)
    if check_count and expected is not None and not low <= n <= high:
        problems.append(f"{n} samples, but a complete conversion gives {low if low == high else f'{low} to {high}'}")
    report["problems"] = problems[:50] + ([f"... and {len(problems) - 50} more"] if len(problems) > 50 else [])
    return report


def _table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _fmt_span(s, unit=""):
    return "none" if s is None else f"{s['min']:.2f} to {s['max']:.2f}{unit} (mean {s['mean']:.2f})"


def _fmt_matrix(m):
    return "\n".join("    [" + ", ".join(f"{v:g}".rjust(3) for v in row) + "]" for row in m)


def to_markdown(report):
    """The report as a page a person can read top to bottom."""
    name, lines = report["dataset"], []
    status = "no problems found" if not report["problems"] else f"{len(report['problems'])} problem(s), listed at the end"
    lines += [f"# Dataset report: {name}", "", f"Generated {report['generated']} by `python -m cage.data.report`. Status: **{status}**.", "",
              f"- Location: `{report['path']}`", f"- Built from: {', '.join(report['built_from'])}",
              f"- Frame registry: `{report['registry']['path']}`, version {report['registry']['version']}, sha256 {report['registry']['sha256']}"]
    summary = report.get("summary")
    if summary:
        lines += ["", "## Layout", "", "```", f"{name}/"]
        if "splits" in summary:
            lines.append("  " + "   ".join(f"{k}.txt ({v:,} ids)" for k, v in summary["splits"].items()))
        if "files_per_name" in summary:
            lines.append(f"  samples/   ({summary['samples']:,} folders, one per sample)")
            lines.append("    <sample_id>/")
            lines += [f"      {file:<16}{count:>8,}" for file, count in summary["files_per_name"].items()]
        else:
            lines.append(f"  *.png   ({summary['samples']:,} cut-outs)")
        lines += ["```", "", "## Files by type", "",
                  _table(sorted(report["file_types"]["by_extension"].items()), ["extension", "files"])]
    if summary and report["kind"] == "scene":
        K, R, objects = summary["K"], summary["R_gravity"], summary["objects"]
        lines += ["", "## Metadata", "",
                  _table([["image sizes (width x height: samples)", ", ".join(f"{k}: {v:,}" for k, v in summary["image_sizes"].items())],
                          ["flags (samples with the flag set)", ", ".join(f"{k}: {v:,}" for k, v in summary["flags"].items()) or "none"],
                          ["depth source", ", ".join(f"{k}: {v:,}" for k, v in summary["depth_source"].items())],
                          ["K: fx (px)", _fmt_span(K["fx"])], ["K: fy (px)", _fmt_span(K["fy"])],
                          ["K: cx (px)", _fmt_span(K["cx"])], ["K: cy (px)", _fmt_span(K["cy"])],
                          ["R_gravity is the identity in", f"{R['identity_in']:,} of {summary['samples']:,} samples"],
                          ["camera pitch (negative looks down)", _fmt_span(R["pitch_deg"], " deg")],
                          ["camera roll", _fmt_span(R["roll_deg"], " deg")]]
                         + ([["t_cam2, mean (m)", ", ".join(f"{v:.4f}" for v in summary["t_cam2"]["mean"])]] if "t_cam2" in summary else []),
                         ["quantity", "value"]),
                  "", "## Objects", "",
                  f"{objects['total']:,} objects: {objects['with_box3d']:,} with a 3D box, {objects['with_visible_mask']:,} with a visible mask, "
                  f"{objects['with_amodal_mask']:,} with an amodal mask.", "",
                  _table(list(objects["per_class"].items())[:45], ["class", "objects"])]
    if report.get("geometry"):
        g = report["geometry"]
        fmt = lambda v, f: "none" if v is None else format(v, f)
        lines += ["", "## Geometry checks", "", f"Read in full: {g['samples_read']} samples, drawn with a fixed seed.", "",
                  _table([["box reprojection: median IoU of the projected 3D box with bbox2d_amodal",
                           f"{fmt(g['median_box_iou'], '.3f')} over {g['boxes_projected']:,} boxes", f"more than {MIN_BOX_IOU}"],
                          ["boxes not projected because a corner is behind or beside the camera", f"{g['boxes_not_in_front']:,}",
                           "a small share: objects cut by the image border"],
                          [f"points in boxes: boxes at least {MIN_BOX_PX} px tall with no depth point inside",
                           f"{g['tall_boxes_without_points']:,} of {g['tall_boxes']:,}", f"under {MAX_EMPTY_BOXES:.0%}"],
                          ["share of pixels with a depth measurement", fmt(g["valid_pixel_share"], ".3f"), "compare with D8.3"],
                          ["median depth (m)", fmt(g["median_depth_m"], ".2f"), "compare with D8.3"]],
                         ["check", "measured", "passes when"])]
    if summary and report["kind"] == "object":
        lines += ["", "## Metadata", "",
                  _table([["occlusion", ", ".join(f"{k}: {v:,}" for k, v in summary["occlusion"].items())],
                          ["truncated", f"{summary['truncated']:,}"],
                          [f"points range over {summary['points_checked']} samples, min (x, y, z)", summary.get("points_min_xyz")],
                          ["max (x, y, z)", summary.get("points_max_xyz")]], ["quantity", "value"]),
                  "", "## Classes", "", _table(list(summary["per_class"].items()), ["class", "samples"])]
    if report["transforms"]:
        lines += ["", "## Transformations that produced this dataset", ""]
        for t in report["transforms"]:
            lines += [f"**{t['step']}.** `{t['formula']}`  ", f"Defined in: {t['source']}"]
            for key in ("matrix", "ideal_rotation"):
                if key in t:
                    m = t[key]
                    lines += ["", f"`{m['name']}` maps {m['from']} to {m['to']}:", "", "```", _fmt_matrix(m["matrix"]), "```"]
            if "bins_deg" in t:
                lines += ["", f"Pitch bins (degrees): {t['bins_deg']['pitch']}; roll bins: {t['bins_deg']['roll']}"]
            if "total3d_to_cage_corner" in t:
                lines += ["", f"Total3D corner i is CAGE corner {t['total3d_to_cage_corner']}[i]."]
            lines.append("")
    lines += ["## Problems", ""] + ([f"- {p}" for p in report["problems"]] or ["None."])
    return "\n".join(lines) + "\n"


def write_report(name, processed=PROCESSED, check_count=True):
    """Build the report of one dataset and, when the dataset exists, write REPORT.md and report.json into it."""
    report = build_report(name, processed, check_count)
    if report["exists"]:
        root = processed / name
        (root / "report.json").write_text(json.dumps(report, indent=1))
        (root / "REPORT.md").write_text(to_markdown(report))
    return report


def write_raw_inventory(raw=RAW, processed=PROCESSED):
    """Count the files of every download and write processed/RAW_INVENTORY.md and processed/raw_inventory.json."""
    inventory = raw_inventory(raw)
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "raw_inventory.json").write_text(json.dumps(inventory, indent=1))
    lines = ["# Raw inventory", "", f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} by `python -m cage.data.report`.", ""]
    for name, entry in inventory.items():
        lines += [f"## {name}", "", f"`{entry['path']}`: " + (f"{entry['files']:,} files" if entry["exists"] else "**missing**"), ""]
        if entry["exists"]:
            rows = [[folder, ", ".join(f"{ext}: {n:,}" for ext, n in counts.items())] for folder, counts in entry["by_folder"].items()]
            lines += [_table(rows[:60], ["folder", "files by extension"]), ""]
    (processed / "RAW_INVENTORY.md").write_text("\n".join(lines))
    return inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    args = parser.parse_args()
    if not args.dataset:
        inventory = write_raw_inventory()
        for name, entry in inventory.items():
            print(f"raw {name:<18} " + (f"{entry['files']:>8,} files" if entry["exists"] else "MISSING"))
    for name in [args.dataset] if args.dataset else DATASETS:
        report = write_report(name)
        if not report["exists"]:
            print(f"{name:<18} not built yet")
        else:
            print(f"{name:<18} {report['summary']['samples']:>7,} samples, {len(report['problems'])} problem(s)  -> {report['path']}/REPORT.md")
            for problem in report["problems"][:5]:
                print(f"    {problem}")


if __name__ == "__main__":
    main()
