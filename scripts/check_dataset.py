import argparse, random
import numpy as np
from cage.config import PROCESSED
from cage.data.sample_io import read_sample, read_split
from cage.geometry.camera import unproject_depth, project_points
from cage.geometry.boxes import box3d_corners, points_in_box

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=None) # needed?
    args = p.parse_args()
    
    root = PROCESSED / args.dataset
    ids = [ i for split in ("train", "val", "test") for i in read_split(root, split)]
    
    if seed is not None:
        random.seed(seed)
    
    pick = random.sample(ids, len(ids))
    results = {}
    
    # Completeness check: every listed sample loads and its arrays have the image's size
    bad_samples = []
    for i in ids:
        try:
            sample = read_sample(root, i)
            H, W = sample['image'].shape[:2] # last 2 values 
            assert sample['depth'] is None or s['depth'].shape == (H,W)
            assert sample['instances'] is None or s['instances'].shape == (H,W)
        except Exception as e:
            bad_samples.append((i, repr(e)))
    
    results['completeness'] = (not bad_samples, f"{len(bad_samples)} of {len(ids)} samples fail; " + (f"first: {bad_samples[0]}" if bad_samples else ""))
    
    ious, empty, tall, valid_ratio, medians = [], 0, 0, [], []
    
    for i in pick:
        sample = read_sample(root, i)
        metadata = sample['meta']
        K, R = np.array(metadata["K"], np.array(metadata["R_gravity"])) # intrinsic matrix and rotational matrix
        if sample["depth"] is not None:
            all_points, valid_points = unproject_depth(sample["depth"], K) # point map recovery from camera intrinsics
            valid_ratio.append(valid_points.mean())
            medians.append(np.median(sample["depth"][valid]))
            points_U = all_points[valid] @ R.T
        for obj in sample["objects"]:
            if obj["box3d"] is not None:
                box = obj["box3d"]
                corners_cam = box3d_corners(box["center"], box["size_lwh"], box["heading"]) @ R 
                if (corners_cam[:, 2] > 0.1).all():
                    uv = project_points(corners_cam, K)
                    proj = [max(uv[:, 0].min(), 0), max(uv[:, 1].min(), 0),
                            min(uv[:, 0].max(), m["width"] - 1), min(uv[:, 1].max(), m["height"] - 1)]
                    ious.append(iou2d(proj, o["bbox2d_amodal"]))
                x1, y1, x2, y2 = o["bbox2d_amodal"]
                if s["depth"] is not None and y2 - y1 >= 25:
                    tall += 1
                    empty += not points_in_box(pts_U, b["center"], b["size_lwh"], b["heading"]).any()
    
    # 2. Box reprojection: box3d, K and R_gravity agree with the stored amodal 2D box
    if ious:
        results["box reprojection"] = (np.median(ious) > 0.99, f"median IoU {np.median(ious):.3f} (needs > 0.99)")
    # 3. Points in boxes: depth and boxes are in the same place
    if tall:
        results["points in boxes"] = (empty / tall < 0.05, f"{empty} of {tall} boxes at least 25 px tall hold no depth point (needs < 5%)")
    # 4. Depth statistics, compared by eye with the expectations in D7.4
    if valid_ratio:
        results["depth statistics"] = (True, f"valid-pixel ratio {np.mean(valid_ratio):.3f}, median depth {np.median(medians):.2f} m")

    for name, (ok, msg) in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")
        

if __name__ == "__main__":
    main()