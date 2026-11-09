"""View one processed sample.
SCC or laptop: python scripts/view_sample.py --dataset kitti_object --id 000123          # writes files
Laptop:        python scripts/view_sample.py --dataset kitti_object --id 000123 --show   # writes and opens
Laptop:        python scripts/view_sample.py --open runs/views/kitti_object/000123      # opens copied files
"""
import argparse
from pathlib import Path
import numpy as np

def draw_panel(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n = 1 + (s["depth"] is not None) + (s["instances"] is not None)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    ax = ax[0]
    ax[0].imshow(s["image"])
    for o in s["objects"]:
        x1, y1, x2, y2 = o["bbox2d_amodal"]
        ax[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="yellow", lw=1))
        ax[0].text(x1, y1, o["class"], color="yellow", fontsize=7)
    k = 1
    if s["depth"] is not None:
        shown = ax[k].imshow(np.ma.masked_equal(s["depth"], 0), cmap="turbo")
        fig.colorbar(shown, ax=ax[k], label="depth (m)")
        k += 1
    if s["instances"] is not None:
        ax[k].imshow(s["image"])
        ax[k].imshow(np.ma.masked_equal(s["instances"], 0) % 20, cmap="tab20", alpha=0.5, interpolation="nearest")
    for a in ax:
        a.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def write_3d(s, out):
    import trimesh
    from cage.geometry.camera import unproject_depth
    from cage.geometry.boxes import box3d_corners
    R = np.array(s["meta"]["R_gravity"])
    if s["depth"] is not None:
        pts, valid = unproject_depth(s["depth"], np.array(s["meta"]["K"]))
        trimesh.PointCloud(pts[valid] @ R.T, colors=s["image"][valid]).export(out / "cloud.ply")   # upright frame
    corners = [box3d_corners(o["box3d"]["center"], o["box3d"]["size_lwh"], o["box3d"]["heading"])
               for o in s["objects"] if o["box3d"] is not None]
    np.save(out / "boxes.npy", np.array(corners).reshape(-1, 8, 3))

def show(out):
    import open3d as o3d
    from cage.geometry.boxes import BOX_EDGES
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)]
    if (out / "cloud.ply").exists():
        geoms.append(o3d.io.read_point_cloud(str(out / "cloud.ply")))
    for c in np.load(out / "boxes.npy"):
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(c), o3d.utility.Vector2iVector(BOX_EDGES))
        lines.paint_uniform_color((1.0, 0.0, 0.0))
        geoms.append(lines)
    F = np.diag([1.0, -1.0, -1.0])               # the upright frame has y down; Open3D shows y up
    for g in geoms:
        g.rotate(F, center=(0.0, 0.0, 0.0))
    o3d.visualization.draw_geometries(geoms, window_name=str(out))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--id")
    p.add_argument("--show", action="store_true")
    p.add_argument("--open")
    a = p.parse_args()
    if a.open:
        return show(Path(a.open))
    from cage.config import PROCESSED, RUNS
    from cage.data.sample_io import read_sample
    s = read_sample(PROCESSED / a.dataset, a.id)
    out = RUNS / "views" / a.dataset / a.id
    out.mkdir(parents=True, exist_ok=True)
    draw_panel(s, out / "panel.png")
    write_3d(s, out)
    print(f"wrote {out}")
    if a.show:
        show(out)

if __name__ == "__main__":
    main()