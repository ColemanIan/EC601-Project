The cage/data directory contains code for procssing data related to the model, specifically not the same as pre-processing datasets. All dataset selection and pre-processing is outside the scope of CAGE documentatoin because CAGE is supposed to be a generalizable framework. 

While some of this could be changed depending on training implementation, CAGE works off the following structured data conventions.


```json
# example_kitti_object.json
{
  "dataset": "kitti_object",
  "sample_id": "000123",
  "split": "train",
  "width": 1242, "height": 375,
  "K": [[721.54, 0.0, 609.56], [0.0, 721.54, 172.85], [0.0, 0.0, 1.0]],
  "R_gravity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "depth_source": "kitti_annotated",
  "depth_bench_split": "train",
  "label_type": "lidar",
  "has_depth": true, "has_masks": true, "has_amodal_masks": true, "has_boxes3d": true, "has_layout": false,
  "raw_ref": "2011_09_26/2011_09_26_drive_0005_sync/0000000109"
}
```

```json
# example_object.json
[
  {
    "id": 1,
    "class": "Car",
    "bbox2d_visible": [612.4, 170.1, 702.9, 230.8],
    "bbox2d_amodal": [605.0, 168.2, 709.3, 232.0],
    "box3d": {"center": [1.84, 1.47, 8.41], "size_lwh": [3.69, 1.67, 1.56], "heading": -1.56},
    "truncated": 0.0, "occluded": 0,
    "has_mask": true,
    "mask_amodal": {"size": [375, 1242], "counts": "<compressed COCO RLE string>"}
  }
]
```