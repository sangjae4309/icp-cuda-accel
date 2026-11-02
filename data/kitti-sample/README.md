# KITTI sample pair

Goal: use ICP to estimate the rotation and translation that align source frame 1
with target frame 0.

- `velodyne/000001.bin`: original source point cloud (sequence 00, frame 1).
- `velodyne/000000.bin`: original target point cloud (sequence 00, frame 0).
- `source.xyz`, `target.xyz`: corresponding prepared clouds, stored as readable
  `x y z` coordinates in meters, one point per line, with 2,048 points each.
- `preprocessing.json`: conversion settings and point counts.

The binaries store float32 `(x, y, z, reflectance)` records. The XYZ files omit
reflectance, filter to 2–60 m, merge points into 0.2 m voxel centroids, and sample
2,048 points per cloud. They are reduced inputs, not ground truth.
