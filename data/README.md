# KITTI binary frames

Local original KITTI odometry sequence 00 frames 0–5 provide five adjacent
registration pairs: source 1 -> target 0, 2 -> 1, ..., 5 -> 4.
The default input pair is source `000001.bin`, target `000000.bin`.
No sampling, voxelization or range filtering was applied.

| Frame file | Points |
| --- | ---: |
| 000000.bin | 124668 |
| 000001.bin | 124605 |
| 000002.bin | 124478 |
| 000003.bin | 124167 |
| 000004.bin | 123969 |
| 000005.bin | 123924 |

## Binary layout

Each file is a contiguous sequence of little-endian float32 records. Each point
occupies 16 bytes: `x`, `y`, `z`, `reflectance`, in that order. Coordinates are in
meters in the LiDAR frame. There is no text header or point-count field.
The point count is file size divided by 16. ICP uses XYZ and skips reflectance.
The C++ binary reader assumes the project's little-endian Linux host.

```text
byte offset     0       4       8       12      16 ...
field           x0      y0      z0      r0      x1 ...
type            float32 float32 float32 float32 float32
```

For inspection with NumPy:

```python
import numpy as np
xyzi = np.fromfile('data/000001.bin', dtype='<f4').reshape(-1, 4)
xyz = xyzi[:, :3]
```

## Decoded XYZ examples

These are the first eight XYZ records decoded from each original binary.
They are shown here for readability only; ICP reads the full `.bin` files.
The fourth float in each record (reflectance) is omitted in this display.

Frame 000001 (source):

```text
x y z
52.3059425 0.0229897071 1.97799456
53.2597351 0.106952362 2.00997448
53.2843208 0.274877578 2.01093411
71.1380768 1.04556751 2.61476636
71.0096664 1.26747561 2.61071682
71.5034561 1.38842654 2.62769055
73.8280487 1.66631663 2.70563126
72.2256393 1.85623503 2.65158796
```

Frame 000000 (target):

```text
x y z
52.8979416 0.0229897387 1.99799454
53.7505264 0.192914292 2.02695394
53.8031158 0.361839265 2.02891374
72.6006622 1.29646635 2.66471171
72.1182556 1.51337612 2.64766335
74.4768448 1.79826391 2.72760296
73.2674408 1.99917877 2.6875577
72.1100235 2.19309592 2.64851379
```

Raw frames are copied from
`/projects/standard/zhenman/shared/home/sangjae4309/kitti/velodyne/dataset/sequences/00/velodyne/`.
Binary files remain local and are ignored by Git. Pose/calibration remain in the
shared KITTI directory and are used only for evaluation. This directory has data
for five pairs; it does not imply that those five registrations have been run.

Each implementation has its own `common/kitti_parser.h` and
`common/kitti_parser.cpp`. The entry point calls `load_kitti()` directly.
Only KITTI binary inputs are supported by the C++ executables; there is no XYZ
text parser or intermediate text conversion. Rebuild before using this interface.
