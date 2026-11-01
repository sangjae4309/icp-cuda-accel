#!/usr/bin/env python3
"""Convert KITTI odometry camera poses to source-LiDAR -> target-LiDAR ground truth."""
import argparse
from pathlib import Path
import numpy as np

def load_poses(path: Path):
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] != 12:
        raise ValueError('KITTI odometry poses must contain 12 values per row')
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses

def load_calibration(path: Path):
    for line in path.read_text().splitlines():
        if line.startswith('Tr:'):
            values = np.fromstring(line.split(':', 1)[1], sep=' ')
            if values.size != 12:
                raise ValueError('Tr must have 12 values')
            C = np.eye(4); C[:3] = values.reshape(3, 4)
            return C  # T_camera0_lidar, rectified odometry convention
    raise ValueError('Missing Tr: in KITTI ODOMETRY calib.txt (not a raw-data calibration file)')

def relative_lidar(poses, C, source_index, target_index):
    if min(source_index, target_index) < 0 or max(source_index, target_index) >= len(poses):
        raise IndexError('Frame index outside pose sequence')
    # T_w_cam for each frame; column-vector convention.
    return np.linalg.inv(C) @ np.linalg.inv(poses[target_index]) @ poses[source_index] @ C

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poses', type=Path, required=True)
    p.add_argument('--calib', type=Path, required=True)
    p.add_argument('--source-index', type=int, required=True)
    p.add_argument('--target-index', type=int, required=True)
    p.add_argument('--out', type=Path, default=Path('data/prepared/T_target_source_gt.txt'))
    a = p.parse_args()
    T = relative_lidar(load_poses(a.poses), load_calibration(a.calib), a.source_index, a.target_index)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out, T, fmt='%.17g')
    print(T)
    print('Ground truth for evaluation ONLY. Do not pass this as --initial in an accuracy experiment.')
