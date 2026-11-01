#!/usr/bin/env python3
"""Relative-pose error, not the KITTI trajectory benchmark metric."""
import argparse
import json
from pathlib import Path
import numpy as np

def pose_error(estimated, ground_truth):
    E = np.linalg.inv(ground_truth) @ estimated
    rotation_deg = np.rad2deg(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) * .5, -1, 1)))
    translation = np.linalg.norm(E[:3, 3])
    return float(translation), float(rotation_deg)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--gt', type=Path, required=True)
    a = p.parse_args()
    r = json.loads(a.result.read_text())
    T = np.asarray(r['T_target_source'], dtype=float)
    gt = np.loadtxt(a.gt)
    if gt.shape == (3, 4): gt = np.vstack([gt, [0, 0, 0, 1]])
    if gt.shape != (4, 4): raise ValueError('Ground truth must be 3x4 or 4x4')
    te, re = pose_error(T, gt)
    print(json.dumps({'translation_error_input_units': te, 'rotation_error_deg': re,
                      'final_inlier_rmse': r['final_rmse'], 'final_inliers': r['final_inliers']}, indent=2))
