#!/usr/bin/env python3
"""Offline preprocessing shared by ALL backends; no download or SLAM dependencies."""
import argparse
import json
from pathlib import Path
import numpy as np

def load_cloud(path: Path):
    if path.suffix.lower() == '.bin':
        raw = np.fromfile(path, dtype='<f4')
        if raw.size == 0 or raw.size % 4:
            raise ValueError(f'{path}: expected KITTI float32 xyzi records')
        xyz = raw.reshape(-1, 4)[:, :3].astype(np.float64)
    else:
        xyz = np.loadtxt(path, comments='#', ndmin=2)[:, :3]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError('Expected nonempty Nx3 cloud')
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite point; clean the input explicitly')
    return xyz

def preprocess(xyz, voxel, cap, min_range, max_range, seed):
    radius = np.linalg.norm(xyz, axis=1)
    xyz = xyz[(radius >= min_range) & (radius <= max_range)]
    if voxel > 0 and len(xyz):
        keys = np.floor(xyz / voxel).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(sums, inv, xyz)
        xyz = sums / counts[:, None]
    if len(xyz) > cap:
        selected = np.random.default_rng(seed).choice(len(xyz), cap, replace=False)
        xyz = xyz[np.sort(selected)]
    if len(xyz) < 3:
        raise ValueError('Preprocessing left fewer than 3 points')
    return xyz

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('data/prepared'))
    p.add_argument('--max-points', type=int, default=2048)
    p.add_argument('--voxel-size', type=float, default=0.2)
    p.add_argument('--min-range', type=float, default=2.0)
    p.add_argument('--max-range', type=float, default=60.0)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.max_points < 3 or a.voxel_size < 0 or a.min_range < 0 or a.max_range <= a.min_range:
        p.error('Invalid preprocessing parameters')
    a.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for i, (name, path) in enumerate([('source', a.source), ('target', a.target)]):
        raw = load_cloud(path)
        xyz = preprocess(raw, a.voxel_size, a.max_points, a.min_range, a.max_range, a.seed + i)
        np.savetxt(a.out / f'{name}.xyz', xyz, fmt='%.9g')
        counts[name] = {'original': len(raw), 'prepared': len(xyz)}
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    meta['counts'] = counts
    (a.out / 'preprocessing.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(counts, indent=2))
