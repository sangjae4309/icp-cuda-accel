#!/usr/bin/env python3
"""Create a reproducible rigid pair. T_target_source is ground truth, not initialization."""
import argparse
from pathlib import Path
import json
import numpy as np

def generate(n: int, seed: int = 42, angle_deg: float = 3.0):
    if n < 4:
        raise ValueError("Use at least four non-coplanar points")
    rng = np.random.default_rng(seed)
    target = rng.uniform([-2.0, -1.6, -0.8], [2.0, 1.6, 1.2], size=(n, 3))
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]])
    t = np.array([0.03, -0.025, 0.02])
    source = (target - t) @ R  # inverse rigid transform with row-vector storage
    target = target[rng.permutation(n)]  # correspondence is NOT the same file index
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return source, target, T

def save_pair(out: Path, n: int, seed: int = 42, angle_deg: float = 3.0):
    source, target, T = generate(n, seed, angle_deg)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / 'source.xyz', source, fmt='%.9g')
    np.savetxt(out / 'target.xyz', target, fmt='%.9g')
    np.savetxt(out / 'T_target_source_gt.txt', T, fmt='%.17g')
    (out / 'metadata.json').write_text(json.dumps({
        'kind': 'synthetic, exact rigid transform, no noise, full overlap',
        'points': n, 'seed': seed, 'angle_deg': angle_deg,
        'target_permuted': True, 'units': 'arbitrary length units',
        'purpose': 'correctness test, not evidence of real-world SLAM accuracy'
    }, indent=2) + '\n')
    return source, target, T

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data/synthetic'))
    p.add_argument('--points', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--angle-deg', type=float, default=3.0)
    args = p.parse_args()
    save_pair(args.out, args.points, args.seed, args.angle_deg)
    print(args.out)
