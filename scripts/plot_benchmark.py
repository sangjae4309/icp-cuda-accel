#!/usr/bin/env python3
"""Plots only measured CSV values. No fabricated CUDA bars or projected speedups."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('csv', type=Path)
p.add_argument('--out', type=Path, default=Path('results/plots'))
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
groups = defaultdict(list)
with a.csv.open() as f:
    for row in csv.DictReader(f): groups[(row['backend'],int(row['points']))].append(row)
if not groups: raise ValueError('No data')
for metric, label in [('loop_wall_ms','ICP loop time (ms)'),('registration_wall_ms','Registration time incl. setup/final evaluation (ms)')]:
    fig, ax = plt.subplots(figsize=(8,5))
    for backend in sorted({k[0] for k in groups}):
        sizes = sorted(k[1] for k in groups if k[0] == backend)
        vals = [statistics.median(float(r[metric]) for r in groups[(backend,n)]) for n in sizes]
        ax.plot(sizes, vals, marker='o', label=backend)
    ax.set(xlabel='Points per cloud',ylabel=label,title='Measured median across process-cold repetitions')
    ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(a.out/f'{metric}.png', dpi=180); plt.close(fig)
if any(k[0]=='cpu' for k in groups) and any(k[0].startswith('cuda') for k in groups):
    fig, ax = plt.subplots(figsize=(8,5))
    for backend in sorted({k[0] for k in groups if k[0].startswith('cuda')}):
        sizes = sorted(n for b,n in groups if b==backend and ('cpu',n) in groups)
        speed = [statistics.median(float(r['loop_wall_ms']) for r in groups[('cpu',n)]) /
                 statistics.median(float(r['loop_wall_ms']) for r in groups[(backend,n)]) for n in sizes]
        ax.plot(sizes,speed,marker='o',label=backend)
    ax.set(xlabel='Points per cloud',ylabel='CPU brute-force loop / CUDA loop',title='Same-algorithm speedup (not versus CPU KD-tree)')
    ax.axhline(1,linestyle='--'); ax.legend(); ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(a.out/'speedup.png',dpi=180);plt.close(fig)
print(a.out)
