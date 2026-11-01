#!/usr/bin/env python3
"""Headless figures for reports; X-Y projection is NOT the 3-D registration metric."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_pair import load_cloud
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('results/alignment'))
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
source=load_cloud(a.source);target=load_cloud(a.target)
T=np.array(json.loads(a.result.read_text())['T_target_source']);aligned=source@T[:3,:3].T+T[:3,3]
for name, pts in [('before',source),('after',aligned)]:
    fig,ax=plt.subplots(figsize=(7,6));ax.scatter(target[:,0],target[:,1],s=4,label='Target',alpha=.65)
    ax.scatter(pts[:,0],pts[:,1],s=4,label='Source' if name=='before' else 'Aligned source',alpha=.65)
    ax.set(xlabel='x',ylabel='y',title=f'{name.capitalize()} registration (XY projection)');ax.set_aspect('equal');ax.legend()
    fig.tight_layout();fig.savefig(a.out/f'{name}.png',dpi=180);plt.close(fig)
print(a.out)
