#!/usr/bin/env python3
"""OPTIONAL small real RGB-D registration example; requires Open3D + network access."""
from pathlib import Path
import argparse
import numpy as np
from prepare_pair import preprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('data/open3d_demo'))
p.add_argument('--max-points',type=int,default=2048);a=p.parse_args()
try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit('Optional dependency missing: install open3d in a compatible Python environment.') from exc
sample=o3d.data.DemoICPPointClouds();a.out.mkdir(parents=True,exist_ok=True)
for i,name in enumerate(['source','target']):
    cloud=o3d.io.read_point_cloud(sample.paths[i]);xyz=np.asarray(cloud.points)
    xyz=preprocess(xyz,0.04,a.max_points,0,float('inf'),42+i)
    np.savetxt(a.out/f'{name}.xyz',xyz,fmt='%.9g')
# Approximate INITIAL GUESS from the official tutorial; NOT ground truth.
T=np.array([[.862,.011,-.507,.5],[-.139,.967,-.215,.7],[.487,.255,.835,-.1],[0,0,0,1.]])
U,_,Vt=np.linalg.svd(T[:3,:3]);D=np.diag([1,1,np.linalg.det(U@Vt)]);T[:3,:3]=U@D@Vt
np.savetxt(a.out/'initial_guess_NOT_ground_truth.txt',T,fmt='%.17g')
print(a.out)
