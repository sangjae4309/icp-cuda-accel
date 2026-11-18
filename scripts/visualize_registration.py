"""Render saved KITTI registration poses as 3D PNG and before/after GIF.

Run from the repository root. Edit SOURCE, TARGET, RESULT and OUT below to
visualize another completed registration; this script never runs ICP.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

SOURCE = Path('data/000001.bin')
TARGET = Path('data/000000.bin')
RESULT = Path('scripts/result/full-cloud-check/vanilla.json')
OUT = Path(__file__).resolve().parent / 'result'

out = OUT
out.mkdir(parents=True, exist_ok=True)
source = np.fromfile(SOURCE, dtype='<f4').reshape(-1, 4)[:, :3].astype(float)
target = np.fromfile(TARGET, dtype='<f4').reshape(-1, 4)[:, :3].astype(float)
T = np.array(json.loads(RESULT.read_text())['T_target_source'])
aligned = source @ T[:3, :3].T + T[:3, 3]
low = np.array([8., -8.5, -2.])
high = low + 3.
# Keep the same source points visible in both endpoint views; no frame-wise resampling.
sidx = np.flatnonzero(np.all((source >= low) & (source <= high), axis=1) & np.all((aligned >= low) & (aligned <= high), axis=1))
tidx = np.flatnonzero(np.all((target >= low) & (target <= high), axis=1))
rng = np.random.default_rng(42)
sidx = np.sort(rng.choice(sidx, min(2048, len(sidx)), replace=False))
tidx = np.sort(rng.choice(tidx, min(2048, len(tidx)), replace=False))
p, a, q = source[sidx], aligned[sidx], target[tidx]
np.savez(out / 'closeup-indices.npz', source=sidx, target=tidx)
handles = [Line2D([], [], color='#0072B2', marker='o', linestyle='', label='Target: fixed'),
           Line2D([], [], color='#D55E00', marker='o', linestyle='', label='Source: moved by saved pose')]

def draw(ax, points, title):
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=9, c='#0072B2', alpha=.72, depthshade=False, linewidths=0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=9, c='#D55E00', alpha=.72, depthshade=False, linewidths=0)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
           xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=title)
    ax.view_init(elev=23, azim=-65)
    ax.set_xticks(np.arange(8, 11.1, .5))
    ax.set_yticks(np.arange(-8.5, -5.4, .5))
    ax.set_zticks(np.arange(-2, 1.1, .5))

caption = 'Same local samples: {} source / {} target | 3 m window per axis | no displacement magnification'.format(len(p), len(q))
fig = plt.figure(figsize=(14, 7), dpi=160)
for index, (points, title) in enumerate([(p, 'BEFORE: identity'), (a, 'AFTER: saved vanilla pose')]):
    draw(fig.add_subplot(1, 2, index + 1, projection='3d'), points, title)
fig.suptitle('Vanilla ICP: fixed-camera close-up', fontsize=19)
fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(.5, .035))
fig.text(.5, .015, caption, ha='center', fontsize=10)
fig.tight_layout(rect=(0, .09, 1, .94))
fig.savefig(out / 'vanilla-closeup.png', facecolor='white')
plt.close(fig)
frames = []
for points, title in [(p, 'BEFORE: identity'), (a, 'AFTER: saved vanilla pose')]:
    fig = plt.figure(figsize=(8, 8), dpi=120)
    draw(fig.add_subplot(111, projection='3d'), points, title)
    fig.suptitle('Vanilla ICP | fixed camera, fixed samples', fontsize=16)
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(.5, .045))
    fig.text(.5, .025, 'Before / final result toggle, not iteration playback', ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
    plt.close(fig)
frames[0].save(out / 'vanilla-closeup-toggle.gif', save_all=True, append_images=frames[1:], duration=1600, loop=0)
(out / 'closeup-metadata.json').write_text(json.dumps(dict(bounds=[low.tolist(), high.tolist()], source_count=len(p), target_count=len(q), sampling='within ROI, max 2048 per cloud, fixed source visible before and after', pose=str(RESULT), iterations=json.loads(RESULT.read_text()).get('iterations'), no_icp_rerun=True), indent=2) + '\n')
print(caption)

# Overview: same independently sampled points and camera in both panels.
rng = np.random.default_rng(42)
sample_source = source[rng.choice(len(source), min(2048, len(source)), replace=False)]
sample_target = target[rng.choice(len(target), min(2048, len(target)), replace=False)]
sample_aligned = sample_source @ T[:3, :3].T + T[:3, 3]
fig = plt.figure(figsize=(14, 6), dpi=160)
for i, (points, title) in enumerate([(sample_source, 'Before'), (sample_aligned, 'After')]):
    ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    ax.scatter(sample_target[:, 0], sample_target[:, 1], sample_target[:, 2], s=3,
               c='#0072B2', alpha=.65, depthshade=False)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3,
               c='#D55E00', alpha=.65, depthshade=False)
    ax.set(xlim=(-10, 35), ylim=(-20, 20), zlim=(-3, 8), title=title,
           xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    ax.view_init(elev=25, azim=-120)
fig.legend(handles=handles, loc='lower center', ncol=2)
fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(out / 'alignment-3d.png', facecolor='white')
plt.close(fig)
