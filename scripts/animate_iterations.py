"""Animate saved CPU iteration coordinates, or reconstruct them from saved poses."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'scripts/result/cpu-g5dVQq'
SOURCE = ROOT / 'data/000001.bin'
TARGET = ROOT / 'data/000000.bin'
OUT = ROOT / 'scripts/result'
LOW = np.array([8., -8.5, -2.])
HIGH = LOW + 3.

rows = np.atleast_1d(np.genfromtxt(RUN / 'iterations.csv', delimiter=',', names=True))
source = np.fromfile(SOURCE, dtype='<f4').reshape(-1, 4)[:, :3]
target = np.fromfile(TARGET, dtype='<f4').reshape(-1, 4)[:, :3]
source_indices = np.flatnonzero(np.all((source >= LOW) & (source <= HIGH), axis=1))
target_indices = np.flatnonzero(np.all((target >= LOW) & (target <= HIGH), axis=1))
p = source[source_indices].astype(float)
q = target[target_indices].astype(float)
positions = [p]
mode = 'reconstructed from recorded poses and original binary coordinates'
if (RUN / 'source_0000.csv').exists():
    mode = 'recorded coordinate dumps'
    original = np.atleast_1d(np.genfromtxt(RUN / 'source_0000.csv', delimiter=',', names=True))
    source_indices = original['point_index'].astype(int)
    positions = [np.column_stack([original[axis] for axis in ['x', 'y', 'z']])]
for row in rows:
    iteration = int(row['iteration'])
    if mode == 'recorded coordinate dumps':
        data = np.atleast_1d(np.genfromtxt(RUN / f'source_{iteration:04d}.csv', delimiter=',', names=True))
        assert np.array_equal(data['point_index'].astype(int), source_indices)
        positions.append(np.column_stack([data[axis] for axis in ['x', 'y', 'z']]))
    else:
        R = np.array([[row[f'r{i}{j}'] for j in range(3)] for i in range(3)])
        t = np.array([row[key] for key in ['tx', 'ty', 'tz']])
        # Match the CPU transform's double expression order and float output.
        transformed = np.empty_like(p)
        for i in range(3):
            transformed[:, i] = (R[i, 0] * p[:, 0] + R[i, 1] * p[:, 1]
                                 + R[i, 2] * p[:, 2] + t[i]).astype(np.float32)
        positions.append(transformed)

# One shared view extent covers every recorded position; no per-frame zoom.
visible = np.vstack([q] + positions)
view_low = LOW.copy()
view_high = HIGH.copy()

fig = plt.figure(figsize=(6, 4.8), dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=8, c='#0072B2', alpha=.75,
           depthshade=False, linewidths=0)
points = ax.scatter(positions[0][:, 0], positions[0][:, 1], positions[0][:, 2],
                    s=8, c='#D55E00', alpha=.75, depthshade=False, linewidths=0)
ax.set(xlim=(view_low[0], view_high[0]), ylim=(view_low[1], view_high[1]),
       zlim=(view_low[2], view_high[2]))
ax.set_axis_off()
ax.view_init(elev=23, azim=-65)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
handles = [Line2D([], [], color='#0072B2', marker='o', linestyle='', label='Target (fixed)'),
           Line2D([], [], color='#D55E00', marker='o', linestyle='', label='Source')]
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.canvas.draw()
# Crop to the union of projected point positions across every frame, not the
# mostly empty 3D axes rectangle. One common crop prevents camera/frame jumps.
projected = proj3d.proj_transform(visible[:, 0], visible[:, 1], visible[:, 2], ax.get_proj())
pixels = ax.transData.transform(np.column_stack(projected[:2]))
left, bottom = pixels.min(axis=0)
right, top = pixels.max(axis=0)
width, height = fig.canvas.get_width_height()
center = (left + right) / 2
label = fig.text(center / width, (top + 9) / height, '',
                 ha='center', va='bottom', fontsize=11)
fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
           bbox_to_anchor=(center / width, (bottom - 5) / height),
           bbox_transform=fig.transFigure, fontsize=10, borderaxespad=0)
crop = Bbox.from_extents((left - 8) / fig.dpi, (bottom - 36) / fig.dpi,
                         (right + 8) / fig.dpi, (top + 32) / fig.dpi)
frames = []
for index, xyz in enumerate(positions):
    points._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    label.set_text(f'Vanilla ICP · iteration {index:02d} / {len(rows)}')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches=crop, pad_inches=0)
    buffer.seek(0)
    frames.append(Image.open(buffer).convert('RGB'))
plt.close(fig)
OUT.mkdir(parents=True, exist_ok=True)
output = OUT / 'cpu-g5dVQq-iterations.gif'
frames[0].save(output, save_all=True, append_images=frames[1:],
               duration=[1200] + [400] * (len(frames) - 2) + [2200], loop=0)
frames[0].save(OUT / 'cpu-g5dVQq-first.png')
frames[-1].save(OUT / 'cpu-g5dVQq-last.png')
(OUT / 'cpu-g5dVQq-animation.json').write_text(json.dumps(dict(
    input_run=str(RUN.relative_to(ROOT)), mode=mode, frames=len(frames),
    source_points=len(source_indices), target_points=len(target_indices),
    bounds=[LOW.tolist(), HIGH.tolist()],
    view_bounds=[view_low.tolist(), view_high.tolist()],
    camera_fixed_across_frames=True, interpolation=False,
    source_selection='all original points in the fixed window; never recropped',
    no_registration_rerun=True, input_sha256={str(path.relative_to(ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SOURCE, TARGET, RUN / 'iterations.csv']}
), indent=2) + '\n')
print(output)
print(mode, '| frames:', len(frames), '| source:', len(source_indices), '| target:', len(target_indices))
