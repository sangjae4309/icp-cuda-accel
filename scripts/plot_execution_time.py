"""Read result.csv and render the existing execution times; never run ICP."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'scripts/result.csv'
OUT = ROOT / 'scripts/result'
IMAGE = ROOT / 'docs/images/execution_time.png'

with CSV.open(newline='') as file:
    rows = [{key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(file, skipinitialspace=True)]

labels = [row['method'] for row in rows]
times = np.array([float(row['execution_time_ms']) for row in rows])

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)
y = np.arange(len(rows))
colors = ['#66788a', '#3b9e91', '#2878b5']
ax.barh(y, times, height=.5, color=colors[:len(rows)])
for index, value in enumerate(times):
    ax.text(value * 1.17, index, f'{value:,.3f} ms', va='center', fontsize=11)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12)
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlim(100, max(times) * 4)
ax.set_xlabel('Complete ICP execution time (ms, logarithmic scale)')
ax.set_title('Vanilla ICP · 20 iterations · full KITTI frame pair', pad=16)
ax.grid(axis='x', which='major', alpha=.2)
ax.set_axisbelow(True)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis='y', length=0)
# Arrow endpoints are measured CPU/GPU times on the logarithmic axis.
for index in range(1, len(rows)):
    arrow_y = index + .34
    ratio = times[0] / times[index]
    ax.annotate('', xy=(times[index], arrow_y), xytext=(times[0], arrow_y),
                arrowprops=dict(arrowstyle='->', color=colors[index], lw=1.8))
    ax.text(np.sqrt(times[0] * times[index]), arrow_y,
            f'{ratio:,.0f}× faster vs CPU', ha='center', va='center',
            fontsize=11, color=colors[index],
            bbox=dict(facecolor='white', edgecolor='none', pad=3))
ax.set_ylim(len(rows) - .3, -.55)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
IMAGE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'execution_time.png', facecolor='white')
fig.savefig(IMAGE, facecolor='white')
plt.close(fig)

table = ['| Method | Execution time (ms) | Runs | Statistic |',
         '| --- | ---: | ---: | --- |']
for row in rows:
    table.append(f"| {row['method']} | {float(row['execution_time_ms']):,.3f} "
                 f"| {row['runs']} | {row['statistic']} |")
markdown = '\n'.join(table)
(OUT / 'execution_time_table.md').write_text(markdown + '\n')
readme = ROOT / 'README.md'
text = readme.read_text()
start = '<!-- execution-time-results:start -->'
end = '<!-- execution-time-results:end -->'
a = text.index(start) + len(start)
b = text.index(end, a)
block = ('\n\n' + markdown + '\n\n'
         '![Measured execution times](docs/images/execution_time.png)\n\n')
readme.write_text(text[:a] + block + text[b:])
print(markdown)
print('Plot:', IMAGE)
