#!/usr/bin/env python3
"""Reproducible, fixed-iteration synthetic benchmark. Every repetition is a NEW process."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from generate_synthetic import save_pair
from evaluate import pose_error

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-exe', type=Path, default=Path('build-cpu/icp_run'))
    p.add_argument('--gpu-exe', type=Path, default=Path('build-gpu/icp_run'))
    p.add_argument('--backends', nargs='+', choices=['cpu','cuda-naive','cuda-tiled'], default=['cpu'])
    p.add_argument('--sizes', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--out', type=Path, default=Path('results/benchmark'))
    a = p.parse_args()
    if a.repeats < 1 or a.iterations < 1 or a.timeout <= 0 or min(a.sizes) < 4:
        p.error('Invalid benchmark options')
    a.out.mkdir(parents=True, exist_ok=True)
    fields = ['backend','points','repeat','iterations','loop_wall_ms','registration_wall_ms','process_wall_ms',
              'setup_ms','final_eval_ms','transform_ms','match_ms','reduce_ms','download_ms','solve_ms',
              'final_rmse','translation_error','rotation_error_deg','pair_distances_per_second']
    with (a.out/'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for n in a.sizes:
            data = a.out / f'pair_{n}'
            _, _, gt = save_pair(data, n)
            for backend in a.backends:
                for rep in range(a.repeats):
                    out = a.out / f'{backend}_{n}_{rep}.json'
                    exe = a.cpu_exe if backend == 'cpu' else a.gpu_exe
                    cmd = [str(exe.resolve()),'--source',str(data/'source.xyz'),'--target',str(data/'target.xyz'),
                           '--backend',backend,'--iterations',str(a.iterations),'--max-distance','0.7',
                           '--fixed-iters','--output',str(out)]
                    started = time.perf_counter()
                    try:
                        run = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=a.timeout)
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        (a.out/'failure.txt').write_text(str(cmd)+'\n'+str(e)+'\n'+str(getattr(e,'stderr','')))
                        raise RuntimeError('Benchmark failed; see failure.txt. No failed run is converted to a speedup.') from e
                    process_ms = (time.perf_counter()-started)*1000
                    r = json.loads(out.read_text())
                    te, re = pose_error(np.asarray(r['T_target_source']), gt)
                    if te > 1e-4 or re > 0.01 or r['final_rmse'] > 1e-4:
                        raise RuntimeError(f'Correctness gate failed: {out}; translation={te}, rotation={re}')
                    row = {k:r[k] for k in ['iterations','loop_wall_ms','registration_wall_ms','setup_ms','final_eval_ms','final_rmse']}
                    row.update(backend=backend, points=n, repeat=rep, process_wall_ms=process_ms,
                               translation_error=te, rotation_error_deg=re)
                    for k in ['transform_ms','match_ms','reduce_ms','download_ms','solve_ms']:
                        row[k] = sum(x[k] for x in r['trace'])
                    row['pair_distances_per_second'] = n*n*r['iterations']/(row['match_ms']/1000)
                    writer.writerow(row); f.flush()
                    print(run.stdout.strip())
    print('Saved', a.out/'runs.csv')
if __name__ == '__main__': main()
