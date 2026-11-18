# Full-cloud CPU and CUDA comparison

Observed 2026-11-17 in the user's existing interactive-gpu allocation 7514083,
node agd06. The allocation was reused; the local shell scripts themselves do not
invoke Slurm. Release builds and all seven executions completed successfully.

## Conditions

- KITTI sequence 00: original frame 1 (124605 points) into frame 0 (124668).
- No sampling, voxelization or range filtering; binary inputs read directly.
- Identity initial pose, 1 m correspondence cutoff, 20-iteration cap for all.
- Translation/rotation convergence thresholds remain 1e-6 m / 1e-6 rad.
- CPU: serial implementation on AMD EPYC 9534; one run. Four CPU cores were
  allocated, but the registration implementation uses one CPU thread.
- GPU: NVIDIA L40S, SM89, driver 580.196.07; three runs per search method.
- GCC 8.5.0, CUDA 12.1.105, Release; contraction disabled on CPU and GPU.
- CPU followed by alternating CUDA naive/tiled runs, sequentially. No dedicated
  warm-up sequence, CPU affinity tuning or fixed GPU clocks. Point dumps disabled.

`ICP_ITERATIONS=20` overrides the vanilla maximum for this matched comparison;
the default CPU maximum is 20 and CUDA default remains 100. All variants actually
completed 20 iterations and returned `converged=0`, so this compares a fixed amount
of observed work, not converged solutions. It does not replace the historical
69-iteration CPU evaluation.

## Accuracy

| Metric | CPU | CUDA brute-force | CUDA tiled |
| --- | ---: | ---: | ---: |
| Completed iterations | 20 | 20 | 20 |
| Converged | no | no | no |
| Final inliers | 123329 | 123329 | 123329 |
| Final inlier RMSE (m) | 0.142262415058 | 0.142262415058 | 0.142262415058 |
| GT translation error (m) | 0.240848923628 | 0.240848923628 | 0.240848923628 |
| GT rotation error (degrees) | 0.125556895097 | 0.125556895097 | 0.125556895097 |

Final poses agree at the 12-significant-digit precision printed by the executable,
across all CUDA repeats. This is output-level parity, not proof of bit-identical
intermediate arrays or a general correctness guarantee. CUDA acceleration did not
improve or worsen the observed pose accuracy on this pair. The CPU 20-iteration
output also reproduces the output supplied by the user before this experiment.

GT is evaluated only after registration using
`inverse(Tr) * inverse(W_0) * W_1 * Tr`, where KITTI W_i are camera poses and
Tr maps LiDAR to camera. Error is `inverse(T_gt) * T_est`; translation norm and
rotation trace angle are reported. GT was not used for initialization or tuning.

## Latency

| Measurement | CPU | CUDA brute-force | CUDA tiled |
| --- | ---: | ---: | ---: |
| Complete registration call (ms) | 315393.125 | 442.488 | 312.209 |
| Complete call range across repeats (ms) | one run | 441.354–484.793 | 311.972–335.569 |
| Mean iteration interval (ms) | 15017.942 | 12.700 | 6.452 |

GPU complete-call values are medians of three runs. GPU mean iteration intervals
are the median of three per-run means. CPU values come from its single run.
The complete-call metric uses the same host CLOCK_MONOTONIC boundary for CPU and
GPU: workspace/event initialization, iterations, final evaluation, timing-log
writes and cleanup. It excludes input loading, compilation and final printing.

Iteration intervals have different scopes and must not be divided to claim a
matched speedup: CPU measures steps 1-5 on the host; CUDA events span pose upload,
steps 1-3 and statistics download, excluding host SVD/pose update. CUDA event
spans can include host submission gaps. Both exclude their timing-log writes.

The observed complete-call latency ratios relative to this serial CPU baseline
are approximately 713 for naive and 1010 for tiled; tiled's median complete-call
latency is approximately 1.42 times lower than naive. These are observations on
this machine, workload and implementation, not general architectural speedups.
There is no threaded/SIMD-tuned CPU or KD-tree baseline, no fixed-clock campaign,
and only one CPU timing sample. The exhaustive CPU search performs roughly
15.5 billion distance comparisons per iteration.

## Reproduction and evidence

Inside an allocated machine with the required toolchain, the existing runners
can build/run directly:

```sh
ICP_ITERATIONS=20 make cpu RUN=1
ICP_ITERATIONS=20 make gpu SEARCH=all RUN=1
```

This experiment set the installed Eigen/CUDA prefixes explicitly in its saved
run script. `scripts/result/cpu-gpu-comparison/` contains exact commands, build
log, seven runtime logs, input/source/binary hashes, per-run comparison JSON,
host identity, GT matrix and the analysis script. Each runtime log identifies
its separate iteration CSV directory. GPU directories contain only iteration
CSV files, no point-coordinate dumps.

The jagent-experiment record
`.agent/memory/experiments/cpu-cuda-fullpair20-l40s-20261117/record.json`
copies the small reports and provenance. It records unknown clocks as unknown;
it is not a clock-controlled matched hardware experiment. No sanitizer,
per-stage intermediate-array verification, CPU point-dump execution or multi-pair
accuracy campaign was performed. Existing generated logs/results remain ignored
by Git; the experiment ledger and this report preserve the selected evidence.
