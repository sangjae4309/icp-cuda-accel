# CPU and CUDA: whole 20-iteration loop timing

Remeasured on 2026-11-17 at the user's request under allocation 7514083 on
agd06: AMD EPYC 9534 CPU and NVIDIA L40S (SM89). The earlier complete-call
442.488 ms naive result included initialization and logging; it is not the
execution metric used in the current README. That historical measurement is
retained below rather than relabeled as computation-only time.

## Identical measurement conditions

- Original KITTI sequence 00 frame 1 -> 0; 124605 source / 124668 target points.
- Identity pose, 1 m cutoff, 1e-6 translation/rotation thresholds, same precision.
- Exactly 20 updates, early stopping disabled, one execution per method.
- Same node/allocation, Release builds, GCC 8.5.0, CUDA 12.1.105, contraction off.
- Serial CPU baseline; four allocated CPU cores do not make it multithreaded.
- Sequential CPU, CUDA naive, CUDA tiled, no competing benchmark processes.
- No special warm-up, fixed-clock/affinity campaign or repeat aggregation.

A single `CLOCK_MONOTONIC` interval begins immediately before the iteration loop
and ends immediately after it, identically on CPU and CUDA. This is not a sum of
per-iteration times and is not a CUDA-event-only metric. CUDA Step 4 (host SVD)
and Step 5 (host pose update/convergence) are included, along with loop control
and the per-update pose upload/statistics download needed by the algorithm.
Each statistics download is synchronous; GPU work completes before host Step 4,
so there is no unfinished device work when the loop stop timestamp is taken.

Excluded on both targets: file loading, allocation, initial original-cloud upload,
initialization, all logging/CSV writes/coordinate dumps, per-iteration timers and
CUDA events, final error evaluation and memory cleanup. Final errors are still
evaluated after timing. Empty iteration CSV headers and initial/final output are
written outside the timed interval; no iteration rows or output calls execute
inside benchmark mode. These results use the same timing boundary and run count,
but naturally compare different CPU/GPU hardware and algorithms' access paths.

## Current results

| Method | Whole loop (ms) | Translation error | Rotation error | Inliers |
| --- | ---: | ---: | ---: | ---: |
| CPU serial | 300,596.867 | 24.08 cm | 0.12556° | 123,329 |
| CUDA brute-force | 253.918 | 24.08 cm | 0.12556° | 123,329 |
| CUDA tiled brute-force | 129.478 | 24.08 cm | 0.12556° | 123,329 |

All return ICP_OK and complete 20 updates, with `converged=0`. Final printed pose,
RMSE (0.142262415058 m) and counts match across targets. Output-level parity does
not prove bitwise parity of intermediate arrays. GT translation and rotation
errors remain 0.240848923628 m / 0.125556895097 degrees. No GT initialization or
tuning was used. The observed loop-time ratios relative to this serial baseline
are about 1184 (naive) and 2322 (tiled); tiled is about 1.96 times faster than naive
for this loop. These are this workload's measured ratios, not a general speedup
claim or a statistically characterized benchmark distribution.

## Commands and evidence

```sh
ICP_BENCHMARK=1 ICP_ITERATIONS=20 make cpu RUN=1
ICP_BENCHMARK=1 ICP_ITERATIONS=20 make gpu SEARCH=all RUN=1
```

`ICP_BENCHMARK=1` forces fixed iterations, skips all per-iteration diagnostic
output/timers, disables CPU point dumps and selects `result.loop_time_ms` for
`execution_time_ms`. `timing_scope=whole_loop_steps_1_to_5_no_logging` confirms
that boundary in the output. Normal diagnostic mode retains its earlier
complete-registration-call metric and iteration logs; do not mix those times
with benchmark results.

Exact commands, build logs, three runtime logs, GT comparison and hashes are in
`scripts/result/cpu-gpu-loop20-comparison/`. The analysis script verifies 20
updates, the scope marker, no iteration output, no iteration data rows and frozen
source identities. Public `scripts/result.csv` and the README plot use these new
values. No sanitizer, per-stage-array verification, point-dump execution, other
frame pair or frequency-controlled comparison was performed.

---

# Historical complete-call CPU and CUDA comparison

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
