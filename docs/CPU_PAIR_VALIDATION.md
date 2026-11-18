# CPU KITTI pair validation — 2026-11-17

Release CPU build and registration completed successfully on the dirty worktree
based on a8ca470. This checks the bundled 2,048-point source frame 1 and target
frame 0. It does not establish ground-truth pose accuracy or CUDA correctness.

## Execution

The user approved an exception to run this CPU validation on `interactive`.
Build job 7486168 completed on acn08, GCC 8.5.0 and Eigen 3.3.7.
Command:

```sh
CMAKE_PREFIX_PATH=/common/software/install/manual/eigen/3.3.7 PARTITION=interactive make cpu RUN=1
```

Defaults: identity initial pose, maximum 20 iterations, 1 m correspondence cutoff,
translation/rotation convergence tolerances 1e-6. Output: ICP_OK, 20 iterations,
converged=0, 1,756 final inliers, final RMSE 0.501347938655 m.
Maximum iterations was reached; ICP_OK does not imply convergence.

Independent exhaustive NumPy nearest-neighbor evaluation completed in job 7486322
on acn01. It recomputes distances in double from float input/transformed coordinates.

| Metric | Identity pose | Final pose after 20 iterations |
|---|---:|---:|
| Inliers within 1 m | 1682 / 2048 | 1756 / 2048 |
| Inlier RMSE (m) | 0.5707264071 | 0.5013479394 |
| All-source nearest-neighbor RMSE (m) | 0.9129600745 | 0.8443720646 |
| Median nearest-neighbor distance (m) | 0.5980510361 | 0.4788589009 |

Inlier sets differ between poses. The independent final RMSE agrees with the C++
output within approximately 8e-10 m. The clouds are independently sampled real
scans; they are not expected to become pointwise identical. No GT pose is bundled.
A larger iteration limit is a reasonable diagnostic, but has not yet been run.

## Evidence and limitations

- Build log: results/build/cpu-1794937398810297304-1544590.log.
- Independent script/log: results/cpu-pair-check/evaluate_pair.py and evaluation.log.
- Metrics and pose: results/cpu-pair-check/comparison.json.
- Dirty-source, executable and input hashes: results/cpu-pair-check/provenance.json.
- Initial build job 7486118 failed to locate Eigen; its log is retained at
  results/build/cpu-1794937243260463132-1089030.log. Setting the existing Eigen
  prefix resolved it; no dependencies were downloaded.
- GPU-partition submissions were cancelled while pending. A CPU-only request to
  that partition was rejected because it requires a GPU allocation. The wrapper
  now explicitly requests one node/task and supports the approved CPU partition.
- No deleted tests were recreated. No GPU build, sanitizer, benchmark, GT pose
  validation or broader regression suite was run.

## Maximum-100-iteration comparison

User requested increasing only the iteration limit on the same bundled pair.
On 2026-11-17 job 7486869 completed on acn07 in the approved `interactive`
partition. Both defaultOptions copies now set max_iterations=100; only CPU was
built and executed. The threshold, initial pose, convergence tolerances, source
and target are unchanged. Existing early stopping remains enabled.

| Metric | Limit 20 | Limit 100 |
|---|---:|---:|
| Actual iterations | 20 | 21 |
| Converged | false | true |
| Final inliers | 1756 | 1756 |
| C++ final RMSE (m) | 0.501347938655 | 0.501347943001 |
| Independent all-source NN RMSE (m) | 0.8443720646 | 0.8443720669 |

The additional update produces no meaningful improvement. This supports a plateau
under the current correspondence and convergence rules, not a globally correct
pose or an absence of implementation bugs. Increasing the iteration budget alone
does not resolve this pair's residual error. A known-transform recovery check and
correspondence inspection remain the next diagnostic steps; neither was run here.

Execution: `srun --partition=interactive --nodes=1 --ntasks=1 --cpus-per-task=4
--mem=4G --time=00:01:00 bash results/cpu-pair-100/run.sh` (one shell command).
The script calls the existing Make build entrypoint with the Eigen prefix and
runs independent NumPy evaluation. Full output: results/cpu-pair-100/run.log;
metrics: results/cpu-pair-100/comparison.json; source/input/executable hashes:
results/cpu-pair-100/provenance.json. Original 20-iteration evidence was preserved.


## Full original frames, 2026-11-17

At the user's request, CPU job 7489138 on ahn2 evaluated every original point:
source frame 1 has 124605 points and target frame 0 has 124668. Sampling,
voxelization and range filtering were all disabled. Prepared float32 XYZ values
were checked for exact equality against the original KITTI binaries. Reflectance
is unused. The unchanged Release binary ran with identity initialization,
100-iteration limit, 1 m correspondence cutoff and 1e-6 convergence tolerances.
Vanilla queries every source point against every target point; the cutoff still
excludes distant matches from the pose update. Feature examines all points during
classification and uses selected surface/edge constraints during optimization.

| Metric | Vanilla, 2048 points | Vanilla, full cloud | PDF-flow feature, 2048 points | PDF-flow feature, full cloud |
|---|---:|---:|---:|---:|
| GT translation error (m) | 0.138137651443 | 0.199837766427 | 0.301632049097 | 0.177527568806 |
| GT rotation error (degrees) | 0.071504933703 | 0.124028484230 | 0.626876303721 | 0.213475984412 |
| Final NN inlier RMSE (m) | 0.501347943001 | 0.141579759330 | 0.511649023107 | 0.142813824567 |
| NN inliers | 1756 | 123338 | 1738 | 123339 |
| Iterations | 21 | 69 | 13 | 20 |
| Converged | yes | yes | yes | yes |

Full-cloud feature extracted 38630 surfaces/33370 edges from source and
38436 surfaces/34081 edges from target; final constraints were 38025 planes and
18620 lines, with feature RMSE 0.137397775152 m.

Full-cloud vanilla improved its NN residual but worsened both GT pose metrics
relative to the reduced input. Full-cloud feature improved both pose metrics
relative to its reduced-input result. Among full-cloud methods, feature has
smaller translation error and vanilla smaller rotation error. These observations
do not establish a general ranking or isolate the effect of sampling: the old
inputs also applied 2–60 m range filtering and 0.2 m voxelization. A denser target
changes the nearest-neighbor residual itself. One pair does not establish SLAM
trajectory accuracy, and the source of the remaining pose bias is undiagnosed.
No ground truth was used for initialization or parameter tuning.

Both processes returned ICP_OK / exit 0; Slurm job completed with exit 0:0.
Observed wall times were approximately 1141 s (vanilla) and 30 s (feature), polled
at 5 s intervals. They ran concurrently in a two-CPU allocation; these are not
controlled benchmark speedups. Runtime is dominated by different searches and
iteration counts. No source changes or GPU execution were performed.

Reproduction and evidence: `results/full-cloud-check/README.md`, `run.py`,
`evaluate.py`, `run.log`, per-method log/result/error/execution files and
`provenance.json`. The latter records binary, source, original/prepared data and
result hashes. Input/output coordinate convention and GT are the same as in
[the feature comparison](feature/README.md). The prior reduced-input evidence
remains in `results/feature-pdf-check/` and `results/kitti-gt-check/`.
