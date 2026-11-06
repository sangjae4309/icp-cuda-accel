# Draft implementation tasks

Paths refer to `src_cpu/` and `src_gpu/`. Assignees are intentionally unassigned. Infrastructure
completion does not imply ICP correctness. See DRAFT_VALIDATION.md for evidence.

| Task ID | Priority | Stage | Function/File | Input | Output | Owner Role | Assignee | Dependencies | Definition of Done | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| INF-1 | P0 | Infrastructure | io.cpp, icp.h, main.cpp | XYZ / options | Validated host clouds | Integration / Infrastructure | TBD | None | Load/reject inputs, explicit ownership, no result claims | Implemented |
| INF-2 | P0 | Memory | icp_gold.cpp, cuda_memory.cu | N, M / host clouds | Workspaces / device copies | Integration / Infrastructure | TBD | INF-1 | Allocation/free and round-trip tests pass | Implemented |
| FLOW-1 | P0 | Integration | icp_run.cpp | Clouds/options | Status/result | Integration / Infrastructure | TBD | INF-2 | Stage order, cleanup and TODO propagation | Draft implemented |
| TR-1 | P0 | Transform | transformPointsCPU/CUDA | Original source / current pose | Transformed source | Transform | TBD | INF-2 | CPU/GPU agree without source mutation | Implemented |
| NN-1 | P0 | Correspondence | findCorrespondencesCPU, naive kernel | Transformed source / target / cutoff | Indices / squared distances | Correspondence | TBD | TR-1 | Brute-force parity, tie/threshold/many-to-one tests | Implemented |
| NN-2 | P1 | Correspondence | tiled kernel | Same as NN-1 | Same as NN-1 | Correspondence | TBD | NN-1 | Partial tiles, all lanes reach both barriers, parity | Implemented |
| ST-1 | P0 | Statistics | computeStatisticsCPU/CUDA, statistics_kernel.cu | Valid matched pairs | Count, centroids, SSE, H | Statistics / Reduction | TBD | NN-1 | Two-pass double reductions, duplicate targets counted | TODO |
| SO-1 | P0 | Solver | solveRigidUpdateCPU | Centroids / H | Incremental delta | Solver / Pose Update | TBD | ST-1 | SVD choice, reflection, finite and degeneracy checks | TODO |
| SO-2 | P0 | Pose | composePoseCPU, checkConvergenceCPU | Delta / current / tolerances | Next / convergence | Solver / Pose Update | TBD | SO-1 | Left composition and documented tolerance semantics | TODO |
| VAL-1 | P0 | Final evaluation | icp_run.cpp | Final pose / original clouds | Valid result / final RMSE | Validation / Benchmark | TBD | SO-2 | Recompute final matches/residuals before publishing | TODO |
| VAL-2 | P0 | Correctness | tests/ | Authored fixtures | CPU/GPU evidence | Validation / Benchmark | TBD | TR-1, NN-1, ST-1, SO-2 | All future tests below run, no skipped GPU pass | TODO |
| BEN-1 | P2 | Benchmark | Future benchmark harness | Identical workloads | Measured timings | Validation / Benchmark | TBD | VAL-2 | Fixed iterations, CLOCK_MONOTONIC / CUDA events | TODO |
| ADV-1 | Optional | Search | Future KD-tree | Points | Correspondences | Advanced Research | TBD | VAL-2 | Define/validate search contract first | Deferred |
| ADV-2 | Optional | Search | Future voxel search | Points | Correspondences | Advanced Research | TBD | VAL-2 | Define/validate search contract first | Deferred |
| ADV-3 | Optional | Features | Future preprocessing | Local kNN / covariance | Eigenvalue classes | Advanced Research | TBD | VAL-2 | Study thresholds before implementation | Research only |
| ADV-4 | Optional | Feature ICP | Future advanced path | Surface / edge features | Proposed optimizer | Advanced Research | TBD | ADV-3 | Resolve missing equations and validate design | Research only |

## Future correctness tests

- Identity, known translation, known rotation and known rigid transform.
- N != M and N not divisible by BLOCK_SIZE (allocation tests alone are insufficient).
- Exact-distance tie-breaking, many-to-one matches and threshold boundary.
- Zero valid matches and very few valid matches.
- CPU vs CUDA correspondence and statistics comparisons.
- Reflection case and rank-deficient geometry.
- Full ICP pose comparison and final-pose re-evaluation.

## Optional research scope

User-provided proposal description: local kNN -> local covariance -> eigenvalues
-> surface/edge/corner classification. Surface: lambda1 approximately lambda2 >>
lambda3; edge: lambda1 >> lambda2 approximately lambda3; corner: lambda1
approximately lambda2 approximately lambda3. Numerical thresholds are unspecified.

Advanced surface path: transform -> KD-tree KNN -> QR -> back substitution.
Advanced edge path: transform -> KD-tree KNN -> eigen decomposition. A proposed
Levenberg-Marquardt optimization follows. This needs further study; do not invent
missing equations or implement it in this scaffold. No proposal PDF was present
for independent inspection. Full SLAM is outside project scope.
