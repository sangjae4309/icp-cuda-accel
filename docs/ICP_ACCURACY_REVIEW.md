# CPU ICP accuracy review and saved-pose visualization

## Scope

Parent source review, 2026-11-17. No new vanilla registration or production
rebuild was requested. CPU vanilla's default iteration cap was changed to 20;
CPU feature keeps a separate cap of 100. The CLI transfers only backend selection
into feature options so the vanilla cap does not silently change feature behavior.
The existing production binary still reflects the previous build until rebuilt.
Historical results retain their actual iteration counts (full-cloud vanilla 69).
Twenty iterations are a chosen budget, not a demonstrated full-cloud convergence
threshold. The earlier near-equality of iterations 20 and 21 concerned reduced input.

## Observed implementation versus accuracy hypotheses

`src_cpu/vanilla/icp_gold.cpp` searches every target point with Euclidean float
squared distance, strict first-index ties, many-to-one matches and a 1 m cutoff.
This is exact nearest-neighbor search under its arithmetic contract. Finding the
nearest point does not prove it is the same physical surface location in two scans.
The centered covariance, reflection-corrected SVD and left composition reviewed
in `icp_gold.cpp`, `icp_solver.cpp` and `common/pose.cpp` are consistent with the
stated point-to-point equations; this review did not identify a sign/composition
mistake. That is not a proof that no implementation defect remains.

| Present implementation | Potential limitation; not a diagnosed cause |
|---|---|
| Identity initialization, scan-to-scan only | Can converge to a local solution; no motion prediction or local map |
| Fixed 1 m match cutoff and equal weights | Nearby but wrong/dynamic correspondences can bias the least-squares fit |
| Every full-cloud point contributes equally if matched | Dense structures dominate; more points need not add independent pose constraints |
| Raw XYZ, no per-point time motion compensation | A moving scanner can produce a non-rigidly distorted scan |
| Feature surface QR checks rank, then query-to-plane distance | It does not separately bound each neighbor's plane-fitting residual; mixed-surface neighborhoods may produce poor geometry |

These are simplifications or candidate weaknesses, not proof of why this KITTI
pair has its measured error. Point-to-point ICP and exact nearest-neighbor search
are not inherently erroneous. A KD tree changes search cost, not the selected
nearest point when arithmetic, cutoff and ties are equivalent.

The original ICP paper characterizes convergence to a local minimum, not a
promise of ground-truth pose recovery:
[Besl and McKay, 1992](https://graphics.stanford.edu/~smr/ICP/comparison/besl-mckay-icp-pami92.pdf).
KISS-ICP also uses point-to-point registration, adding adaptive correspondence
thresholds, robust weighting, motion compensation and subsampling. Thus a feature
objective is not a prerequisite for accurate odometry:
[KISS-ICP paper](https://arxiv.org/abs/2209.15397).
Its reference pipeline additionally uses motion prediction and a local map:
[author implementation](https://github.com/PRBonn/kiss-icp/blob/main/cpp/kiss_icp/pipeline/KissICP.cpp).
This is a comparison of components, not an identical-input benchmark against
KISS-ICP. No external implementation was executed.

## Images from saved full-cloud poses

![Before, vanilla, feature and ground truth](../results/full-cloud-check/visualization/alignment.png)

Blue is target frame 0; orange is source frame 1. Each original cloud was sampled
independently to 2048 display points, seed 42, with exactly the same indices in
every panel. The top row shows an XY overview; the bottom row uses identical
zoom bounds. Columns are identity, saved vanilla (69 iterations), saved feature
(20 iterations), and GT for reference. Images were generated without rerunning
registration. Plot crops do not affect registration inputs or transforms.

The close-up shows source structures moving toward target structures, including
the near-vertical group around x=10 m. Color overlap does not establish pointwise
correspondence; even GT need not perfectly overlay independently sampled scans.
Use the measured GT errors in CPU_PAIR_VALIDATION.md for quantitative assessment.

[3D before/after view](../results/full-cloud-check/visualization/alignment-3d.png).
Rendering script, sample indices and scope notes live in
`results/full-cloud-check/visualization/`. The 3D figure uses the same camera and
axis bounds before/after; use the equal-aspect XY figure to judge distances.

## Feature iteration limit

The original full-cloud feature run had a cap of 100 but stopped after 20 outer
iterations; the reduced-input PDF-flow run stopped after 13. In
`feature/optimizer.cpp`, convergence checks an undamped Gauss-Newton increment
against translation and rotation tolerances of 1e-6. On convergence it returns
without updating pose. `feature/icp.cpp` then stops unless fixed iterations are
requested. Merely raising the cap cannot bypass this early-stop rule. Forcing
outer-loop repetitions also does not bypass the solver's small-step return.
Convergence describes this selected geometry/objective, not GT accuracy.


The user authorized a feature-only larger-cap check. CPU job 7490071 on ahn2
compiled a separate CLI harness against the existing validated CPU libraries and
set the cap to 200. Full inputs, initialization and tolerances were unchanged.
It completed with exit 0:0 and again stopped at 20 iterations. The complete feature
result text (counts, residuals and pose) matched the cap-100 output at printed
precision: GT translation error 0.177527568806 m and rotation error
0.213475984412 degrees. Thus merely doubling the iteration budget did not improve
this case. No tighter convergence tolerance or altered solver was tested.
Evidence: `results/feature-iteration-check/`, including harness, logs, evaluated
result and library hashes. This did not rebuild or run the production vanilla
executable with its newly edited 20-iteration default.


A subsequent fixed-camera 3 m close-up uses 1340 source / 1259 target points
selected once from the original clouds. The same local samples are used before
and after, with no displacement magnification. The stored vanilla 69-iteration
pose is reused; no registration rerun occurred.
[Close-up PNG](../results/full-cloud-check/visualization/vanilla-closeup.png) and
[before/after toggle GIF](../results/full-cloud-check/visualization/vanilla-closeup-toggle.gif).
The GIF toggles endpoint poses, not intermediate ICP iterations. Exact bounds,
selection rules and indices accompany the visualization artifacts.
