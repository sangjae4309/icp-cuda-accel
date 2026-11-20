# GPU KD-tree correspondence: work summary

**Date:** 2026-11-20
**Branch:** `TN/KD-tree` (based on `a77350f`, "Add KD-tree with brute-force test and make kdtree-test target")
**Machine:** GPU lab `gpulab21.ece.umn.edu`, Tesla P100-PCIE-16GB (sm_60), CUDA 12.3

## Result

The CUDA KD-tree search is implemented, tested on its own, and wired into the
full ICP loop. A full ICP run with `cuda-kd-tree` produces **exactly the same**
iterations, inliers, RMSE, and final pose as `cuda-naive`, and runs about
**47× faster per iteration**.

| Search | Iterations | Inliers | RMSE | Per-iteration time (steady state) | Complete registration call |
| --- | --- | --- | --- | --- | --- |
| `cuda-naive` | 69 | 123,338 | 0.14157975933 | ~108.8 ms | 7807.6 ms |
| `cuda-kd-tree` | 69 | 123,338 | 0.14157975933 | ~2.3 ms | 386.4 ms |
| Speed-up | | | | **~47×** | **~20×** |

- Frames: `data/000001.bin` (source, 124,605 points) → `data/000000.bin` (target, 124,668 points).
- Tree: 124,668 nodes, depth 17. Building and uploading it costs **53.0 ms once**, outside the timed iteration loop.
- `diff` of the `iterations=` line and the 4×4 `T_target_source` matrix printed by both runs: **IDENTICAL**.
- Per-iteration times are measured with CUDA events and cover the pose upload, steps 1–3, and the statistics download. The "complete registration call" time is wall-clock for all of `runICP`, including setup, the tree build and upload, and logging. This is why that speed-up is smaller.
- These are single runs. The matched benchmark (`ICP_BENCHMARK=1 ICP_ITERATIONS=20`, including `cuda-tiled`) has not been run yet.

## What was built

### 1. CUDA KD-tree search: `src_gpu/vanilla/icp_step2_kdtree.cu`

The file replaced the `// FILL ME` stub with:

| Function | Purpose |
| --- | --- |
| `findCorrespondencesKdTreeKernel` | One thread per transformed source point. Line-for-line port of `KdTree::nearest()`: iterative traversal with a fixed 64-entry stack, the same tie rule (smallest original index), and the same exact pruning (`diff2 <= best`). Writes the **original** target index, which step 3 needs. |
| `uploadKdTreeCUDA` | Copies the host tree's three arrays (points, original indices, split axes) to the GPU. Rejects a size mismatch with `num_target` or a depth above 64, and frees everything on failure. |
| `freeKdTreeCUDA` | Frees the three arrays and resets the pointers. Safe to call more than once. |
| `findCorrespondencesKdTreeCUDA` | Launcher called by the existing step-2 dispatcher for `SEARCH_KD_TREE`. Same validation and grid as the brute-force launches. |

It keeps its own copy of `squaredDistance` (marked "keep textually in sync"),
because device functions can't be shared across `.cu` files without `-rdc`.

### 2. Wiring into ICP

| File | Change |
| --- | --- |
| `src_gpu/vanilla/icp_utils.h` | Added `kd_points_d`, `kd_indices_d`, `kd_axes_d`, `kd_count` to `CudaIcpBuffers`. Declared `uploadKdTreeCUDA` and `freeKdTreeCUDA`. Added `#include <cstdint>` and `class KdTree;`. |
| `src_gpu/vanilla/icp_utils.cu` | `freeIcpBuffers()` calls `freeKdTreeCUDA(w)` before `*w = {};`. |
| `src_gpu/vanilla/icp_run.cpp` | For `SEARCH_KD_TREE`, builds the tree from the target and uploads it right after `uploadPointClouds`, outside the timed loop, and prints `kd_tree nodes=… depth=… build_upload_ms=…`. Added `#include "icp_kdtree.h"`. |
| `src_gpu/CMakeLists.txt` | Added `vanilla/icp_kdtree_build.cpp` to `icp_core`, in the C++ list so it gets `-ffp-contract=off`. |
| `tests/run_gpu.sh` | Added `src_gpu/vanilla/icp_kdtree_build.cpp` to `host_sources`. This is the direct-`nvcc` path the lab uses, because it has no `cmake`. |

### 3. Isolated step-2 GPU test

| File | Change |
| --- | --- |
| `src_gpu/vanilla/icp_kdtree_gpu_test.cpp` | New. Runs only step 2: queries are copied straight into `transformed_d`. Compares the CUDA KD-tree against the CUDA naive kernel and the host tree, bit for bit. Outputs are filled with garbage before each run, and the test synchronizes after every kernel. |
| `Makefile` | New `kdtree-gpu-test` target with `KDGPU_ARGS`, `CUDA_ARCH`, and `NVCC` overrides, plus a help line. |

Step-2 test results (`make kdtree-gpu-test`):

| Case | Mismatches vs naive / host | KD-tree | Naive | Speed-up |
| --- | --- | --- | --- | --- |
| Synthetic grid, half-step ties | 0 / 0 | 0.118 ms | 0.068 ms | 0.6× |
| 1000 identical points | 0 / 0 | 0.645 ms | 0.088 ms | 0.1× |
| Single point | 0 / 0 | 0.007 ms | 0.007 ms | 1.0× |
| Partial block (n = 257) | 0 / 0 | 0.066 ms | 0.062 ms | 0.9× |
| KITTI source → target | 0 / 0 | 2.448 ms | 109.443 ms | 44.7× |
| KITTI moved source (2°, 0.5 m) → target | 0 / 0 | 2.422 ms | 108.590 ms | 44.8× |
| KITTI target → target | 0 / 0 | 0.208 ms | 108.591 ms | 520.9× (best case) |

The synthetic cases check correctness, not speed. They are tiny, and "1000
identical points" is the KD-tree's worst case, where nothing can be pruned.

### 4. Documentation

`docs/KDTREE.md` was rewritten to be concise, covering how the tree works,
the files, the API, and both tests (`make kdtree-test` and
`make kdtree-gpu-test`) with expected output. Its status table still says full
ICP is "not wired in yet". Update it with the results above.

## Problems hit and how they were fixed

| Problem | Cause | Fix |
| --- | --- | --- |
| `ssh: Could not resolve hostname ece-gpulab21` from Ubuntu | Short name only resolves on the Mac | Use the full name `gpulab21.ece.umn.edu`. `DEVELOPMENT_WORKFLOW.md` still says `ece-gpulab21`. |
| `expected unqualified-id before '-' token` at a line reading `EOF` | Heredoc text, including its `EOF` terminator, was pasted into an editor. The test code also overwrote `icp_kdtree_build.cpp`. | Restored `icp_kdtree_build.cpp` from `TN/KD-tree` and deleted the stray `EOF` line. |
| `set: pipefail: invalid option name` | Windows CRLF line endings in `tests/run_gpu.sh` and other files | `grep -rlIP '\r' Makefile tests src_gpu src_cpu \| xargs -r sed -i 's/\r$//'` |
| `Illegal variable name.` / `Ambiguous output redirect.` | The lab's login shell is **tcsh**, not bash | Use tcsh syntax (`\|&` instead of `2>&1`, `env VAR=1 cmd`, no `$'…'`), or run `bash` first. |
| `undefined reference to KdTree::build` | `icp_kdtree_build.cpp` not in the build lists | Steps 3–4 above (CMake and `run_gpu.sh`). |

## How to reproduce (GPU lab, tcsh)

```
cd /home/class/nguy5505/EE5351/icp-cuda-accel

# Step 2 only
make kdtree-gpu-test KDGPU_ARGS=--synthetic-only
make kdtree-gpu-test

# Full ICP
make gpu
./build-gpu/icp_run data/000001.bin data/000000.bin cuda-kd-tree | tee /tmp/kd.txt
./build-gpu/icp_run data/000001.bin data/000000.bin cuda-naive   | tee /tmp/naive.txt
grep -A4 '^iterations=' /tmp/kd.txt    > /tmp/kd_result.txt
grep -A4 '^iterations=' /tmp/naive.txt > /tmp/naive_result.txt
diff /tmp/kd_result.txt /tmp/naive_result.txt && echo "IDENTICAL"
```

## Next steps

1. **Matched benchmark.** Repeat each several times:
   ```
   env ICP_BENCHMARK=1 ICP_ITERATIONS=20 ./build-gpu/icp_run data/000001.bin data/000000.bin cuda-kd-tree
   env ICP_BENCHMARK=1 ICP_ITERATIONS=20 ./build-gpu/icp_run data/000001.bin data/000000.bin cuda-naive
   env ICP_BENCHMARK=1 ICP_ITERATIONS=20 ./build-gpu/icp_run data/000001.bin data/000000.bin cuda-tiled
   ```
   Report `build_upload_ms` separately as a one-time cost.
2. **Memory check:** `compute-sanitizer ./build-kdtree-gpu/icp_kdtree_gpu_test --synthetic-only`.
3. **Other frame pairs:** for example `SOURCE=data/000002.bin TARGET=data/000001.bin`, to confirm results stay identical.
4. **Commit.** The edits were made on the GPU lab, which can't reach GitHub. Copy these files back to the local checkout, review `git diff`, and commit to `TN/KD-tree`:
   `Makefile`, `src_gpu/CMakeLists.txt`, `tests/run_gpu.sh`, `src_gpu/vanilla/{icp_step2_kdtree.cu, icp_utils.h, icp_utils.cu, icp_run.cpp, icp_kdtree_gpu_test.cpp}`, and `docs/KDTREE.md`.
   Before syncing again, set `git config core.autocrlf input` locally so CRLF line endings don't come back.
5. **Housekeeping:**
   - In the `Makefile` help text, change `make gpu SEARCH=kd-tree RUN=1 FILL ME (not implemented)` to a real description.
   - Consider adding `kd-tree` to `SEARCH=all` in `run_gpu.sh`.
   - Update the hostname in `DEVELOPMENT_WORKFLOW.md`.
   - Update the status in `docs/KDTREE.md`.