# KD-tree correspondence search

The KD-tree replaces brute force in ICP step 2 (nearest target point for every
transformed source point). Brute force costs $O(NM)$ per iteration, about
$1.55 \times 10^{10}$ distance evaluations for one KITTI frame pair. The tree is
built once over the fixed target cloud, and each query then costs about
$O(\log M)$.

**Status**

| Part | State |
| --- | --- |
| Host tree build + search (`KdTree`) | Done, matches brute force bit for bit |
| CUDA search kernel (`findCorrespondencesKdTreeCUDA`) | Done, matches the naive CUDA kernel bit for bit on the GPU lab |
| Full ICP run (`make gpu SEARCH=kd-tree RUN=1`) | Not wired in yet (see [Next steps](#next-steps)) |

## How it works

**Build (host, once per ICP run).** Pick the widest axis of the current range's
bounding box, put the median point on that axis at the node, and recurse into
both halves. The tree stays balanced: depth $\lceil \log_2(M+1) \rceil$, which
is 17 for a 124,668-point frame.

**Layout.** The tree is stored as three flat arrays with no pointers, so it
copies to the GPU with three `cudaMemcpy` calls:

| Array | Contents |
| --- | --- |
| `nodePoints()[k]` | Point at node `k` |
| `nodeIndices()[k]` | That point's index in the **original** target cloud |
| `nodeAxes()[k]` | Split axis: 0 = x, 1 = y, 2 = z |

The node for index range `[lo, hi)` is at `mid = lo + (hi - lo) / 2`, with
children `[lo, mid)` and `[mid + 1, hi)`.

**Search.** Walk down the side of each split plane the query is on, and push the
other side onto a stack together with its lower bound $(q_a - p_a)^2$. Skip any
pushed range whose bound is greater than the best distance found so far. The
search is iterative with a fixed 64-entry stack, so the same code runs in
`KdTree::nearest()` on the CPU and in the CUDA kernel (one thread per source
point).

**Same answers as brute force.** Results match brute force exactly, bit for bit:

- the same float `squaredDistance` formula, with the query as the first operand;
- on an exact distance tie, the **smallest original target index** wins;
- a range is skipped only when its bound is strictly greater than the best
  distance, so points at exactly the best distance are always checked. Keep
  `diff2 <= best`; do not change it to `<`;
- `match_dist2` always holds the nearest distance, and `match_index = -1` when
  that distance is greater than the cutoff.

This depends on the `-ffp-contract=off` (C++) and `-fmad=false` (CUDA) build
flags.

## Files

| File | Contents |
| --- | --- |
| `src_gpu/vanilla/icp_kdtree.h` | `KdTree` class and layout |
| `src_gpu/vanilla/icp_kdtree_build.cpp` | Tree build, host search, invariant check. **No `main()`.** |
| `src_gpu/vanilla/icp_step2_kdtree.cu` | CUDA kernel, `uploadKdTreeCUDA`, `freeKdTreeCUDA`, launcher |
| `src_gpu/vanilla/icp_utils.h` | `kd_points_d`, `kd_indices_d`, `kd_axes_d`, `kd_count` in `CudaIcpBuffers` |
| `src_gpu/vanilla/icp_kdtree_test.cpp` | CPU test (tree vs brute force) |
| `src_gpu/vanilla/icp_kdtree_gpu_test.cpp` | GPU test (CUDA tree vs CUDA naive vs host tree) |
| `Makefile` | `kdtree-test` and `kdtree-gpu-test` targets |

## Usage

```cpp
KdTree tree;
tree.build(&target);                       // host, once
uploadKdTreeCUDA(&buffers, &tree);         // after allocateIcpBuffers; tree can be freed after this

findCorrespondencesCUDA(&buffers, SEARCH_KD_TREE, max_distance2);  // every iteration

freeKdTreeCUDA(&buffers);                  // before freeIcpBuffers
```

`uploadKdTreeCUDA` rejects a tree whose size differs from `num_target` or whose
depth exceeds 64.

## Testing

Run both tests from the repository root. Each test passes only if the index
**and** the exact bits of the squared distance match the reference for every
query. Both exit with status 0 on success and 1 on failure, and print up to five
mismatches for any failing case.

### 1. Tree build, CPU: `make kdtree-test`

Needs only a C++17 compiler, so it runs on any machine, including WSL, macOS,
and the GPU lab.

```bash
make kdtree-test                       # check every 10th query against brute force
make kdtree-test STRIDE=1              # check every query (about 10 min)
make kdtree-test SOURCE=data/000003.bin TARGET=data/000002.bin
```

It checks:

- the tree itself: split invariant at every node, indices are a permutation,
  and stored points are bit-identical to the input;
- synthetic clouds with duplicates and exact ties, 1000 identical points, a
  single point, and NaN/null input being rejected;
- KITTI source → target (cutoff 1 m) and target → target.

Expected output:

```text
[PASS] synthetic grid, cutoff on half-step ties: 5083/5083 checked, 1403 within cutoff, 0 mismatches | ...
[PASS] 1000 identical points: 2/2 checked, 1 within cutoff, 0 mismatches | ...
[PASS] single point: 2/2 checked, 1 within cutoff, 0 mismatches | ...
[PASS] NaN / null input rejected
source data/000001.bin: 124605 points, target data/000000.bin: 124668 points
[PASS] built KD-tree: 124668 nodes, depth 17, 25.4 ms; split invariant ok, node points match source ok
[PASS] KITTI source->target, cutoff 1 m: 12461/124605 checked, 12341 within cutoff, 0 mismatches | ...
[PASS] KITTI target->target self-match: 12467/124668 checked, 12467 within cutoff, 0 mismatches | ...
ALL PASSED
```

### 2. Point correspondence, GPU: `make kdtree-gpu-test`

Needs `nvcc` and an NVIDIA GPU, so run it on the GPU lab. This test runs **only
step 2**: the query points are copied straight into `transformed_d`, so step 1
and the rest of the ICP loop are not involved.

```bash
make kdtree-gpu-test KDGPU_ARGS=--synthetic-only   # small cases, a few seconds
make kdtree-gpu-test                               # + full KITTI frames
make kdtree-gpu-test CUDA_ARCH=60                  # if nvidia-smi can't report the arch
```

For each case it builds the tree on the host, uploads it, runs the CUDA KD-tree
kernel and the CUDA naive kernel on the same queries, and compares the KD-tree
results against both the naive kernel and the host tree. Before each run the
outputs are filled with garbage values, so a thread that never writes is caught.
It also synchronizes after each kernel, so faults inside the kernel are
reported.

The cases are the synthetic clouds, a query count that leaves the last block
partly filled (`n = 257`), and three KITTI cases: source → target, source moved
by 2° yaw and 0.5 m, and target → target.

Memory check:

```bash
compute-sanitizer ./build-kdtree-gpu/icp_kdtree_gpu_test --synthetic-only   # expect: ERROR SUMMARY: 0 errors
```

Result on the GPU lab (Tesla P100, sm_60), frames 1 → 0, single run per case:

```text
[PASS] synthetic grid, half-step ties: n=5083 m=630 depth=10, 1403 within cutoff | mismatches vs naive 0, vs host tree 0 | kd 0.118 ms, naive 0.068 ms (0.6x)
[PASS] 1000 identical points: n=3 m=1000 depth=10, 1 within cutoff | mismatches vs naive 0, vs host tree 0 | kd 0.645 ms, naive 0.088 ms (0.1x)
[PASS] single point: n=3 m=1 depth=1, 1 within cutoff | mismatches vs naive 0, vs host tree 0 | kd 0.007 ms, naive 0.007 ms (1.0x)
[PASS] partial block (n=257): n=257 m=630 depth=10, 55 within cutoff | mismatches vs naive 0, vs host tree 0 | kd 0.066 ms, naive 0.062 ms (0.9x)
source data/000001.bin: 124605 points, target data/000000.bin: 124668 points
[PASS] KITTI source->target, cutoff 1 m: n=124605 m=124668 depth=17, 123466 within cutoff | mismatches vs naive 0, vs host tree 0 | kd 2.448 ms, naive 109.443 ms (44.7x)
[PASS] KITTI moved source->target, cutoff 1 m: n=124605 m=124668 depth=17, 121507 within cutoff | mismatches vs naive 0, vs host tree 0 | kd 2.422 ms, naive 108.590 ms (44.8x)
[PASS] KITTI target->target self-match: n=124668 m=124668 depth=17, 124668 within cutoff | mismatches vs naive 0, vs host tree 0 | kd 0.208 ms, naive 108.591 ms (520.9x)
ALL PASSED
```

How to read the timings:

- **About 45×** on real frames is the representative speed-up.
- The self-match case is a best case: every query finds a point at distance 0
  right away, so nearly everything is pruned.
- The synthetic cases are too small to measure speed. "1000 identical points"
  is the tree's worst case, because every split plane passes through the query
  and nothing can be pruned.
- These are single runs. For reported numbers, repeat the runs and also compare
  against the tiled kernel.