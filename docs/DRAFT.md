# ICP draft code

`../src_cpu/` and `../src_gpu/` contain independent procedural draft code. Previous implementations are retained in Git history at `d46bb9486a361bebdfdb6ab7ad84d9a561bf9b1b`; no reference directory is needed. Each draft directory builds independently.

Implemented: POD data structures, malloc/free host ownership, cudaMalloc/cudaFree
ownership, explicit H2D point/pose transfers, small D2H statistics transfer,
XYZ loading, input checks, identity initialization, cleanup and execution flow.
Algorithm functions are TODO and return `ICP_NOT_IMPLEMENTED`. Empty CUDA kernels
are not launched. CPU-only CUDA entry points return `ICP_UNAVAILABLE`, with no fallback.
Normal execution stops at the first unimplemented stage. `pose_valid` remains false;
no pose, RMSE, speedup or convergence is reported as a registration result.

Following the revised scope, there is no dry-run mode, output writer, initial-pose
file parser or extended command-line interface. Infrastructure checks live in tests.
The options struct provides initial pose and iteration controls for future callers.

## Files and boundaries

- `icp.h`: shared structs and function contracts; no CUDA headers.
- `main.cpp`, `io.cpp`: caller-owned input clouds, minimal arguments, XYZ reader.
- `icp_run.cpp`: orchestration and lifetime of workspaces.
- `icp_gold.cpp`: host workspace allocation and CPU stage placeholders.
- `icp_solver.cpp`: CPU SVD, composition and convergence placeholders.
- `cuda_memory.cu`: device ownership and copies.
- `transform_kernel.cu`, `correspondence_kernel.cu`, `statistics_kernel.cu`: stage wrappers and kernel placeholders.
- `cuda_unavailable.cpp`: explicit CPU-only stubs in src_cpu.
- `tests/test_draft.cpp`: infrastructure tests; no algorithm correctness claims.

Coordinates are float; poses, sums, SSE and covariance are double. Matrices use
row-major storage and column-vector equations. `N` and `M` are independent.
All owners must be zero-initialized; allocate only empty owners. Failed partial
allocations are freed. Deallocation accepts empty owners and clears their fields.
Allocation does not initialize algorithm outputs; TODO wrappers never consume them.

XYZ accepts three finite coordinates per data row; additional columns are ignored.
Blank lines and leading-whitespace `#` comments are allowed. Empty/malformed inputs
are rejected. Input ownership transfers to the caller only on successful load.

## Build and run

Run these commands in a Slurm `interactive-gpu` allocation (one minute for small checks).

```sh
cmake -S . -B build-draft-cpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=OFF
cmake --build build-draft-cpu -j4
ctest --test-dir build-draft-cpu --output-on-failure
./build-draft-cpu/icp_run data/kitti-sample/source.xyz data/kitti-sample/target.xyz cpu
```

The executable intentionally returns nonzero at `ICP_NOT_IMPLEMENTED`.

Inspect `nvidia-smi`, `nvcc --version` and host compiler first. For an A40 with CUDA 12:

```sh
cmake -S . -B build-draft-gpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-draft-gpu -j4
ctest --test-dir build-draft-gpu --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build-draft-gpu/icp_tests --gpu
compute-sanitizer --tool synccheck --error-exitcode 1 ./build-draft-gpu/icp_tests --gpu
```

Choose architecture for the actual GPU: P100=60, GTX1080=61, V100=70, A40=86,
L40S=89. Pre-Turing devices require compatible CUDA 11/12, not CUDA 13+.
A skipped CUDA test is not a pass. Synccheck currently exercises infrastructure;
empty algorithm kernels are not launched and their future synchronization is untested.

For independent builds, use `-S src_cpu` or `-S src_gpu` with fresh build
directories. The GPU executable accepts `cuda-naive` or `cuda-tiled` as its
third positional argument; the default backend is CPU in src_cpu and CUDA naive in src_gpu.
