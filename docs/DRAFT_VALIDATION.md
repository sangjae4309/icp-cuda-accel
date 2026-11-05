# Draft validation after directory correction

The draft now lives in `src_cpu/` and `src_gpu/`; `scaffold/` was removed.
Original tracked files from `src_cpu/` and `src_cuda/` were moved to
`reference/src_cpu/` and `reference/src_cuda/`. Each preserved file was compared
byte-for-byte with its Git HEAD content, with no differences.
Root CMake now selects the CPU or GPU draft. README and draft documentation reflect
these paths. Both draft projects also have independent CMake entry points.

Slurm job 5795221 ran on `interactive-gpu` with one GPU, four CPUs, 4 GB memory and
a one-minute limit. Raw local log: `results/draft-check-5795221.log`.

Executed commands (from repository root inside the allocation):

```sh
nvidia-smi
export PATH=/common/software/install/manual/cuda/12.1.1/bin:$PATH
nvcc --version
c++ --version
cmake -S . -B build-draft-cpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=OFF
cmake --build build-draft-cpu -j4
ctest --test-dir build-draft-cpu --output-on-failure
python3 -m unittest discover -s tests -p 'test_tools.py' -v
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
cmake -S . -B build-draft-gpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER=/common/software/install/manual/cuda/12.1.1/bin/nvcc
cmake --build build-draft-gpu -j4
ctest --test-dir build-draft-gpu --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build-draft-gpu/icp_tests --gpu
compute-sanitizer --tool synccheck --error-exitcode 1 ./build-draft-gpu/icp_tests --gpu
```

Both Release builds passed. CPU CTest: 1/1 passed. GPU CTest: 2/2 passed, no skips.
Existing Python tests: 5 passed. CUDA memcheck and synccheck: zero errors.
These tests cover infrastructure, copies, invalid inputs and NOT_IMPLEMENTED
propagation. No algorithm kernels were launched, and no ICP accuracy, final-pose
comparison, convergence or speedup was measured. Standalone source-directory builds
were not separately repeated after the move; root builds exercise their CMake files.
Preserved reference projects were not rebuilt. See HISTORICAL_VALIDATION.md for the
original infrastructure validation before this directory correction.

## Reference cleanup

Before upload, the reference copies were verified against commit
`d46bb9486a361bebdfdb6ab7ad84d9a561bf9b1b` again and removed at the user's request.
That commit is also present on origin/main. Removing reference-only files and
editing contribution documentation do not change the tested draft binaries.
