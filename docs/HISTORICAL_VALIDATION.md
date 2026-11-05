# Historical validation before directory correction

These commands describe the former scaffold directory. See DRAFT_VALIDATION.md for the current layout.

# Scaffold validation — 2026-11-04

Changes are additions under `scaffold/` and `docs/` only. Existing implementation
and root build files were not modified. Scope follows the revised request: no
`--output`, dry-run CLI, timing/reporting feature or ICP math implementation.

## Executed environment

Slurm job 5794545, partition `interactive-gpu`, one GPU, four CPUs, 4 GB memory,
one-minute limit. NVIDIA A40, architecture 86, driver 580.196.07; nvcc 12.1.105
(toolkit installed at `/common/software/install/manual/cuda/12.1.1`), GCC 8.5.0.
The CUDA 13.0 value shown by nvidia-smi is driver capability, not the compiler used.
Raw local log: `results/scaffold-check-5794545.log` (ignored generated artifact).

## Executed commands

Submitted with `sbatch results/check-scaffold.sh`. Inside the allocation:

```sh
nvidia-smi
export PATH=/common/software/install/manual/cuda/12.1.1/bin:$PATH
nvcc --version
c++ --version
cmake -S scaffold -B build-scaffold-cpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=OFF
cmake --build build-scaffold-cpu -j4
ctest --test-dir build-scaffold-cpu --output-on-failure
python3 -m unittest discover -s tests -p 'test_tools.py' -v
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
cmake -S scaffold -B build-scaffold-gpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER=/common/software/install/manual/cuda/12.1.1/bin/nvcc
cmake --build build-scaffold-gpu -j4
ctest --test-dir build-scaffold-gpu --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build-scaffold-gpu/test_scaffold --gpu
compute-sanitizer --tool synccheck --error-exitcode 1 ./build-scaffold-gpu/test_scaffold --gpu
```

## Results

- CPU Release build passed; CTest 1/1 passed (0.01 s).
- Existing Python tool tests: 5 passed (0.118 s).
- CUDA Release build passed; CTest 2/2 passed (0.44 s), with no skipped GPU test.
- GPU copy checks use N=257, M=3, two source blocks; source, target and pose
  round trips and small statistics download passed.
- Both GPU backends propagate `ICP_NOT_IMPLEMENTED` and retain invalid result state.
- Host checks cover XYZ rows/comments/extra columns, malformed/empty/nonfinite
  input, allocation/free, repeated cleanup, invalid backend, unavailable CUDA
  in CPU-only builds and TODO propagation.
- Memcheck: `ERROR SUMMARY: 0 errors`.
- Synccheck: `ERROR SUMMARY: 0 errors`.

## Not tested / deliberately unimplemented

No transform, correspondence, reduction, SVD, composition, convergence or final
pose algorithm is implemented or validated. Empty kernel bodies are never launched;
synccheck does not validate future tiled barriers or reductions. No registration,
synthetic GT pose comparison or benchmark was performed because the scaffold
intentionally stops before its first calculation. No speedup claim is made.
Existing CPU/CUDA implementation binaries were not rebuilt as they were unchanged.
CLI execution against a file pair was not separately exercised; loading and runICP
were exercised through infrastructure tests. Allocation exhaustion and CUDA error
injection were not tested.

The repository instruction references docs/MATH_KO.md, docs/CUDA_MAPPING_KO.md,
docs/BENCHMARK_KO.md, docs/ENVIRONMENT_KO.md and VALIDATION.md; these files were absent.
No PDF was available. Algorithm contracts use the user's supplied specification.
