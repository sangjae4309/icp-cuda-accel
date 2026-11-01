# CUDA-accelerated SLAM ICP

```text
src_cpu/              Standalone CPU project: main, headers, ICP stages, and tests
src_cuda/             Standalone CUDA project: main, headers, GPU kernels, and tests
scripts/              Data preparation, evaluation, and benchmark tools
tests/                Python tool tests
data/synthetic_small/ Small synthetic pair with a ground-truth transform
```

Each implementation has its own `CMakeLists.txt` and `icp.hpp`, with no source or
header dependencies on the other directory. The root `CMakeLists.txt` selects one
implementation. `requirements.txt` lists Python tool dependencies.
