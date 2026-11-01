# CUDA-accelerated SLAM ICP

```text
src_cpu/              CPU ICP loop and computation stages
src_cuda/             CUDA ICP loop and GPU kernels
src_common/           Shared command-line entry point
include/icp/          Shared types and function declarations
scripts/              Data preparation, evaluation, and benchmark tools
tests/                C++ and Python tests
data/synthetic_small/ Small synthetic pair with a ground-truth transform
build-cpu/            CPU build output (ignored by Git)
build-gpu/            CUDA build output (ignored by Git)
results/              Execution and validation output (ignored by Git)
```

`CMakeLists.txt`: C++/CUDA build configuration. `requirements.txt`: Python dependencies.
