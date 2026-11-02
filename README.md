# CUDA-accelerated SLAM ICP

```text
src_cpu/              Standalone CPU project: main, headers, ICP stages, and tests
src_cuda/             Standalone CUDA project: main, headers, GPU kernels, and tests
scripts/              Data preparation, evaluation, and benchmark tools
tests/                Python tool tests
data/kitti-sample/    Local KITTI sample pair and prepared XYZ inputs
```

Each implementation has its own `CMakeLists.txt` and `icp.hpp`, with no source or
header dependencies on the other directory. The root `CMakeLists.txt` selects one
implementation. `requirements.txt` lists Python tool dependencies.

Dataset: [KITTI sample pair](data/kitti-sample/) (local data, excluded from Git).
`velodyne/` contains the original frames; `source.xyz` and `target.xyz` contain
the prepared inputs. ICP estimates the rotation and translation from source
frame 1 to target frame 0.
