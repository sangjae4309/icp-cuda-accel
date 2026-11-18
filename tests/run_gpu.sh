#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export RUN="${RUN:-1}"
export SEARCH="${SEARCH:-all}"
export SOURCE="${SOURCE:-data/000001.bin}"
export TARGET="${TARGET:-data/000000.bin}"
export CUDACXX="${CUDACXX:-$(command -v nvcc)}"

if [[ "$SEARCH" != all && "$SEARCH" != brute-force && "$SEARCH" != tiled && "$SEARCH" != kd-tree && "$SEARCH" != voxel-grid ]]; then
    echo 'SEARCH must be all, brute-force, tiled, kd-tree, or voxel-grid.' >&2
    exit 1
fi

if [[ "${ICP_DUMP:-0}" == 1 ]]; then
    echo 'GPU records iteration time with CUDA events; point dumping is CPU-only.'
fi

mkdir -p scripts/result
log="scripts/result/gpu-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee "$log") 2>&1
set -x
nvidia-smi
"$CUDACXX" --version
architecture=${CUDA_ARCHITECTURES:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')}
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_ARCHITECTURES="$architecture"
cmake --build build-gpu -j4
if [[ "$RUN" == 1 ]]; then
    if [[ "$SEARCH" == all || "$SEARCH" == brute-force ]]; then
        ./build-gpu/icp_run "$SOURCE" "$TARGET" cuda-naive
    fi
    if [[ "$SEARCH" == all || "$SEARCH" == tiled ]]; then
        ./build-gpu/icp_run "$SOURCE" "$TARGET" cuda-tiled
    fi
    if [[ "$SEARCH" == kd-tree ]]; then
        ./build-gpu/icp_run "$SOURCE" "$TARGET" cuda-kd-tree
    fi
    if [[ "$SEARCH" == voxel-grid ]]; then
        ./build-gpu/icp_run "$SOURCE" "$TARGET" cuda-voxel-grid
    fi
fi
