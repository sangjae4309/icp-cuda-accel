#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN="${RUN:-1}"
export SEARCH="${SEARCH:-all}"
export SOURCE="${SOURCE:-data/000001.bin}"
export TARGET="${TARGET:-data/000000.bin}"
build_system=${BUILD_SYSTEM:-auto}
case "$build_system" in
    auto)
        if command -v cmake >/dev/null 2>&1; then
            build_system=cmake
        else
            build_system=direct
        fi
        ;;
    cmake|direct) ;;
    *) echo 'BUILD_SYSTEM must be auto, cmake, or direct.' >&2; exit 1 ;;
esac
if ! cuda_compiler=$(command -v "${CUDACXX:-nvcc}"); then
    echo 'CUDA compiler not found. Add nvcc to PATH or set CUDACXX=/path/to/nvcc.' >&2
    exit 1
fi
export CUDACXX="$cuda_compiler"

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
if [[ "$build_system" == cmake ]]; then
    cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
        -DICP_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDACXX" \
        -DCMAKE_CUDA_ARCHITECTURES="$architecture"
    cmake --build build-gpu -j4
else
    eigen_include=${EIGEN3_INCLUDE_DIR:-}
    if [[ -z "$eigen_include" ]]; then
        for candidate in /usr/include/eigen3 /usr/local/include/eigen3; do
            if [[ -f "$candidate/Eigen/Core" ]]; then
                eigen_include=$candidate
                break
            fi
        done
    fi
    if [[ ! -f "$eigen_include/Eigen/Core" ]]; then
        echo 'Set EIGEN3_INCLUDE_DIR to the directory containing Eigen/Core.' >&2
        exit 1
    fi

    cuda_flags=()
    IFS=';' read -r -a architectures <<< "$architecture"
    for arch in "${architectures[@]}"; do
        if [[ ! "$arch" =~ ^[0-9]+$ ]]; then
            echo 'Direct builds require numeric CUDA_ARCHITECTURES, e.g. 60.' >&2
            exit 1
        fi
        cuda_flags+=("-gencode=arch=compute_$arch,code=[sm_$arch,compute_$arch]")
    done
    includes=(-Isrc_gpu/common -Isrc_gpu/vanilla "-I$eigen_include")
    host_sources=(
        src_gpu/common/io.cpp
        src_gpu/common/kitti_parser.cpp
        src_gpu/common/pose.cpp
        src_gpu/vanilla/icp_step4_solve.cpp
        src_gpu/vanilla/icp_step5_update.cpp
        src_gpu/vanilla/icp_log.cpp
        src_gpu/vanilla/icp_run.cpp
        src_gpu/main.cpp
    )
    cuda_sources=(
        src_gpu/vanilla/icp_utils.cu
        src_gpu/vanilla/icp_step1_transform.cu
        src_gpu/vanilla/icp_step2_correspondence.cu
        src_gpu/vanilla/icp_step2_kdtree.cu
        src_gpu/vanilla/icp_step2_voxel_grid.cu
        src_gpu/vanilla/icp_step3_statistics.cu
    )
    mkdir -p build-gpu/direct
    objects=()
    for source_file in "${host_sources[@]}"; do
        object="build-gpu/direct/$(basename "${source_file%.cpp}").o"
        "$CUDACXX" -O3 -DNDEBUG -std=c++17 -DICP_HAS_CUDA=1 \
            -Xcompiler=-ffp-contract=off,-Wall,-Wextra,-Wpedantic \
            "${includes[@]}" -c "$source_file" -o "$object"
        objects+=("$object")
    done
    for source_file in "${cuda_sources[@]}"; do
        object="build-gpu/direct/$(basename "${source_file%.cu}").o"
        "$CUDACXX" -O3 -DNDEBUG -std=c++17 -DICP_HAS_CUDA=1 \
            -fmad=false "${cuda_flags[@]}" "${includes[@]}" \
            -c "$source_file" -o "$object"
        objects+=("$object")
    done
    "$CUDACXX" "${cuda_flags[@]}" "${objects[@]}" -o build-gpu/icp_run
fi
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
