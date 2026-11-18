#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export RUN="${RUN:-1}"
export METHOD="${METHOD:-vanilla}"
export SOURCE="${SOURCE:-data/000001.bin}"
export TARGET="${TARGET:-data/000000.bin}"


export ICP_DUMP="${ICP_DUMP:-0}"

mkdir -p scripts/result
log="scripts/result/cpu-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee "$log") 2>&1
set -x
c++ --version
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DICP_ENABLE_CUDA=OFF
cmake --build build-cpu -j4
if [[ "$RUN" == 1 ]]; then
    ./build-cpu/icp_run "$SOURCE" "$TARGET" "$METHOD"
fi
