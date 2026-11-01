#!/usr/bin/env bash
set -eu
mkdir -p results
{
  date -u
  uname -a
  command -v lscpu >/dev/null && lscpu
  command -v nvidia-smi >/dev/null && nvidia-smi
  command -v nvcc >/dev/null && nvcc --version
  cmake --version
  c++ --version
  python3 --version
} > results/environment.txt 2>&1
cat results/environment.txt
