#!/usr/bin/env bash
set -e

echo "=== Linux Wheel Build Start $(date) ===" > /home/ussoewwin/SageAttention/build_linux.log

export PATH="/home/ussoewwin/venv314/bin:/usr/local/cuda-13.2/bin:/usr/bin:/bin:/usr/local/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME="/usr/local/cuda-13.2"
export CUDA_PATH="/usr/local/cuda-13.2"
export LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-13.2/lib64/stubs:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-13.2/lib64/stubs:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0;12.1"
export NVCC_APPEND_FLAGS="--threads 2 --expt-relaxed-constexpr -diag-suppress=20013 -diag-suppress=20015"
export MAX_JOBS=4

cd /home/ussoewwin/SageAttention

rm -rf dist

/home/ussoewwin/venv314/bin/python setup.py bdist_wheel --dist-dir dist >> /home/ussoewwin/SageAttention/build_linux.log 2>&1

echo "=== Linux Wheel Build Done $(date) ===" >> /home/ussoewwin/SageAttention/build_linux.log
