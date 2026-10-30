#!/usr/bin/env bash
set -e

echo "=== Linux sa3 Wheel Build Start $(date) ===" > /home/ussoewwin/sageattention3_blackwell/build_linux.log

export PATH="/home/ussoewwin/venv314/bin:/usr/local/cuda-13.2/bin:/usr/bin:/bin:/usr/local/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME="/usr/local/cuda-13.2"
export CUDA_PATH="/usr/local/cuda-13.2"
export LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-13.2/lib64/stubs:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-13.2/lib64/stubs:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="10.0;12.0;12.1"
export MAX_JOBS=4

cd /home/ussoewwin/sageattention3_blackwell

rm -rf build dist

/home/ussoewwin/venv314/bin/python setup.py clean --all >> /home/ussoewwin/sageattention3_blackwell/build_linux.log 2>&1

/home/ussoewwin/venv314/bin/python setup.py bdist_wheel --dist-dir dist >> /home/ussoewwin/sageattention3_blackwell/build_linux.log 2>&1

echo "=== Linux sa3 Wheel Build Done $(date) ===" >> /home/ussoewwin/sageattention3_blackwell/build_linux.log
