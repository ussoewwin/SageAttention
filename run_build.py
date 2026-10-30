import os
import sys
import subprocess

os.environ["PATH"] = "/home/ussoewwin/venv314/bin:/usr/local/cuda-13.2/bin:/usr/bin:/bin:/usr/local/bin"
os.environ["CC"] = "/usr/bin/gcc"
os.environ["CXX"] = "/usr/bin/g++"
os.environ["CUDAHOSTCXX"] = "/usr/bin/g++"
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.2"
os.environ["CUDA_PATH"] = "/usr/local/cuda-13.2"
os.environ["LIBRARY_PATH"] = f"/usr/lib/wsl/lib:/usr/local/cuda-13.2/lib64/stubs:{os.environ.get('LIBRARY_PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib:/usr/local/cuda-13.2/lib64/stubs:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0;10.0;12.0;12.1"
os.environ["NVCC_APPEND_FLAGS"] = "--threads 2 --expt-relaxed-constexpr -diag-suppress=20013 -diag-suppress=20015"
os.environ["EXT_PARALLEL"] = "1"
os.environ["MAX_JOBS"] = "4"

print("Starting bdist_wheel in WSL...")
sys.stdout.flush()

res = subprocess.run([
    "/home/ussoewwin/venv314/bin/python",
    "setup.py",
    "bdist_wheel",
    "--dist-dir",
    "dist"
], cwd="/home/ussoewwin/SageAttention")

sys.exit(res.returncode)
