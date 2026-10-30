@echo off

echo === build_a1111_cp314 %date% %time% TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;10.0;12.0;12.1 === > D:\USERFILES\GitHub\SageAttention\build_a1111_cp314.log

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "Path=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%Path%"

set DISTUTILS_USE_SDK=1
set MAX_JOBS=4
set TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;10.0;12.0;12.1

cd /d D:\USERFILES\GitHub\SageAttention

D:\USERFILES\A1111\venv\Scripts\python.exe setup.py clean --all >> D:\USERFILES\GitHub\SageAttention\build_a1111_cp314.log 2>&1

D:\USERFILES\A1111\venv\Scripts\python.exe setup.py bdist_wheel --dist-dir=dist >> D:\USERFILES\GitHub\SageAttention\build_a1111_cp314.log 2>&1

echo === done %date% %time% === >> D:\USERFILES\GitHub\SageAttention\build_a1111_cp314.log
