@echo off

echo === build_comfyui_cp313_sageattn3 %date% %time% TORCH_CUDA_ARCH_LIST=10.0;12.0;12.1 === > D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\build_comfyui_cp313_sageattn3.log

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "Path=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%Path%"

set DISTUTILS_USE_SDK=1
set MAX_JOBS=4
set TORCH_CUDA_ARCH_LIST=10.0;12.0;12.1

cd /d D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell

D:\USERFILES\ComfyUI\python_embeded\python.exe setup.py clean --all >> D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\build_comfyui_cp313_sageattn3.log 2>&1

D:\USERFILES\ComfyUI\python_embeded\python.exe setup.py bdist_wheel --dist-dir=dist >> D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\build_comfyui_cp313_sageattn3.log 2>&1

echo === done %date% %time% === >> D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\build_comfyui_cp313_sageattn3.log
