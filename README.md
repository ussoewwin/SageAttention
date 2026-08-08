# Another Sage Attention Windows Fork (Sage Attention 3)

This is **another Sage Attention Windows fork**, adding support for **Sage Attention 3**.

It is based on the official Sage Attention repository, with **minimal changes**: the goal is only to fix Windows-specific build issues and provide a Windows build with **feature parity** to upstream Sage Attention.

---

## What This Fork Provides

- Based on **official Sage Attention** with the smallest possible patch set.
- Only changes related to:
  - Windows build fixes
  - Packaging / wheel generation
- No new features are added on top of upstream; the functionality should be **identical** to the official project, just **buildable and usable on Windows**.

Due to limitations of my local environment, I can currently only provide wheels for:

- **PyTorch:** 2.6.0 – 2.9.1  
- **Python:** 3.11 – 3.13  

---

## CUDA Architectures

CUDA architectures are kept **aligned with the official Windows PyTorch builds**:

- During build, the script calls:

  ```python
  torch.cuda.get_arch_list()
  ```

- The result is used to populate the `TORCH_CUDA_ARCH_LIST` environment variable.
- This keeps Sage Attention’s supported architectures **as close as possible** to those of the installed PyTorch wheel.

However, the **upstream constraints still apply** and are **not** bypassed in this fork:

- **Sage 2++** does **not** support architectures **below 8.0**.
- **Sage 3** only supports architectures **below 10.0**.
- This fork does **not** extend or change those architecture limits.

---

## Why Does This Fork Exist?

Sage Attention is a great project: it uses **quantized attention** to achieve significant speedups while keeping the CUDA kernels relatively straightforward (especially compared to something like FlashAttention). For me, it’s a **must-have attention implementation for ComfyUI**.

Unfortunately, I am a **Windows user**, and Sage Attention’s Windows support is, as many people know, quite poor.

Thankfully, there are some excellent forks out there, especially **woct0rdho**’s Windows fork. That fork:

- Ships Windows wheels
- Fixes multiple platform-specific issues
- In my experience, is sometimes even **more stable** than the original upstream

I have been using their wheels for a long time and absolutely love that branch.

Then **Sage Attention 3** arrived — and that’s where the problems started.

woct0rdho used **Python abi3** wheels to avoid version hell, which is a brilliant idea. However, Sage Attention 3 seems to rely on some new pieces that make it impossible (or at least very difficult) to keep using abi3 for building the wheels.

That was pretty frustrating.

Fortunately, I managed to get Sage Attention 3 building locally on Windows after some not-too-complicated fixes (huge thanks to:

- woct0rdho’s fork, and  
- relevant PRs from **pamparamm**  

which were very helpful).

Given the current level of Windows support in the main Sage Attention project, I felt it would be a shame to keep this to myself — so I decided to publish this fork and share the Windows builds.

The following is based on woct0rdho/SageAttention main README
---

# [SageAttention](https://github.com/thu-ml/SageAttention) fork for Windows wheels and easy installation

This repo makes it easy to build SageAttention for multiple Python, PyTorch, and CUDA versions, then distribute the wheels to other people.

The latest wheels support GTX 16xx, RTX 20xx/30xx/40xx/50xx, A100, H100, AGX Orin (sm75/80/86/87/89/90/120). There are also reports that SageAttention works with B200 (sm100) and DGX Spark (sm121), but I did not bundle these kernels in the wheels, and you need to build from source.

## Installation

1. Know how to use pip to install packages in the correct Python environment, see https://github.com/woct0rdho/triton-windows
2. Install triton-windows
3. Install a wheel on the release page: https://github.com/woct0rdho/SageAttention/releases
    * Unlike triton-windows, you need to manually choose a wheel in the GitHub release page for SageAttention
    * Choose the wheel for your PyTorch version. For example, 'torch2.7.0' in the filename
        * The recent wheels use libtorch stable ABI and have `torch2.10.0andhigher` in the filenames, so they support all versions of PyTorch >= 2.10
    * No need to worry about the CUDA minor version (12.8/12.9 ...). It can be different from yours, because SageAttention does not yet use any breaking API
        * But there is a difference between CUDA major version 12 and 13
    * No need to worry about the Python minor version (3.10/3.11 ...). The recent wheels use Python stable ABI (also known as ABI3) and have `cp310-abi3` in the filenames, so they support all versions of Python >= 3.10

If you see any error, please open an issue at https://github.com/woct0rdho/SageAttention/issues

We've simplified the installation by a lot. There is no need to install Visual Studio or CUDA toolkit to use Triton and SageAttention (unless you want to step into the world of building from source).

## Use notes

Before using SageAttention in larger projects like ComfyUI, please run [test_sageattn.py](https://github.com/woct0rdho/SageAttention/blob/main/tests/test_sageattn.py) to test if SageAttention itself works.

To use SageAttention in ComfyUI, you just need to add `--use-sage-attention` when starting ComfyUI.

Some models such as Wan and Qwen-Image may produce black or noise output when SageAttention is used, because some intermediate values overflow SageAttention's quantization. In this case, you may use the `PatchSageAttentionKJ` node in KJNodes, and choose `sageattn_qk_int8_pv_fp16_cuda`, which is the least likely to overflow.

You may also adjust `pv_accum_dtype`. `pv_accum_dtype="fp16+fp32"` (or `"fp32+fp16"` in some interfaces) is faster than `pv_accum_dtype="fp32"`, and `pv_accum_dtype="fp16"` is even faster, but more likely to cause black/noise/degraded output.

## Build from source

(This is for developers)

If you need to build and run SageAttention on your own machine:
1. Install Visual Studio (MSVC and Windows SDK), and CUDA toolkit
2. Clone this repo
   * Checkout `abi3_stable` branch if you want ABI3 and libtorch stable ABI, which supports PyTorch >= 2.10
   * Checkout `abi3` branch if you want ABI3, which supports PyTorch >= 2.4
   * There is no Python API difference between `main/abi3/abi3_stable` branches, but I recommend `abi3_stable` whenever possible, because it's more compatible with other PyTorch features such as `torch.compile` and multi-GPU
3. Install the dependencies in [`pyproject.toml`](https://github.com/woct0rdho/SageAttention/blob/main/pyproject.toml), including the desired torch version such as `torch 2.10.0+cu128`
4. Run `python setup.py install --verbose` to install directly, or `python setup.py bdist_wheel --verbose` to build a wheel
   * This avoids the environment checks of pip. There are warnings like `Please avoid running ``setup.py`` directly`, but currently it's the simplest way to deal with the exact version of torch

## Dev notes

* The wheels are built using the [workflow](https://github.com/woct0rdho/SageAttention/blob/main/.github/workflows/build-sageattn.yml)
    * It's tricky to specify both torch (with index URL at download.pytorch.org ) and pybind11 (not in that index URL) in an isolated build environment. The easiest way I could think of is to use [simpleindex](https://github.com/uranusjr/simpleindex)
* CUDA kernels for sm80/89/90 are bundled in the wheels, and also sm120 for CUDA >= 12.8
* For Turing GPUs (GTX 16xx, RTX 20xx), SageAttention 2 runs Triton kernels, which are the same as SageAttention 1. If you want to help improve the CUDA kernels for Turing, you may see https://github.com/Ph0rk0z/SageAttention2/tree/updates
* Volta GPUs (V100) are not supported because they do not have int8 tensor core
* The wheels do not use CXX11 ABI
* We cannot publish the wheels to PyPI, because PyPI does not support multiple PyTorch/CUDA variants for the same version of SageAttention. People are working on this, see https://astral.sh/blog/introducing-pyx and https://wheelnext.dev/proposals/pep817_wheel_variant_support/
* The recent wheels are bitwise reproducible given the same MSVC/WinSDK/PyTorch/CUDA/cibuildwheel versions. They're built with `/experimental:deterministic`, `/Brepro`, `/pathmap`
* The recent wheels are built with attestation. Given the .whl file, you can query the attestation from GitHub's server and see the workflow's commit hash and the link to job metadata. The checked out code's commit hash is kept in the job name, even after the build log expires
