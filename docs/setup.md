# Lambda Stack 22.04 (arm64) Environment Setup

This guide documents the exact environment required to run the Illustrious LoRA training notebook on a Lambda Stack 22.04 machine with an ARM64 GPU node (e.g., Jetson, Grace Hopper, or ARM-based Lambda Cloud instances). It includes the canonical versions, installation steps, verification commands, and troubleshooting guidance.

## Required Software Versions

| Component | Version | Notes |
| --- | --- | --- |
| Python | 3.10.12 | Default Python in Lambda Stack 22.04 (Ubuntu 22.04 LTS). |
| CUDA Toolkit | 12.1 | Installed with Lambda Stack 22.04; pairs with PyTorch 2.1 wheels. |
| cuDNN | 8.9.2 | Included with CUDA 12.1 in Lambda Stack 22.04 image. |
| PyTorch | 2.1.2 (`torch==2.1.2+cu121`) | Official Linux aarch64 wheel, compiled against CUDA 12.1 + cuDNN 8.9. |
| torchvision | 0.16.2 (`torchvision==0.16.2+cu121`) | Must match PyTorch CUDA minor version. |
| torchaudio | 2.1.2 (`torchaudio==2.1.2+cu121`) | Optional but recommended for parity with upstream wheels. |
| diffusers | 0.25.0 | Stable baseline for SDXL LoRA training. |
| accelerate | 0.25.0 | Handles mixed precision and accelerator detection. |
| transformers | 4.36.2 | Required for text encoders in SDXL pipelines. |
| xFormers | 0.0.23.post1 | Prebuilt wheel available for aarch64 CUDA 12.1; enables memory-efficient attention. |
| bitsandbytes | 0.41.3 | Provides 8-bit optimizers; community ARM64 CUDA wheel available. |
| safetensors | 0.4.1 | Recommended format for model weights. |
| peft | 0.7.1 | Hugging Face PEFT library for LoRA utilities. |

> **Note:** If you must run entirely on CPU, you can install `torch==2.1.2` without the CUDA suffix, but LoRA training will be prohibitively slow.

## Installation Steps

1. **Ensure system packages are current**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3.10 python3.10-venv python3-pip build-essential git
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python3.10 -m venv ~/envs/illustrious-arm64
   source ~/envs/illustrious-arm64/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   ```

3. **Install CUDA-enabled PyTorch wheels** for ARM64:
   ```bash
   # Lambda Stack already configures CUDA 12.1 libraries system-wide.
   pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
       --extra-index-url https://download.pytorch.org/whl/cu121
   ```
   If you encounter SSL issues with the extra index, export `PIP_NO_VERIFY_CERT=false` or use `pip --trusted-host download.pytorch.org`.

4. **Install optional performance libraries**:
   ```bash
   # xFormers build for CUDA 12.1 / aarch64.
   pip install https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/xformers-0.0.23/xformers-0.0.23.post1-cp310-cp310-linux_aarch64.whl

   # bitsandbytes prebuilt wheel targeting CUDA 12.x on aarch64.
   pip install https://github.com/jllllll/bitsandbytes-wheels/releases/download/v0.41.3/bitsandbytes-0.41.3-cp310-cp310-linux_aarch64.whl
   ```
   If these wheels fail due to GLIBC version mismatch, see the troubleshooting section below for source build flags.

5. **Install remaining Python dependencies** used by the notebook:
   ```bash
   pip install diffusers==0.25.0 accelerate==0.25.0 transformers==4.36.2 \
       safetensors==0.4.1 peft==0.7.1 tqdm==4.66.1 datasets==2.15.0
   ```

6. **(Optional) Enable SDPA fallback** if xFormers is unavailable:
   ```bash
   pip install triton==2.1.0
   # Use `--enable_xformers_memory_efficient_attention False` when launching training.
   ```

## Verifying the Installation

Run the following diagnostics inside the virtual environment:

```bash
python -c "import sys; print(f'Python {sys.version}')"
python -c "import torch; print('PyTorch', torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version())"
python -c "import torch; print('Device count:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
python -c "import xformers.ops; print('xFormers OK')"
python -c "import bitsandbytes as bnb; print('bitsandbytes uses CUDA', bnb.__version__)"
```

Expected output includes PyTorch `2.1.2+cu121`, CUDA `12.1`, cuDNN `8902`, and the detected GPU model (e.g., `NVIDIA A10G` or `NVIDIA GH200`).

## Troubleshooting (ARM64-specific)

### PyTorch wheel fails with `illegal instruction`

- Ensure you are using the correct wheel (`+cu121`) for CUDA 12.1.
- Double-check that your GPU driver is at least `535.xx`, as older Lambda Stack images may not support CUDA 12.x.
- Verify that the host kernel exposes SVE/NEON instructions required by PyTorch by running `lscpu | grep Flags`.

### `xformers` import errors (`undefined symbol` or `GLIBCXX_3.4.30 not found`)

- Install `libstdc++-12-dev` (`sudo apt install g++-12`) and retry.
- If your GLIBC version is older than 2.35, rebuild from source:
  ```bash
  sudo apt install ninja-build cmake
  CXX=g++-12 pip install git+https://github.com/facebookresearch/xformers.git@v0.0.23
  ```

### `bitsandbytes` reports `CUDA Setup failed` on ARM64

- Confirm that `LD_LIBRARY_PATH` includes `/usr/lib/wsl/lib` (WSL) or `/usr/lib/aarch64-linux-gnu` (native).
- If the prebuilt wheel fails, build from source with:
  ```bash
  sudo apt install -y git cmake ninja-build
  CUDA_VERSION=121 make CUDA_VERSION=121 python
  ```
  (See [bitsandbytes build instructions](https://github.com/TimDettmers/bitsandbytes#compile-from-source) for additional flags.)

### `torch.cuda.get_device_name(0)` raises `AssertionError: Torch not compiled with CUDA enabled`

- Reinstall PyTorch with the CUDA build (`torch==2.1.2+cu121`).
- Ensure you activated the virtual environment that contains the CUDA wheel.
- Check `nvcc --version` to verify CUDA toolkit presence. If missing, reinstall Lambda Stack or CUDA 12.1 toolkit.

### Mixed precision instability on ARM GPUs

- Use `accelerate config` to set `bf16` mixed precision only if your GPU advertises BF16 support (`nvidia-smi --query-gpu=compute_cap --format=csv` >= 8.9).
- Otherwise, set `--mixed_precision fp16` or disable mixed precision.

## Useful References

- [Lambda Stack 22.04 Release Notes](https://lambdalabs.com/blog/lambda-stack-deep-learning-software) – driver and CUDA compatibility.
- [PyTorch ARM64 Install Guide](https://pytorch.org/get-started/locally/) – use "Linux / Pip / Python / CUDA 12.1 / ARM 64" selectors.
- [Hugging Face Diffusers Installation](https://huggingface.co/docs/diffusers/installation) – GPU-specific tips.

