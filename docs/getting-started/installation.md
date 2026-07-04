# Installation

!!! tip "No local GPU? Try Google Colab"

    Our [Google Colab tutorial](https://colab.research.google.com/drive/1UIJiLVpyeg8_-GUKDrqPKuT3RL8QMPdo) runs the full detect + segment pipeline on a sample orthomosaic on a free T4 GPU and visualizes the results — no local install needed. Its setup is adapted for Colab, and you can reuse its cells in your own notebook to run inference on your own data. Colab is slower and can't handle large orthomosaics, so it's best for testing or small datasets; for production use, install CanopyRS locally.

## Requirements

- **OS:** Linux (Ubuntu 22.04 recommended). Windows 10 is also supported but might be trickier to setup. MacOS is untested.
- **Python:** 3.10
- **CUDA:** 12.6 — You can install CUDA by following the NVIDIA CUDA installation [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), or directly downloading it from this [link](https://developer.nvidia.com/cuda-12-6-0-download-archive). The command `nvcc --version` should show version 12.6.

## Step-by-step

**1. Clone the repository**

```bash
git clone https://github.com/hugobaudchon/CanopyRS.git
cd CanopyRS
```

**2. Create a conda environment with mamba**

```bash
conda create -n canopyrs -c conda-forge python=3.10 mamba
conda activate canopyrs
```

**3. Install GDAL via mamba**

```bash
mamba install gdal=3.6.2 -c conda-forge
```

**4. Install PyTorch with CUDA 12.6 support**

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

**5. Initialize submodules**

```bash
git submodule update --init --recursive
```

**6. Install CanopyRS and dependencies**

```bash
python -m pip install -e .
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-build-isolation -e ./detrex/detectron2 -e ./detrex
```

On NVIDIA L4 hosts, `8.9` is the correct compute capability. If you build on a different GPU family, replace `TORCH_CUDA_ARCH_LIST` accordingly. The CUDA toolkit must be installed on the host and `nvcc --version` must work before you run the install step above.

## SAM 3 — Hugging Face access request

SAM 3 is a gated model hosted by Meta on Hugging Face. If you plan to use SAM 3 (either directly or through a SAM 3 preset), you must **request access before your first run**:

1. Go to the [facebook/sam3](https://huggingface.co/facebook/sam3) model page on Hugging Face.
2. Click **"Request access"** and accept Meta's license terms.
3. Make sure you are logged in to Hugging Face on your machine:
   ```bash
   huggingface-cli login
   ```

Without this step, any pipeline using SAM 3 will fail when trying to download the base model weights.

## Known issues

You will likely encounter this error during installation:

```
sam2 0.4.1 requires iopath>=0.1.10, but you have iopath 0.1.9 which is incompatible
```

This is a conflict between Detectron2 and SAM2 libraries, but it can be ignored and should not impact installation or usage of the pipeline.

## Verify the installation

```bash
python -c "import canopyrs; print('CanopyRS installed successfully')"
```

For detrex models, also verify that the CUDA extension executes on GPU:

```bash
python - <<'PY'
import torch
from detrex.layers import MultiScaleDeformableAttention

device = torch.device("cuda")
module = MultiScaleDeformableAttention(
    embed_dim=32,
    num_heads=4,
    num_levels=1,
    num_points=2,
    batch_first=True,
).to(device)

query = torch.randn(1, 2, 32, device=device)
value = torch.randn(1, 4, 32, device=device)
reference_points = torch.full((1, 2, 1, 2), 0.5, device=device)
spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long, device=device)
level_start_index = torch.tensor([0], dtype=torch.long, device=device)

output = module(
    query=query,
    value=value,
    reference_points=reference_points,
    spatial_shapes=spatial_shapes,
    level_start_index=level_start_index,
)

torch.cuda.synchronize()
print(output.device)
PY
```

If this fails with `Not compiled with GPU support`, `detrex` was built without CUDA ops and DINO or Mask2Former inference will not run on GPU until you rebuild it.
