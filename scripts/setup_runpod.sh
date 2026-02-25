#!/bin/bash
# One-shot RunPod setup for TAP-Score + Diffusion Policy
# Usage: bash scripts/setup_runpod.sh
set -euo pipefail

echo "=== TAP-Score RunPod Setup ==="

# System dependencies (cmake needed by robomimic's egl_probe)
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq cmake > /dev/null 2>&1
echo "  Done."

# Python dependencies
echo "[2/6] Installing Python dependencies..."
pip install -q \
  torch torchvision numpy \
  zarr h5py \
  matplotlib seaborn \
  scikit-learn pandas \
  tqdm \
  hydra-core omegaconf dill \
  gym==0.26.2 pygame pymunk==6.4.0 shapely \
  opencv-python-headless scikit-image \
  einops diffusers numba scipy \
  wandb robomimic==0.2.0
echo "  Done."

# Clone and install Diffusion Policy
echo "[3/6] Setting up Diffusion Policy..."
if [ ! -d "baselines/diffusion_policy" ]; then
  mkdir -p baselines
  git clone https://github.com/real-stanford/diffusion_policy.git baselines/diffusion_policy
fi
cd baselines/diffusion_policy && pip install -q -e . && cd ../..
echo "  Done."

# Download Push-T data
echo "[4/6] Downloading Push-T dataset..."
if [ ! -d "data/raw/pusht/pusht_cchi_v7_replay.zarr" ]; then
  mkdir -p data/raw
  wget -q -O data/raw/pusht.zip "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
  cd data/raw && python3 -c "import zipfile; zipfile.ZipFile('pusht.zip','r').extractall('.'); print('  Unzipped.')" && cd ../..
  rm -f data/raw/pusht.zip
else
  echo "  Already exists."
fi
mkdir -p baselines/diffusion_policy/data
ln -sfn "$(pwd)/data/raw/pusht" baselines/diffusion_policy/data/pusht

# Download DP checkpoint
echo "[5/6] Downloading robomimic lowdim datasets (lift, can)..."
if [ ! -f "data/robomimic/datasets/lift/ph/low_dim.hdf5" ] || \
   [ ! -f "data/robomimic/datasets/can/ph/low_dim.hdf5" ]; then
  mkdir -p data/robomimic/datasets
  python -m robomimic.scripts.download_datasets \
    --tasks lift can \
    --dataset_types ph \
    --hdf5_types low_dim \
    --download_dir data/robomimic/datasets
else
  echo "  Already exists."
fi
# Symlink into DP's expected path
mkdir -p baselines/diffusion_policy/data/robomimic
ln -sfn "$(pwd)/data/robomimic/datasets" baselines/diffusion_policy/data/robomimic/datasets

echo "[6/6] Downloading Diffusion Policy checkpoint (~4GB)..."
CKPT=baselines/diffusion_policy/data/checkpoints/pusht_image_latest.ckpt
if [ ! -f "$CKPT" ]; then
  mkdir -p baselines/diffusion_policy/data/checkpoints
  wget -O "$CKPT" "https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"
else
  echo "  Already exists."
fi

# Verify
echo ""
echo "=== Verifying Setup ==="
python3 -c "
from tap.contrastive import ContrastiveTAPScore; print('  TAP import: OK')
import zarr; z=zarr.open_group('data/raw/pusht/pusht_cchi_v7_replay.zarr','r'); print(f'  Zarr data: {z[\"data\"][\"action\"].shape[0]} timesteps')
import os; s=os.path.getsize('$CKPT'); print(f'  DP checkpoint: {s/1e9:.1f} GB')
import h5py
for task in ['lift', 'can']:
    p = f'data/robomimic/datasets/{task}/ph/low_dim.hdf5'
    if os.path.exists(p):
        with h5py.File(p, 'r') as f:
            n = len([k for k in f['data'].keys() if k.startswith('demo_')])
            print(f'  Robomimic {task}: {n} demos')
    else:
        print(f'  Robomimic {task}: NOT FOUND')
import torch; print(f'  CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')
print('  All good!')
"

echo ""
echo "=== Setup Complete ==="
echo "Run smoke test with:"
echo "  bash scripts/run_smoke.sh"
