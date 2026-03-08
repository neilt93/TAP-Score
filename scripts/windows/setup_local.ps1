# setup_local.ps1 — Full environment setup for Windows + RTX 4080
# Windows equivalent of scripts/setup_runpod.sh
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\windows\setup_local.ps1
#
# Prerequisites:
#   - Python 3.10.x installed and on PATH
#   - Git installed and on PATH
#   - NVIDIA driver installed (RTX 4080)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

Write-Host "=============================================="
Write-Host "  TAP-Score Windows Local Setup"
Write-Host "  Project root: $ProjectRoot"
Write-Host "=============================================="

Push-Location $ProjectRoot
try {

# ── Step 1: Create venv ──
Write-Host ""
Write-Host "[1/8] Creating Python virtual environment..."
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "  Created venv."
} else {
    Write-Host "  venv already exists."
}

# Activate venv
& ".\venv\Scripts\Activate.ps1"
Write-Host "  Activated venv. Python: $(python --version)"

# ── Step 2: Install PyTorch with CUDA 12.1 ──
Write-Host ""
Write-Host "[2/8] Installing PyTorch with CUDA 12.1..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
Write-Host "  Done."

# ── Step 3: Install core dependencies ──
Write-Host ""
Write-Host "[3/8] Installing core Python dependencies..."
pip install -q `
    numpy zarr h5py `
    matplotlib seaborn `
    scikit-learn pandas `
    tqdm `
    hydra-core omegaconf dill `
    einops diffusers numba scipy `
    "gym==0.26.2" pygame "pymunk==6.4.0" shapely `
    opencv-python-headless scikit-image `
    wandb
Write-Host "  Done."

# ── Step 4: Install cmake (needed by robomimic build) ──
Write-Host ""
Write-Host "[4/8] Installing cmake via pip..."
pip install -q cmake
Write-Host "  Done."

# ── Step 5: Install MuJoCo + robomimic ──
Write-Host ""
Write-Host "[5/8] Installing MuJoCo and robomimic..."
pip install -q mujoco

# Try wheels first, fall back to source
try {
    pip install -q "robomimic==0.2.0"
    Write-Host "  robomimic installed from wheel."
} catch {
    Write-Host "  Wheel install failed, trying from source..."
    pip install -q "git+https://github.com/ARISE-Initiative/robomimic.git@v0.2.0"
    Write-Host "  robomimic installed from source."
}

# ── Step 6: Clone and install Diffusion Policy ──
Write-Host ""
Write-Host "[6/8] Setting up Diffusion Policy..."
if (-not (Test-Path "baselines\diffusion_policy")) {
    New-Item -ItemType Directory -Path "baselines" -Force | Out-Null
    git clone https://github.com/real-stanford/diffusion_policy.git baselines/diffusion_policy
}
Push-Location "baselines\diffusion_policy"
pip install -q -e .
Pop-Location
Write-Host "  Done."

# ── Step 7: Download robomimic datasets ──
Write-Host ""
Write-Host "[7/8] Downloading robomimic lowdim datasets (lift, can)..."
$LiftHDF5 = "data\robomimic\datasets\lift\ph\low_dim.hdf5"
$CanHDF5 = "data\robomimic\datasets\can\ph\low_dim.hdf5"

if ((-not (Test-Path $LiftHDF5)) -or (-not (Test-Path $CanHDF5))) {
    New-Item -ItemType Directory -Path "data\robomimic\datasets" -Force | Out-Null
    python -m robomimic.scripts.download_datasets `
        --tasks lift can `
        --dataset_types ph `
        --hdf5_types low_dim `
        --download_dir data/robomimic/datasets
} else {
    Write-Host "  Datasets already exist."
}

# Create directory junction for DP's expected path
$DPDatasetLink = "baselines\diffusion_policy\data\robomimic\datasets"
if (-not (Test-Path $DPDatasetLink)) {
    New-Item -ItemType Directory -Path "baselines\diffusion_policy\data\robomimic" -Force | Out-Null
    $Target = (Resolve-Path "data\robomimic\datasets").Path
    New-Item -ItemType Junction -Path $DPDatasetLink -Target $Target | Out-Null
    Write-Host "  Created directory junction: $DPDatasetLink -> $Target"
}
Write-Host "  Done."

# ── Step 8: Download pretrained DP checkpoints ──
Write-Host ""
Write-Host "[8/8] Downloading robomimic lowdim DP checkpoints..."
$CkptBase = "https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim"
$CkptDir = "data\robomimic\checkpoints"
New-Item -ItemType Directory -Path $CkptDir -Force | Out-Null

$LiftCkpt = "$CkptDir\lift_ph_diffusion_policy_cnn.ckpt"
if (-not (Test-Path $LiftCkpt)) {
    Write-Host "  Downloading lift checkpoint (this may take a while)..."
    Invoke-WebRequest `
        -Uri "$CkptBase/lift_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt" `
        -OutFile $LiftCkpt
    Write-Host "  Lift checkpoint downloaded."
} else {
    Write-Host "  Lift checkpoint already exists."
}

$CanCkpt = "$CkptDir\can_ph_diffusion_policy_cnn.ckpt"
if (-not (Test-Path $CanCkpt)) {
    Write-Host "  Downloading can checkpoint (this may take a while)..."
    Invoke-WebRequest `
        -Uri "$CkptBase/can_ph/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt" `
        -OutFile $CanCkpt
    Write-Host "  Can checkpoint downloaded."
} else {
    Write-Host "  Can checkpoint already exists."
}
Write-Host "  Done."

# ── Verify ──
Write-Host ""
Write-Host "=============================================="
Write-Host "  Verifying Setup"
Write-Host "=============================================="
python scripts/windows/verify_setup.py

Write-Host ""
Write-Host "=============================================="
Write-Host "  Setup Complete"
Write-Host "=============================================="
Write-Host "Next steps:"
Write-Host "  1. python scripts\windows\verify_setup.py       (detailed check)"
Write-Host "  2. python scripts\inspect_robomimic_lowdim.py --task all  (validate HDF5)"
Write-Host "  3. powershell scripts\windows\run_full_pipeline.ps1       (run pipeline)"

} finally {
    Pop-Location
}
