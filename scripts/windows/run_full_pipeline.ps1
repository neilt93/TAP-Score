# run_full_pipeline.ps1 — Full robomimic preflight + headroom audit pipeline
# Windows equivalent of scripts/run_full_pipeline.sh
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\windows\run_full_pipeline.ps1
#
# No MUJOCO_GL needed — headroom audit uses render=False, render_offscreen=False,
# and Windows uses native WGL for any MuJoCo rendering anyway.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# Checkpoint paths (match setup_local.ps1 / setup_runpod.sh naming)
$LiftCkpt = "data\robomimic\checkpoints\lift_ph_diffusion_policy_cnn.ckpt"
$CanCkpt = "data\robomimic\checkpoints\can_ph_diffusion_policy_cnn.ckpt"

Push-Location $ProjectRoot
try {

# Activate venv if not already active
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path "venv\Scripts\Activate.ps1") {
        & ".\venv\Scripts\Activate.ps1"
    }
}

Write-Host "=============================================="
Write-Host "  TAP-Score Robomimic Full Pipeline (Windows)"
Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=============================================="

# Verify checkpoints exist before starting
if (-not (Test-Path $LiftCkpt)) {
    Write-Error "Lift checkpoint not found: $LiftCkpt. Run setup_local.ps1 first."
}
if (-not (Test-Path $CanCkpt)) {
    Write-Error "Can checkpoint not found: $CanCkpt. Run setup_local.ps1 first."
}

# Create output directory
New-Item -ItemType Directory -Path "eval_results" -Force | Out-Null

# ── Step 1: Stochasticity check (lift) ──
Write-Host ""
Write-Host "=== STEP 1/4: Stochasticity check (lift) ==="
python scripts/check_policy_stochasticity.py `
    --dp_checkpoint $LiftCkpt --K 8 --n_obs 5
Write-Host "STEP 1 DONE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ── Step 2: Stochasticity check (can) ──
Write-Host ""
Write-Host "=== STEP 2/4: Stochasticity check (can) ==="
python scripts/check_policy_stochasticity.py `
    --dp_checkpoint $CanCkpt --K 8 --n_obs 5
Write-Host "STEP 2 DONE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ── Steps 3+4: Headroom audits (lift + can) IN PARALLEL ──
# MuJoCo physics runs on CPU; only policy inference uses GPU.
# Running both in parallel roughly doubles GPU utilization.
Write-Host ""
Write-Host "=== STEPS 3+4: Headroom audits (lift + can) — PARALLEL ==="
Write-Host "  Both tasks share the GPU. Use --resume to restart if interrupted."

$LiftJob = Start-Job -ScriptBlock {
    param($Root, $Ckpt)
    Set-Location $Root
    python scripts/robomimic_headroom_audit.py `
        --dp_checkpoint $Ckpt `
        --n_episodes 50 --K 8 `
        --decision_interval 5 --skip_first 2 `
        --device cuda --resume `
        --output eval_results/robomimic_headroom_lift_K8_n50.json 2>&1
} -ArgumentList $ProjectRoot, $LiftCkpt

$CanJob = Start-Job -ScriptBlock {
    param($Root, $Ckpt)
    Set-Location $Root
    python scripts/robomimic_headroom_audit.py `
        --dp_checkpoint $Ckpt `
        --n_episodes 50 --K 8 `
        --decision_interval 5 --skip_first 2 `
        --device cuda --resume `
        --output eval_results/robomimic_headroom_can_K8_n50.json 2>&1
} -ArgumentList $ProjectRoot, $CanCkpt

Write-Host "  Lift job: $($LiftJob.Id)   Can job: $($CanJob.Id)"
Write-Host "  Waiting for both to complete..."

# Stream progress updates every 60 seconds
while (($LiftJob.State -eq 'Running') -or ($CanJob.State -eq 'Running')) {
    Start-Sleep -Seconds 60
    $ts = Get-Date -Format 'HH:mm:ss'
    $liftStatus = $LiftJob.State
    $canStatus  = $CanJob.State
    Write-Host "  [$ts] lift=$liftStatus  can=$canStatus"
}

# Collect output
Write-Host ""
Write-Host "--- Lift output ---"
Receive-Job -Job $LiftJob | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "--- Can output ---"
Receive-Job -Job $CanJob | ForEach-Object { Write-Host $_ }

Remove-Job -Job $LiftJob, $CanJob -Force

Write-Host "STEPS 3+4 DONE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ── Summary ──
Write-Host ""
Write-Host "=============================================="
Write-Host "  PIPELINE COMPLETE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=============================================="
Write-Host "Results:"
Write-Host "  eval_results/robomimic_headroom_lift_K8_n50.json"
Write-Host "  eval_results/robomimic_headroom_can_K8_n50.json"
Write-Host ""
Write-Host "Check 'fork_decision' in each JSON."

# Print fork decisions
foreach ($f in Get-ChildItem "eval_results\robomimic_headroom_*.json" -ErrorAction SilentlyContinue) {
    if ($f.Name -notmatch '\.ckpt\.json$') {
        Write-Host ""
        Write-Host "--- $($f.Name) ---"
        python -c @"
import json
d = json.load(open(r'$($f.FullName)'))
keys = {k: v for k, v in d.items() if 'fork' in k.lower() or 'summary' in k.lower()}
print(json.dumps(keys, indent=2))
"@
    }
}

} finally {
    Pop-Location
}
