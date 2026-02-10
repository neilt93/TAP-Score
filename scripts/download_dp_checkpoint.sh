#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DP_ROOT="${PROJECT_ROOT}/baselines/diffusion_policy"

CKPT_DIR="${DP_ROOT}/data/checkpoints"
CKPT_PATH="${CKPT_DIR}/pusht_image_latest.ckpt"

# Published checkpoint from Diffusion Policy model zoo
CKPT_URL="https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"

mkdir -p "${CKPT_DIR}"

if [[ -f "${CKPT_PATH}" ]]; then
  echo "[download_dp_checkpoint] Checkpoint already exists: ${CKPT_PATH}"
  echo "[download_dp_checkpoint] To re-download, delete the file first."
else
  echo "[download_dp_checkpoint] Downloading DP checkpoint to: ${CKPT_PATH}"
  echo "[download_dp_checkpoint] URL: ${CKPT_URL}"
  curl -L --progress-bar -o "${CKPT_PATH}" "${CKPT_URL}"
  echo "[download_dp_checkpoint] Download complete."
fi

# Verify file exists and has nonzero size
if [[ ! -s "${CKPT_PATH}" ]]; then
  echo "[download_dp_checkpoint] ERROR: Downloaded file is empty or missing."
  exit 1
fi

FILE_SIZE=$(stat -f%z "${CKPT_PATH}" 2>/dev/null || stat --printf="%s" "${CKPT_PATH}" 2>/dev/null || echo "unknown")
echo "[download_dp_checkpoint] Checkpoint: ${CKPT_PATH} (${FILE_SIZE} bytes)"
echo "[download_dp_checkpoint] Done."
