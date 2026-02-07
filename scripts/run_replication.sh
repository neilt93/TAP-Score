#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DP_ROOT="${PROJECT_ROOT}/baselines/diffusion_policy"

OUT_DIR="${PROJECT_ROOT}/results/diffusion_policy/pusht_eval"
CKPT_DIR="${DP_ROOT}/data/checkpoints"
CKPT_PATH="${CKPT_DIR}/pusht_image_latest.ckpt"

# Published checkpoint (large file)
CKPT_URL="https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt"

mkdir -p "${OUT_DIR}"

if [[ ! -d "${DP_ROOT}" ]]; then
  echo "[run_replication] ERROR: diffusion_policy repo not found at ${DP_ROOT}"
  exit 1
fi

# Ensure dataset exists where diffusion_policy expects it
if [[ ! -e "${DP_ROOT}/data/pusht" ]]; then
  echo "[run_replication] ERROR: Missing ${DP_ROOT}/data/pusht"
  echo "[run_replication] Run: ${PROJECT_ROOT}/scripts/download_data.sh"
  exit 1
fi

# Choose device automatically
DEVICE="cpu"
CUDA_AVAIL="$(python -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)"
if [[ "${CUDA_AVAIL}" == "1" ]]; then
  DEVICE="cuda:0"
fi
echo "[run_replication] Using device: ${DEVICE}"

# Download checkpoint
mkdir -p "${CKPT_DIR}"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[run_replication] Downloading checkpoint to ${CKPT_PATH}"
  curl -L -o "${CKPT_PATH}" "${CKPT_URL}"
else
  echo "[run_replication] Checkpoint already exists: ${CKPT_PATH}"
fi

# Run eval (official eval.py usage)
cd "${DP_ROOT}"
echo "[run_replication] Running eval..."
python eval.py \
  --checkpoint "${CKPT_PATH}" \
  --output_dir "${OUT_DIR}" \
  --device "${DEVICE}" | tee "${OUT_DIR}/eval.log"

echo "[run_replication] Done. Logs at: ${OUT_DIR}/eval.log"
