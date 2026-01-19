#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_DIR="${PROJECT_ROOT}/data/raw"
PROCESSED_DIR="${PROJECT_ROOT}/data/processed"
DP_ROOT="${PROJECT_ROOT}/baselines/diffusion_policy"

URL="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZIP_PATH="${RAW_DIR}/pusht.zip"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"

echo "[download_data] Downloading Push-T to: ${ZIP_PATH}"
curl -L -o "${ZIP_PATH}" "${URL}"

echo "[download_data] Unzipping..."
unzip -q -o "${ZIP_PATH}" -d "${RAW_DIR}"

# Most commonly extracts to ${RAW_DIR}/pusht
EXTRACT_DIR="${RAW_DIR}/pusht"
if [[ ! -d "${EXTRACT_DIR}" ]]; then
  # Fallback: find a directory containing the expected zarr
  FOUND="$(find "${RAW_DIR}" -maxdepth 2 -type d -name "*.zarr" | head -n 1 || true)"
  if [[ -z "${FOUND}" ]]; then
    echo "[download_data] ERROR: Could not find extracted Push-T zarr after unzip."
    exit 1
  fi
  EXTRACT_DIR="$(cd "$(dirname "${FOUND}")" && pwd)"
fi

echo "[download_data] Found extracted data at: ${EXTRACT_DIR}"

# Symlink for your project code
ln -sfn "${EXTRACT_DIR}" "${PROCESSED_DIR}/pusht"
echo "[download_data] Linked ${PROCESSED_DIR}/pusht -> ${EXTRACT_DIR}"

# Symlink into Diffusion Policy repo so their default paths work
if [[ -d "${DP_ROOT}" ]]; then
  mkdir -p "${DP_ROOT}/data"
  ln -sfn "${EXTRACT_DIR}" "${DP_ROOT}/data/pusht"
  echo "[download_data] Linked ${DP_ROOT}/data/pusht -> ${EXTRACT_DIR}"
else
  echo "[download_data] NOTE: ${DP_ROOT} not found. Skipping baseline symlink."
fi

echo "[download_data] Done."
