#!/bin/bash
# Download benchmark datasets for TAP-Score evaluation
# Usage: ./scripts/download_benchmarks.sh [benchmark]
#        ./scripts/download_benchmarks.sh all

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"

download_pusht() {
    echo "========================================"
    echo "Downloading Push-T dataset..."
    echo "========================================"

    if [ -d "${PROCESSED_DIR}/pusht" ]; then
        echo "Push-T already exists, skipping..."
        return
    fi

    cd "${RAW_DIR}"
    wget -nc "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip" -O pusht.zip || true
    unzip -o pusht.zip -d "${PROCESSED_DIR}/"
    echo "Push-T downloaded to ${PROCESSED_DIR}/pusht"
}

download_lift() {
    echo "========================================"
    echo "Downloading Robomimic Lift dataset..."
    echo "========================================"

    if [ -f "${PROCESSED_DIR}/lift/lift_ph_image.hdf5" ]; then
        echo "Lift already exists, skipping..."
        return
    fi

    mkdir -p "${PROCESSED_DIR}/lift"
    cd "${PROCESSED_DIR}/lift"

    # Robomimic Lift dataset (proficient-human, image observations)
    wget -nc "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image/lift_ph_image.hdf5" || true
    echo "Lift downloaded to ${PROCESSED_DIR}/lift"
}

download_kitchen() {
    echo "========================================"
    echo "Downloading Kitchen dataset..."
    echo "========================================"

    if [ -d "${PROCESSED_DIR}/kitchen" ]; then
        echo "Kitchen already exists, skipping..."
        return
    fi

    cd "${RAW_DIR}"
    wget -nc "https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip" -O kitchen.zip || true
    unzip -o kitchen.zip -d "${PROCESSED_DIR}/"
    echo "Kitchen downloaded to ${PROCESSED_DIR}/kitchen"
}

download_blockpush() {
    echo "========================================"
    echo "Downloading Block Push dataset..."
    echo "========================================"

    if [ -d "${PROCESSED_DIR}/blockpush" ]; then
        echo "Block Push already exists, skipping..."
        return
    fi

    cd "${RAW_DIR}"
    wget -nc "https://diffusion-policy.cs.columbia.edu/data/training/blockpush.zip" -O blockpush.zip || true
    unzip -o blockpush.zip -d "${PROCESSED_DIR}/"
    echo "Block Push downloaded to ${PROCESSED_DIR}/blockpush"
}

print_usage() {
    echo "Usage: $0 [benchmark]"
    echo ""
    echo "Benchmarks:"
    echo "  pusht      - Push-T (2D actions, zarr format)"
    echo "  lift       - Robomimic Lift (7D actions, hdf5 format)"
    echo "  kitchen    - Kitchen (9D actions, zarr format)"
    echo "  blockpush  - Block Push (2D actions, zarr format)"
    echo "  all        - Download all benchmarks"
    echo ""
}

# Main
BENCHMARK="${1:-}"

if [ -z "$BENCHMARK" ]; then
    print_usage
    exit 1
fi

case "$BENCHMARK" in
    pusht)
        download_pusht
        ;;
    lift)
        download_lift
        ;;
    kitchen)
        download_kitchen
        ;;
    blockpush)
        download_blockpush
        ;;
    all)
        download_pusht
        download_lift
        download_kitchen
        download_blockpush
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        print_usage
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo ""
echo "Data locations:"
ls -la "${PROCESSED_DIR}/" 2>/dev/null || echo "(no data yet)"
