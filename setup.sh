#!/bin/bash
# Setup environment for TAP-Score project

set -e

echo "=== TAP-Score Project Setup ==="

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data/raw data/processed results/figures checkpoints

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source venv/bin/activate"
echo "Next steps:"
echo "  1. ./scripts/download_data.sh"
echo "  2. ./scripts/run_replication.sh"
echo "  3. python train_tap.py"
