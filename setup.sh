#!/bin/bash
set -e

echo "=== BitNet Spin Glass Project Setup ==="

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# CPU-only PyTorch (no CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Create output directories
mkdir -p figures results paper dashboard

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
