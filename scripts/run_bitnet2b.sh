#!/bin/bash
# Phase 0: Analyze Microsoft BitNet 2B4T pretrained model
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== BitNet 2B4T Analysis ==="
echo "This downloads ~4GB model and analyzes all weight matrices."
echo ""

python -m analysis.bitnet2b_analysis \
    --output-dir results/bitnet2b \
    --gauge-trials 5000

echo ""
echo "=== Done. Results in results/bitnet2b/ ==="
