#!/bin/bash
# Phase 2: Complete BitNet 2B4T analysis + figure generation.
# ONE COMMAND does everything:
#   1. Downloads the model (first time only, ~4GB)
#   2. Analyzes every layer (frustration, vacancy, weight distribution)
#   3. Generates publication-quality PNG figures
#
# Usage:
#   bash scripts/run_phase2.sh
#
# Results:
#   results/bitnet2b/layer_analysis.json  — raw data
#   figures/bitnet2b_*.png                — 5 PNG figures ready to post
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "============================================"
echo "  Phase 2: BitNet 2B4T Spin Glass Analysis"
echo "============================================"
echo ""
echo "Step 1/2: Analyzing pretrained model weights..."
echo "(First run downloads ~4GB model from HuggingFace)"
echo ""

python -m analysis.bitnet2b_analysis \
    --output-dir results/bitnet2b \
    --gauge-trials 5000

echo ""
echo "Step 2/2: Generating figures..."
echo ""

python -m analysis.plot_bitnet2b \
    --input results/bitnet2b/layer_analysis.json \
    --output-dir figures

echo ""
echo "============================================"
echo "  DONE! Your figures are in figures/"
echo "============================================"
echo ""
echo "Files generated:"
ls -la figures/bitnet2b_*.png 2>/dev/null || echo "  (no figures found — check for errors above)"
echo ""
echo "To view them:"
echo "  xdg-open figures/bitnet2b_summary_dashboard.png"
echo ""
echo "Or copy to your machine:"
echo "  scp yourserver:~/BitNet-Spin-Glass/figures/bitnet2b_*.png ~/Desktop/"
