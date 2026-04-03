#!/bin/bash
# Generate blog post + X thread from analysis results using Claude API.
#
# Prerequisites:
#   export ANTHROPIC_API_KEY="your-key-here"
#   pip install anthropic
#
# Usage:
#   bash scripts/run_content.sh              # BitNet 2B4T results (default)
#   bash scripts/run_content.sh bitnet2b     # Same as above
#   bash scripts/run_content.sh --dry-run    # Preview prompts without calling API
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Install anthropic SDK if not present
pip install anthropic -q 2>/dev/null

PHASE="${1:-bitnet2b}"

if [ "$PHASE" = "--dry-run" ]; then
    python -m scripts.generate_content \
        --results results/bitnet2b/layer_analysis.json \
        --figures-dir figures \
        --phase bitnet2b \
        --dry-run
    exit 0
fi

case "$PHASE" in
    bitnet2b)
        RESULTS_FILE="results/bitnet2b/layer_analysis.json"
        ;;
    replica_training)
        RESULTS_FILE="results/measurements/summary.json"
        ;;
    lr_sweep)
        RESULTS_FILE="results/lr_sweep/summary.json"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [bitnet2b|replica_training|lr_sweep|--dry-run]"
        exit 1
        ;;
esac

if [ ! -f "$RESULTS_FILE" ]; then
    echo "ERROR: Results file not found: $RESULTS_FILE"
    echo "Run the analysis first (e.g., bash scripts/run_phase2.sh)"
    exit 1
fi

echo "============================================"
echo "  Content Generation: $PHASE"
echo "============================================"
echo ""

python -m scripts.generate_content \
    --results "$RESULTS_FILE" \
    --figures-dir figures \
    --phase "$PHASE" \
    --output-dir content

echo ""
echo "Files are in content/"
echo "Open them with any text editor to review before posting."
