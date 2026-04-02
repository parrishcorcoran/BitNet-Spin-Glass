#!/bin/bash
# Phase 6: Learning rate sweep (temperature scanning).
# Maps effective temperature T_eff ~ lr / batch_size.
# M-size model on MNIST, 100 replicas per LR value.
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

MAX_PARALLEL=${MAX_PARALLEL:-8}

LR_VALUES="1e-4 3e-4 1e-3 3e-3 1e-2"
N_REPLICAS=100

echo "=== Learning Rate Sweep (Phase Diagram) ==="
echo "Size: M, Dataset: MNIST"
echo "LR values: $LR_VALUES"
echo "Replicas per LR: $N_REPLICAS"
echo "Max parallel: $MAX_PARALLEL"
echo ""

COMMANDS_FILE=$(mktemp)

for WEIGHT in fp ternary; do
    for ACT in relu sign; do
        for LR in $LR_VALUES; do
            for SEED in $(seq 0 $((N_REPLICAS - 1))); do
                echo "python -m src.train --size M --weight $WEIGHT --activation $ACT --dataset mnist --seed $SEED --lr-override $LR" >> "$COMMANDS_FILE"
            done
        done
    done
done

TOTAL=$(wc -l < "$COMMANDS_FILE")
echo "Total runs: $TOTAL"

cat "$COMMANDS_FILE" | xargs -P "$MAX_PARALLEL" -I {} bash -c '{}'

rm "$COMMANDS_FILE"
echo "=== LR sweep complete ==="
