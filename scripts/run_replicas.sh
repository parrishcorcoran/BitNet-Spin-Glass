#!/bin/bash
# Phase 3: Launch all replica training runs.
# Uses GNU parallel (or xargs) to run 8 jobs simultaneously.
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

MAX_PARALLEL=${MAX_PARALLEL:-8}

SIZES="XS S M L XL"
WEIGHTS="fp ternary"
ACTIVATIONS="relu sign"
DATASETS="mnist cifar10 random_labels_mnist"

echo "=== Replica Training ==="
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# Generate all commands
COMMANDS_FILE=$(mktemp)

for SIZE in $SIZES; do
    # Determine replica count: 100 for M (primary), 30 for others
    if [ "$SIZE" = "M" ]; then
        N_REPLICAS=100
    else
        N_REPLICAS=30
    fi

    for WEIGHT in $WEIGHTS; do
        for ACT in $ACTIVATIONS; do
            for DS in $DATASETS; do
                for SEED in $(seq 0 $((N_REPLICAS - 1))); do
                    echo "python -m src.train --size $SIZE --weight $WEIGHT --activation $ACT --dataset $DS --seed $SEED" >> "$COMMANDS_FILE"
                done
            done
        done
    done
done

TOTAL=$(wc -l < "$COMMANDS_FILE")
echo "Total runs: $TOTAL"
echo ""

# Run with xargs for parallel execution
cat "$COMMANDS_FILE" | xargs -P "$MAX_PARALLEL" -I {} bash -c '{}'

rm "$COMMANDS_FILE"
echo "=== All replica training complete ==="
