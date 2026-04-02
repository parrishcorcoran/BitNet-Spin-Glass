#!/bin/bash
# Phase 4: Run permutation alignment on all replica pairs.
# Must complete BEFORE any overlap computation.
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Permutation Alignment (Hungarian algorithm) ==="
echo "This aligns all replica pairs using weight matching."
echo "For ternary weights, reports tie-breaking sensitivity across 5 seeds."
echo ""

python -c "
import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations

from src.config import MODEL_SIZES, EXPERIMENT_CONFIGS, CHECKPOINTS_DIR, ALIGNMENT_TIE_BREAKING_SEEDS
from src.config import WeightMode, ActivationMode, DatasetName
from src.model import SpinGlassMLP
from analysis.align import align_replicas, align_and_report_sensitivity

results_dir = Path('results/aligned')
results_dir.mkdir(parents=True, exist_ok=True)

ckpt_root = Path(CHECKPOINTS_DIR)

# Find all run groups (same config, different seeds)
run_dirs = sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []

# Group by config (everything except seed)
from collections import defaultdict
groups = defaultdict(list)
for d in run_dirs:
    parts = d.name.rsplit('_seed', 1)
    if len(parts) == 2:
        groups[parts[0]].append(d)

for config_name, dirs in tqdm(groups.items(), desc='Configs'):
    if len(dirs) < 2:
        continue

    print(f'Aligning {config_name}: {len(dirs)} replicas')

    # Load final epoch weights for each replica
    # (use the highest epoch checkpoint available)
    replica_weights = []
    for d in dirs:
        ckpts = sorted(d.glob('epoch_*.pt'))
        if not ckpts:
            continue
        state = torch.load(ckpts[-1], map_location='cpu', weights_only=True)
        # Extract weight matrices (keys containing '.weight')
        weights = [v.numpy() for k, v in state.items() if 'weight' in k and v.ndim == 2]
        replica_weights.append(weights)

    if len(replica_weights) < 2:
        continue

    # Align all replicas to replica 0
    reference = replica_weights[0]
    aligned = [reference]  # replica 0 is the reference

    sensitivity_reports = []
    for i in range(1, len(replica_weights)):
        report = align_and_report_sensitivity(
            [w.copy() for w in reference],
            [w.copy() for w in replica_weights[i]],
            n_seeds=ALIGNMENT_TIE_BREAKING_SEEDS,
        )
        aligned.append(report['aligned_weights'])
        sensitivity_reports.append({
            'replica': i,
            'overlap_mean': report['overlap_mean'],
            'overlap_std': report['overlap_std'],
        })

    # Save aligned weights
    out_path = results_dir / f'{config_name}_aligned.npz'
    save_dict = {}
    for r_idx, ws in enumerate(aligned):
        for l_idx, w in enumerate(ws):
            save_dict[f'r{r_idx}_l{l_idx}'] = w
    np.savez_compressed(str(out_path), **save_dict)

    # Save sensitivity report
    with open(results_dir / f'{config_name}_sensitivity.json', 'w') as f:
        json.dump(sensitivity_reports, f, indent=2)

    print(f'  Saved {len(aligned)} aligned replicas to {out_path}')

print('=== Alignment complete ===')
"

echo "=== Done. Aligned weights in results/aligned/ ==="
