#!/bin/bash
# Phase 5: Run all measurements on aligned replicas.
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Running All Measurements ==="
echo ""

python -c "
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from analysis.overlaps import compute_pairwise_overlaps, compute_overlap_histogram
from analysis.frustration import compute_frustration_index
from analysis.vacancy import compute_vacancy_profile
from analysis.susceptibility import compute_susceptibility
from analysis.binder_cumulant import compute_binder_cumulant
from analysis.effect_size import full_comparison

aligned_dir = Path('results/aligned')
output_dir = Path('results/measurements')
output_dir.mkdir(parents=True, exist_ok=True)

# Load aligned weights
aligned_files = sorted(aligned_dir.glob('*_aligned.npz'))
print(f'Found {len(aligned_files)} aligned config files')

for af in tqdm(aligned_files, desc='Measuring'):
    config_name = af.stem.replace('_aligned', '')
    data = np.load(str(af))

    # Reconstruct replica list
    # Keys are like r0_l0, r0_l1, r1_l0, ...
    keys = sorted(data.files)
    replicas = {}
    for k in keys:
        r_idx, l_idx = k.split('_')
        r_idx = int(r_idx[1:])
        l_idx = int(l_idx[1:])
        if r_idx not in replicas:
            replicas[r_idx] = {}
        replicas[r_idx][l_idx] = data[k]

    # Convert to list of list of arrays
    n_replicas = len(replicas)
    n_layers = max(max(layers.keys()) for layers in replicas.values()) + 1
    aligned_weights = []
    for r in range(n_replicas):
        aligned_weights.append([replicas[r][l] for l in range(n_layers)])

    # Measurement 1: Overlaps
    overlap_result = compute_pairwise_overlaps(aligned_weights)
    q_global = overlap_result['global_overlaps']
    hist_counts, hist_edges = compute_overlap_histogram(q_global)

    # Measurement 4: Frustration (on final weights, per layer)
    frustrations = []
    for l in range(n_layers):
        w = aligned_weights[0][l]  # Use reference replica
        f = compute_frustration_index(w, gauge_trials=5000)
        frustrations.append(f)

    # Measurement 5: Vacancy
    vacancies = compute_vacancy_profile(aligned_weights[0])

    # Measurement 6: Susceptibility
    n_weights = sum(w.size for w in aligned_weights[0])
    chi_sg = compute_susceptibility(q_global, n_weights)

    # Measurement 7: Binder cumulant
    g = compute_binder_cumulant(q_global)

    results = {
        'config': config_name,
        'n_replicas': n_replicas,
        'n_pairs': overlap_result['n_pairs'],
        'overlap_mean': float(np.mean(q_global)),
        'overlap_std': float(np.std(q_global)),
        'frustrations_per_layer': frustrations,
        'vacancies_per_layer': vacancies,
        'chi_sg': chi_sg,
        'binder_cumulant': g,
        'n_weights': n_weights,
    }

    with open(output_dir / f'{config_name}_measurements.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save histogram data
    np.savez(
        str(output_dir / f'{config_name}_pq_histogram.npz'),
        counts=hist_counts, edges=hist_edges, overlaps=q_global,
    )

    print(f'  {config_name}: q_mean={results[\"overlap_mean\"]:.4f}, '
          f'g={g:.4f}, chi_SG={chi_sg:.2f}')

print('=== All measurements complete ===')
"

echo "=== Done. Results in results/measurements/ ==="
