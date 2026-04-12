# BitNet Spin Glass — Project Manifesto

**Status: live build, in progress**
**Last updated:** 2026-04-11

This document is the running narrative of the BitNet Spin Glass project. It captures what the project is, what we've done, what we've learned, what's running, what's broken, and what comes next. It is meant to be readable by a future collaborator (or future-us) who has not seen the day-to-day decisions.

---

## 1. The Big Idea

Train a neural network → minimize a loss function.
Find a magnet's ground state → minimize an energy.

These problems share a common mathematical skeleton. When the neural network's weights are constrained to {-1, 0, +1} — as in Microsoft's BitNet b1.58 architecture — the analogy stops being a metaphor and becomes nearly exact. Each weight is a literal spin on a lattice, with 0 acting as a "vacancy" (missing site).

This project asks a precise question:

> **Do trained ternary neural networks exhibit measurable signatures of spin glass physics?**

If yes, the entire toolbox of disordered systems physics — replica theory, Edwards-Anderson order parameter, ultrametric solution structure, aging dynamics, Parisi's replica symmetry breaking — becomes available for understanding how ternary networks learn and generalize. This would be a genuinely new bridge between statistical physics and the deep learning of low-bit models.

The goal is to be the first published measurement of spin glass observables on a production-scale ternary LLM.

---

## 2. Project Phases — Roadmap

The project is organized into phases. Each phase has a clear deliverable and standalone value.

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Static analysis of pretrained BitNet 2B4T (frustration, vacancy, weight distributions) | ✅ **Complete** |
| **Phase 1** | Infrastructure: training loop, model code, datasets, checkpointing | ✅ **Complete** |
| **Phase 2** | Figures + content (blog post, X thread) for Phase 0 results | ✅ **Complete** |
| **Phase 3** | Replica training: multiple seeds of small ternary MLPs to enable physics observables | 🔄 **In progress (paused)** |
| **Phase 4** | Hungarian alignment of replicas (resolves permutation symmetry between independent runs) | ⏸️ Not started |
| **Phase 5** | Physics observables: Edwards-Anderson, overlaps, ultrametric, Binder cumulant, susceptibility, autocorrelation | ⏸️ Not started |
| **Phase 6** | Learning rate sweep (temperature analog) — looking for a phase transition | ⏸️ Not started |
| **Phase 7** | Scale up: extend to all model sizes (XS, S, M, L, XL) and additional datasets (CIFAR-10, random labels) | ⏸️ Not started |
| **Phase 8** | Comparison: bridge from small replicas back to BitNet 2B4T's 211 layers | ⏸️ Not started |
| **Phase 9** | Write-up and submission to arXiv / journal | ⏸️ Not started |

---

## 3. What We've Done — Phase 0 Results

### 3.1 What we measured

We loaded Microsoft's `bitnet-b1.58-2B-4T-bf16` model (2 billion parameters, 30 transformer layers) and computed for **all 211 weight matrices**:

- **Frustration index** — fraction of elementary plaquettes (4-cycles) in the signed weight graph that cannot all be satisfied simultaneously
- **Random baseline frustration** — same quantity computed on 5,000 random ternary matrices matched to each layer's vacancy fraction
- **Vacancy fraction** — percentage of zero weights (empty sites in the spin lattice)
- **Weight distribution** — exact balance between -1, 0, and +1 values
- **Frustration vs random** — the difference (a negative number means training reduced frustration below chance)

All computation was done on **CPU only**, no GPU required. The full run took approximately **3 days** on a 32-core machine.

### 3.2 Headline results

- **211 out of 211 layers** show frustration *below* the random ternary baseline. Universal. No exceptions.
- **Mean frustration:** 0.4619 (trained) vs **0.4843** (random baseline) — a difference of -0.0224
- **Effect size: Cohen's d = -1.02** (large)
- **Mean vacancy:** 12.8% across all layers (highly variable: 3.7% to 67.6%)

### 3.3 Layer-type hierarchy

Breaking the 211 layers down by component reveals a striking pattern:

| Layer Type | Mean Frustration | Diff vs Random | Mean Vacancy |
|-----------|------------------|----------------|--------------|
| `embed_tokens` | **0.304** | **-0.185** | **53.2%** |
| `k_proj` (attention keys) | 0.422 | -0.056 | 9.2% |
| `q_proj` (attention queries) | 0.451 | -0.034 | 20.9% |
| `gate_proj` (MLP) | 0.469 | -0.019 | 13.8% |
| `up_proj` (MLP) | 0.473 | -0.015 | 14.9% |
| `o_proj` (attention out) | 0.475 | -0.010 | 12.0% |
| `down_proj` (MLP) | 0.479 | -0.009 | 11.2% |
| `v_proj` (attention values) | 0.471 | -0.008 | 6.1% |

**Interpretation:** The attention key matrices show **6x more frustration reduction** than MLP down-projections. The token embedding is in a different regime entirely — half its weights are zero and its frustration is dramatically below baseline. This suggests:
- Different transformer components experience qualitatively different optimization pressures during training
- Quantization strategies should perhaps treat key matrices and embeddings specially
- The "spin glass character" of a layer depends on its functional role

### 3.4 Vacancy structure

The U-shaped vacancy profile is itself a finding: the network has learned a *strategic* sparsity pattern.

- **Embedding layer:** 53% zeros
- **Early attention layers (0-5):** 15-68% zeros (highly variable)
- **Middle/late layers (6-28):** 4-15% zeros (dense)
- **Final layers (29):** vacancy spikes back up, especially `o_proj` at 58%

This is inconsistent with random pruning — the model has learned *where* to zero things out. The boundaries (where it interfaces with discrete tokens) are diluted; the core computation runs dense.

### 3.5 Statistical robustness

- 5,000 gauge trials per layer for the frustration computation (high precision)
- The frustration vs random baseline test is layer-by-layer, not aggregated, so it's robust to outliers
- Cohen's d of -1.02 is a large effect by any standard
- Every single layer agrees in sign (211/211), making the universal direction undeniable

### 3.6 Outputs

- **Raw data:** `results/bitnet2b/layer_analysis.json` (committed)
- **Run log:** `results/bitnet2b/run.log` (committed)
- **Figures:**
  - `figures/bitnet2b_frustration_profile.png` — trained vs random side by side
  - `figures/bitnet2b_frustration_difference.png` — the money shot, all 211 bars below zero
  - `figures/bitnet2b_vacancy_profile.png` — zero-weight fraction by layer
  - `figures/bitnet2b_weight_distribution.png` — {-1, 0, +1} fractions stacked
  - `figures/bitnet2b_summary_dashboard.png` — 4-panel overview
- **Content:**
  - `content/blog_bitnet2b_spinglass.md` — full blog post (~1,400 words)
  - `content/x_thread_bitnet2b.md` — 5-tweet thread with image attachment guide

---

## 4. Phase 3 — Replica Training (In Progress, Paused)

### 4.1 Why replicas matter

The Phase 0 results show *frustration is reduced*, but they don't tell us *what kind* of energy landscape produced that reduction. To answer that, we need many independently trained models on the same problem — what physicists call **replicas**. The statistical structure of the differences between replicas is what reveals spin glass character.

The original plan (in `scripts/run_replicas.sh`) trains a full grid:
- **5 model sizes:** XS (110K), S (260K), M (530K), L (790K), XL (7M params)
- **2 weight modes:** fp (full precision baseline), ternary (BitNet-like)
- **2 activations:** relu (practical), sign (Ising-exact analog)
- **3 datasets:** mnist, cifar10, random_labels_mnist
- **Replica counts:** 100 for M (the primary), 30 for the others
- **Total: 2,640 training runs**

Estimated full run: ~3 days wall time at 8-way parallelism (per the original benchmark).

### 4.2 What we actually started

To validate the pipeline before committing to the full grid, we launched a **minimal viable run**:
- M size, ternary weights, sign activation, MNIST, 30 replicas
- 200 epochs each

### 4.3 What happened — the parallelism trap

**The benchmark lied.** A single-job benchmark predicted ~10 minutes per replica. Reality was ~1 day per replica. The cause: **CPU oversubscription.**

We have 32 cores. The benchmark used all 32. When we launched 8 parallel jobs, each PyTorch job *also* spawned threads for all 32 cores via OpenMP — yielding **256 threads contending for 32 cores**. Load average: ~115. The CPUs spent more time context-switching than computing.

This is a classic mistake in CPU-bound parallel ML work and we should have caught it before launching. The fix is either:
1. Reduce parallelism (e.g. `-P 4` or `-P 1`) so each job has more cores
2. Set `OMP_NUM_THREADS=4` so each job confines itself, then `-P 8` works correctly

### 4.4 Current state of the data on disk

We killed the run at the point of recognition. On disk:

| Replica | Status | Checkpoints |
|---------|--------|-------------|
| seeds 0-7 | ✅ Fully complete | 29/29 each |
| seeds 8-15 | ⚠️ Partial (in progress when killed) | 25-26/29 each |
| seeds 16-29 | ❌ Not started | 0 |

**Net usable for analysis:** 16 replicas (8 fully complete, 8 partial-but-mostly-there). The partial ones may be useful for snapshot analysis but are missing the final epoch states.

### 4.5 Glassy training dynamics observed

One concrete observation worth highlighting: many of the M-size ternary+sign replicas exhibit **non-monotonic glassy training dynamics**. From the run log of seed 5:

```
Epoch 1:  train acc 87.96%   → looks fine
Epoch 2:  train acc 84.06%   → still fine
Epoch 3:  train acc 28.63%   → collapse
Epoch 4-50: train acc ~10%   → STUCK at random chance for ~50 epochs
Epoch 60: train acc 92.71%   → suddenly breaks through
Epoch 200: train acc 99.51%, test acc 97.07%
```

This is the kind of long-plateau-then-breakthrough dynamic that is characteristic of glassy systems navigating rugged energy landscapes. It is *not* what you see when training a vanilla full-precision ReLU MLP. This alone is suggestive — and it's something the planned autocorrelation analysis can quantify directly.

---

## 5. What We've Figured Out So Far

Synthesizing across both phases:

1. **Trained ternary networks are not random.** Phase 0 establishes this for a 2B-parameter production model. The frustration reduction is universal across 211 layers and statistically robust.

2. **Different transformer components carry different spin-glass signatures.** Attention keys are most affected by training; MLP down-projections least. The embedding layer is its own regime entirely. This is novel and immediately suggests practical implications for ternary quantization strategy.

3. **Sparsity is structured, not uniform.** The U-shaped vacancy profile shows the network learned a deliberate dilution pattern.

4. **Training dynamics of small ternary MLPs are visibly glassy.** Long plateaus followed by sudden breakthroughs are reminiscent of replica symmetry breaking transitions in spin glasses.

5. **CPU benchmarks for parallel ML jobs are easy to get wrong.** Always set `OMP_NUM_THREADS` explicitly when running parallel PyTorch on CPU. (Lesson learned the hard way.)

---

## 6. What We Have NOT Figured Out (Yet)

To be honest about scientific status, here is what remains unproven:

- **Whether there is replica symmetry breaking.** Phase 0 shows frustration is reduced, but this doesn't necessarily mean the solution space has the hierarchical, ultrametric structure that defines a true spin glass phase. We need overlap distributions to know.
- **Whether the Edwards-Anderson order parameter is nonzero.** This would establish frozen spins.
- **Whether there is a phase transition** in the learning rate / temperature analog. Phase 6 (LR sweep with Binder cumulant) would test this.
- **Whether these signatures scale with model size.** Phase 0 only looked at 2B; Phase 3+ on small replicas will tell us if the same physics survives at smaller scales.
- **Whether frustration correlates with model performance.** A really tight result would link the physics observables to predictive metrics like loss or downstream accuracy.

What we have right now is **a strong hint that something is real**, plus a concrete pipeline for testing it rigorously.

---

## 7. How Close Are We to Publication?

Honest assessment:

- **Blog post / X thread:** ✅ ready now
- **Workshop paper / preprint on Phase 0 alone:** Possible but thin. The frustration result is universal and clean, but a single observable on a single model is a small story.
- **Full paper:** Not yet. Needs Phase 3-5 (replicas + Edwards-Anderson + overlap distributions) at minimum. Ideally Phase 6 (phase transition) and Phase 7 (scaling).
- **High-impact paper (e.g. ICML, Nature ML):** Needs all of the above plus a connection to actual model performance, plus replication on at least one other ternary model architecture.

**Distance to a defensible preprint:** ~2-4 weeks of compute and analysis assuming Phase 3 finishes cleanly and the Edwards-Anderson signal is positive.

**Distance to a strong paper:** ~2-3 months including additional experiments, ablations, and writing.

---

## 8. Immediate Next Steps

In priority order:

### 8.1 Resume Phase 3 with correct parallelism

Kill is already done. Re-launch the missing replicas (seeds 8-29 + redo any incomplete) using `-P 2` or `-P 1` so each PyTorch job has enough cores. Estimated 2-4 hours wall time vs. the 3-4 days we were on track for.

```bash
# Example: re-run seeds 8-29 sequentially
for SEED in $(seq 8 29); do
  python -m src.train --size M --weight ternary --activation sign --dataset mnist --seed $SEED
done
```

### 8.2 Phase 4 — Hungarian alignment

Once we have 30 trained replicas, they need to be aligned against each other to remove the permutation symmetry on hidden units. Each hidden unit in replica A could correspond to a different hidden unit in replica B. Hungarian matching solves this so that "weight overlap" is meaningful. Code is ready in the codebase.

### 8.3 Phase 5 — Run the physics observables

With aligned replicas in hand, run the existing analysis scripts:
- `analysis/edwards_anderson.py` — Edwards-Anderson order parameter q_EA
- `analysis/overlaps.py` — pairwise overlap distributions P(q)
- `analysis/ultrametric.py` — fraction of triples satisfying ultrametricity
- `analysis/binder_cumulant.py` — Binder cumulant g
- `analysis/susceptibility.py` — spin glass susceptibility χ_SG
- `analysis/autocorrelation.py` — two-time autocorrelation C(t_w + t, t_w) — directly tests for the glassy aging we already saw qualitatively in the training logs

The shape of P(q) is the key diagnostic. A single peak near 0 means replicas find unrelated solutions (paramagnetic). A single peak near 1 means they find the same solution (ferromagnetic / ordered). A *broad* distribution, or a *multi-peaked* distribution, is the spin glass signature.

### 8.4 Decide on full run vs. preliminary

If 16-30 replicas at the M size show interesting overlap structure, scale up to 100 (the original target) and run the full grid (all sizes, all configs). If signal is flat, reconsider — maybe try larger models or longer training first.

### 8.5 Phase 6 — Learning rate sweep

Train M-size replicas at 5 learning rates (1e-4 to 1e-2). LR plays the role of temperature in this analogy. The Binder cumulant should peak at the phase transition if there is one. This is the most physics-paper-shaped experiment we can do.

### 8.6 Backports and improvements

- An optimized batched version of the frustration computation is sitting uncommitted in `analysis/frustration.py`. Should be committed and validated.
- The `generate_content.py` script has placeholder X handles that should be edited before posting.
- Need to install `OMP_NUM_THREADS` discipline into all run scripts to prevent recurrence of the parallelism trap.

---

## 9. Open Questions / Risks

- **Will the small replicas' physics tell us anything about the 2B model?** There's an implicit assumption that the spin glass signatures observed in 530K-parameter MLPs generalize to a 2B-parameter transformer. This is plausible but unproven. The right way to handle this is to do *both* and compare.
- **Are 30 replicas enough?** For overlap distributions, probably yes. For Binder cumulant (which involves 4th moments), probably borderline. The original target of 100 exists for good reason.
- **Compute budget.** All work so far is CPU-only. A GPU would dramatically accelerate Phase 3+ but is not strictly required. If we want to scale to L and XL models, GPU becomes much more attractive.
- **Reproducibility.** Need to make sure seeds, library versions, and CPU-vs-GPU determinism are all locked down before any publication.

---

## 10. Repository Structure

```
BitNet-Spin-Glass/
├── analysis/                       # Physics observable computations
│   ├── bitnet2b_analysis.py        # Phase 0: BitNet 2B4T static analysis
│   ├── plot_bitnet2b.py            # Phase 2: figure generation
│   ├── frustration.py              # Frustration index (signed graph plaquettes)
│   ├── vacancy.py                  # Zero-weight fraction
│   ├── overlaps.py                 # Replica overlap distributions
│   ├── edwards_anderson.py         # Edwards-Anderson order parameter
│   ├── ultrametric.py              # Ultrametricity test
│   ├── binder_cumulant.py          # Binder cumulant for phase transitions
│   ├── susceptibility.py           # Spin glass susceptibility
│   ├── autocorrelation.py          # Two-time autocorrelation / aging
│   ├── functional_diversity.py     # (auxiliary)
│   ├── effect_size.py              # (auxiliary)
│   └── align.py                    # Hungarian alignment of replicas
├── src/                            # Training infrastructure
│   ├── config.py                   # Hyperparameters and model sizes
│   ├── model.py                    # SpinGlassMLP
│   ├── bitlinear.py                # BitNet-style linear layer
│   ├── sign_ste.py                 # Straight-through sign estimator
│   ├── datasets.py                 # MNIST, CIFAR-10, random labels
│   └── train.py                    # Training loop with checkpointing
├── scripts/                        # Pipeline runners
│   ├── run_phase2.sh               # Phase 2: BitNet 2B4T analysis + figures
│   ├── run_replicas.sh             # Phase 3: full replica training grid
│   ├── run_alignment.sh            # Phase 4: align all replicas
│   ├── run_analysis.sh             # Phase 5: physics observables
│   ├── run_lr_sweep.sh             # Phase 6: learning rate sweep
│   ├── run_bitnet2b.sh             # Phase 0 standalone
│   ├── run_content.sh              # Generate blog/X content
│   └── generate_content.py         # Claude API content generator
├── results/
│   ├── bitnet2b/                   # Phase 0 outputs (committed)
│   │   ├── layer_analysis.json
│   │   └── run.log
│   └── checkpoints/                # Phase 3 partial outputs (gitignored)
│       └── M_ternary_sign_mnist_seed{0..15}/
├── figures/                        # Phase 2 figures
│   └── bitnet2b_*.png
├── content/                        # Blog post + X thread
│   ├── blog_bitnet2b_spinglass.md
│   └── x_thread_bitnet2b.md
├── paper/                          # Future write-up location
├── dashboard/                      # Future dashboard location
└── MANIFESTO.md                    # This document
```

---

## 11. Lessons Learned

1. **Always set `OMP_NUM_THREADS` when launching parallel PyTorch CPU jobs.** Otherwise each process tries to use every core, and N parallel jobs gives you N×cores threads contending for cores. We lost ~3 days of compute to this.

2. **Benchmark in the same conditions you'll deploy in.** A single-job benchmark cannot predict the behavior of an N-parallel run. If you plan to launch N jobs, benchmark with N jobs.

3. **Save partial results.** The training loop checkpoints frequently, which means killing the run only loses the most recent ~10-20 epochs of work per replica. The 7 fully completed replicas are still valuable even after the kill.

4. **The frustration result is the kind of clean, universal finding that's worth communicating early** — the blog post and X thread are ready and shouldn't wait for the full paper.

5. **Trust the science but verify the infrastructure.** The physics is sound; the bugs are in the wiring.

---

## 12. Contact and Links

- Repository: https://github.com/parrishcorcoran/BitNet-Spin-Glass
- Model analyzed: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16
- Status of this document: live, updated as project progresses

---

*This manifesto is the project's source of truth for "where are we and what next." Update it after every major milestone.*
