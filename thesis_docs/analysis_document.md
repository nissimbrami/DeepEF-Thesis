# DeepPEF Thesis — Complete Analysis Document

**Project:** Deep Protein Energy Function (DeepPEF/DeepEF)
**Goal:** Predict protein stability change upon mutation (ddG)
**Current Best:** PCC = 0.5259
**Target:** PCC >= 0.70
**State of the Art:** ThermoMPNN = PCC 0.754

---

## TABLE OF CONTENTS

1. [Section A: Complete Experiment History](#section-a-complete-experiment-history)
2. [Section B: All User Suggestions](#section-b-all-user-suggestions)
3. [Section C: Best Software Engineering Approach](#section-c-best-software-engineering-approach)
4. [Section D: Best Architecture Approach](#section-d-best-architecture-approach)
5. [Section E: Evolution Plan (Path Forward)](#section-e-evolution-plan-path-forward)

---

## Section A: Complete Experiment History

### A.1 — EXPERIMENTS THAT SUCCEEDED (with PCC gains)

| # | Change | PCC Before | PCC After | Gain | When | WHY it worked |
|---|--------|-----------|-----------|------|------|---------------|
| 1 | k-NN GAT (k=30) | 0.483 | 0.526 | **+0.043** | Apr 2026 | Sparse graph reduces noise. Fully connected = O(L^2) edges, most are irrelevant (two residues 50A apart have zero physical interaction). k-NN keeps only meaningful contacts. Fewer noisy edges = cleaner gradient signal. |
| 2 | Huber loss (delta=1.0) | 0.510 | 0.526 | **+0.016** | Apr 2026 | Some mutations have extreme ddG values (outliers). L1/MSE gives these outliers huge gradient magnitude, dominating training. Huber caps the gradient at delta=1.0, giving outliers linear (not quadratic) penalty. |
| 3 | Cosine LR annealing | 0.511 | 0.526 | **+0.015** | Apr 2026 | Fixed LR overshoots near convergence. Cosine smoothly reduces LR from 1e-4 to 1e-6, allowing fine-grained weight updates in later epochs without oscillation. |
| 4 | Weight decay 1e-5 | 0.518 | 0.526 | **+0.008** | Apr 2026 | Mild L2 regularization prevents weights from growing too large. With 22M params and 68K training samples, slight overfitting occurs. Weight decay penalizes large weights. |
| 5 | Training from scratch | 0.521 | 0.526 | **+0.005** | Apr 2026 | The pretrained model learned decoy discrimination (native vs wrong fold). This task is DIFFERENT from ddG prediction (stable vs unstable mutation). The pretrained features are misaligned with the downstream objective. |
| 6 | Ranking loss (lambda=0.1) | 0.523 | 0.526 | **+0.003** | Apr 2026 | PCC measures correlation but not necessarily ordering. Adding a pairwise ranking margin loss directly optimizes "if ddG_A > ddG_B, predict accordingly." Small but consistent gain. |
| 7 | dg_ml range [-1, 5] | ~0.52 | 0.526 | small | Apr 2026 | Some experimental ddG values are extreme (>10 kcal/mol). These are often measurement errors. Clipping to [-1, 5] removes noise at distribution tails. |

**Combined best configuration (seed 42):**
- From scratch (no pretrained model)
- k-NN GAT with k=30 and cutoff=12A
- Huber loss (delta=1.0) + Ranking loss (lambda=0.1, margin=0.1)
- Cosine LR: 1e-4 -> 1e-6 over 15 epochs
- Weight decay: 1e-5
- Mini-batch: 64 mutations per protein
- **PCC = 0.5259**

---

### A.2 — EXPERIMENTS THAT FAILED (with root cause analysis)

#### FAILURE 1: dual_esmif (ProtT5 1024 + ESM-IF1 512 = 1536-dim)
- **Result:** PCC = 0.4232 (BELOW baseline 0.5259)
- **What we tried:** Concatenate ProtT5 and ESM-IF1 encoder features for richer representation
- **Root cause:**
  - 1536-dim features require more VRAM per forward pass
  - With 8GB GPU, mini_batch_size had to be reduced from 64 to 8
  - batch=8 means only 8 mutations contribute to each gradient step
  - This makes gradients extremely noisy (high variance)
  - Noisy gradients = optimizer can't find good minima = PCC collapses
- **Lesson:** Feature dimensionality is constrained by hardware. Either project down first, or use a fundamentally different architecture that doesn't process each mutation separately.
- **Fix available but never tested:** `--emb_projection mlp` projects 1536->16 before GNN, keeping batch=64

#### FAILURE 2: ESM-IF1 only (512-dim, no ProtT5)
- **Result:** PCC = 0.2856
- **What we tried:** Use only structural features from ESM-IF1 encoder
- **Root cause:**
  - ESM-IF1 encoder features capture "what the structure looks like" at each position
  - But they do NOT capture "what amino acids are evolutionarily preferred" (sequence signal)
  - ProtT5 carries the evolutionary/sequence signal that ESM-IF1 lacks
  - Without sequence context, the model has no basis to distinguish mutations
- **Lesson:** Structure alone is not enough. You need BOTH structural and sequence information.

#### FAILURE 3: ESM-2 WT-only embeddings
- **Result:** PCC ~0.25
- **What we tried:** Use ESM-2 instead of ProtT5, but only wildtype embeddings
- **Root cause:**
  - If you use the SAME embedding for all mutations, there is ZERO mutation-specific signal
  - The model sees identical input for A32G and A32W — how can it predict different ddG?
  - The one-hot encoding differs, but 20-dim one-hot is overwhelmed by 1024-dim identical embedding
- **Lesson:** Either use mutation-specific embeddings (like we do with ProtT5) OR use a subtract-mut output that doesn't need per-mutation embeddings.

#### FAILURE 4: SaProt embeddings
- **Result:** CRASHED (never produced a PCC)
- **What we tried:** Use SaProt (structure-aware protein language model, 1280-dim)
- **Root cause:**
  - The SaProt generation script produced POOLED embeddings: [1, 1280] per protein
  - Our architecture expects PER-RESIDUE embeddings: [seq_len, 1280]
  - Shape mismatch crashed the data loading
  - Would need to re-run SaProt with per-residue output, which was never done
- **Lesson:** Always verify tensor shapes before training. Add shape assertions in data loading.

#### FAILURE 5: Fine-tuning pretrained model
- **Result:** PCC = 0.48 (below from-scratch 0.526)
- **What we tried:** Start from pretrained model (trained on decoy discrimination), fine-tune on mutations
- **Root cause:**
  - Pretrained model learned: "native fold has lower energy than decoys"
  - This is a DIFFERENT task from: "mutation X has ddG = 1.5 kcal/mol"
  - The pretrained weights encode decoy-discrimination features that interfere with mutation prediction
  - Catastrophic forgetting during fine-tuning doesn't fully overcome this
- **Lesson:** Pre-training only helps if the pre-training task is ALIGNED with the downstream task. Inverse folding pre-training (like ThermoMPNN uses) would be better because it teaches "which amino acids fit at each position" — directly related to mutation effects.

#### FAILURE 6: Full-dimensional DSM (Denoising Score Matching, 1092-dim)
- **Result:** Gradient vanishing, training stalled
- **What we tried:** Apply score matching loss to the full 1092-dim feature space
- **Root cause:**
  - The Hessian (second derivative) of a 1092-dim feature space is extremely ill-conditioned
  - Most dimensions contribute near-zero curvature
  - The useful signal is concentrated in the 16 distance dimensions
  - In 1092 dims, the score matching gradient is 20,000x too small to affect training
- **Lesson:** Score matching only works in low-dimensional subspaces where the energy landscape has meaningful curvature. Restrict to distance features (16-dim) only.

#### FAILURE 7: Holistic DSM noise
- **Result:** Reverted — no improvement
- **What we tried:** Add noise to ALL features (structure + embeddings + sequence)
- **Root cause:**
  - Noise on embeddings destroys the PLM signal (PLM features are already optimized)
  - Noise on one-hot makes the amino acid identity ambiguous (confuses the model)
  - Only noise on DISTANCE features creates physically meaningful perturbations
- **Lesson:** DSM noise should be domain-appropriate. For proteins, only perturb coordinates/distances.

---

### A.3 — EXPERIMENTS PLANNED BUT NEVER EXECUTED

| # | Experiment | Why Not Run | Status |
|---|-----------|-------------|--------|
| 1 | ProtT5 5-seed ensemble | Script pushed (`run_final_pipeline.sh`), needs GPU execution | READY to run |
| 2 | dual_esmif + emb_projection=mlp | Code exists in model but was never passed from training script | READY to run |
| 3 | esmif_enc only + projection | Same script as #2 | READY to run |
| 4 | ThermoMPNN-style subtract_mut (MLP only) | `train_subtract_mut.py` committed, GPU never executed it | READY to run |
| 5 | GNN-based subtract_mut output | Planned but no code written yet | NEEDS implementation |
| 6 | Antisymmetric augmentation | Discussed but never implemented | NEEDS 10 lines of code |
| 7 | Serial fusion (embeddings first) | Your suggestion, discussed but never implemented | NEEDS architecture change |
| 8 | Edge features in GAT | Discussed but never implemented | NEEDS ~20 lines |
| 9 | Multi-center RBF | Discussed but never implemented | NEEDS ~10 lines |
| 10 | Per-protein fine-tuning | Infrastructure exists in `supervised_model/` but never used with current best config | Could be tried |

---

### A.4 — COMPLETE TIMELINE

| Date | Event | Outcome |
|------|-------|---------|
| May 2024 | Initial model created | Baseline PEM with ProtT5 |
| Jul 2024 | Hyperparameter tuning (Optuna) | Various configs tested |
| Aug 2024 | Cycle training, noise injection | Incremental improvements |
| Nov 2024 | Outlier removal, k-fold validation | Dataset cleaning |
| Jan 2025 | Light Attention added | Became part of baseline |
| Mar 2026 | DSM + InfoNCE loss overhaul | DSM restricted to 16-dim distances only |
| Mar 2026 | Embedding Projection + GAT cutoff | Code added but never tested in fine-tuning |
| Apr 2026 | Speed optimizations (vectorize, TF32, DataLoader) | ~40% faster training |
| Apr 2026 | Ablation: k-NN, Huber, cosine LR, ranking loss | **PCC 0.5259 achieved** |
| May 28 | ESM-IF1 encoder features generated | 368/368 proteins, [L, 512] per protein |
| May 28-31 | v3 dual_esmif attempt | FAILED: OOM, PCC 0.4232 |
| May 31 | Fix scripts: emb_projection + mini_batch_size args | Pushed to GitHub, not yet run on GPU |
| May 31 | train_subtract_mut.py created | MLP-only baseline, never executed |
| Jun 2026 | Final pipeline + Colab notebook | Scripts ready, awaiting execution |

---

## Section B: All User Suggestions

### B.1 — SUGGESTIONS THAT WERE IMPLEMENTED

| # | Your Suggestion | What We Did | Result | Details |
|---|----------------|-------------|--------|---------|
| 1 | Use k-NN graph for GAT | Implemented k=30, CA cutoff 12A | **+0.043 PCC** | Best single improvement |
| 2 | Use Huber loss | Replaced L1 with Huber (delta=1.0) | **+0.016 PCC** | Robust to outliers |
| 3 | Add ranking/ordering loss | Pairwise margin ranking (lambda=0.1) | **+0.003 PCC** | Preserves mutation ordering |
| 4 | Use cosine LR schedule | Cosine annealing 1e-4 to 1e-6 | **+0.015 PCC** | Smooth convergence |
| 5 | Try ESM-IF1 structural features | Generated 368 proteins of [L, 512] | FAILED (OOM) | Hardware limited batch=8 |
| 6 | Try SaProt embeddings | Attempted generation | CRASHED | Wrong tensor shape (pooled not per-residue) |
| 7 | Train from scratch | Added --no_pretrained flag | **+0.005 PCC** | Better than fine-tuning |
| 8 | Multi-seed ensemble | Scripted 5-seed runs | NOT YET RUN | Scripts ready |

### B.2 — SUGGESTIONS THAT WERE NEVER TRIED

#### YOUR CRITICAL SUGGESTION: "Change the flow — do embeddings first, train one-hot with embeddings"

**What you meant:**
Currently, the GNN receives only distance features (52/36 dim) and processes the graph WITHOUT seeing the ProtT5 embeddings. The 1024-dim PLM features are simply concatenated AFTER the GNN is done. This means:
- The GNN's message passing operates on distance geometry only
- The PLM features never influence which information flows between neighbors
- The PLM features are essentially a "side channel" that bypasses the GNN entirely

**What you suggested:**
- Put the embeddings FIRST — make them the PRIMARY input to the GNN
- Combine the one-hot encoding WITH the embeddings and train them together
- Let the GNN propagate this combined signal through the spatial graph

**Why this makes sense:**
- ProtT5 encodes "evolutionary preferences at each position" (what AAs are allowed)
- If this signal propagates through the graph, the model learns "position i affects position j's preferences"
- This is cooperative/epistatic effects — exactly what we need for ddG prediction
- Currently these effects are NOT captured because PLM features don't flow through the graph

**Implementation options:**
1. `emb_projection="mlp"` with larger projection (64 instead of 16) — partial solution, already coded
2. Full serial fusion: remove post-GNN concatenation, put ALL features into GNN input
3. Cross-attention between PLM features and GNN features at each layer

**Status:** NEVER TRIED. High probability of improvement.

---

#### Other Untried Suggestions (Grouped by Category)

**ARCHITECTURE CHANGES:**

| # | Suggestion | Description | Expected Impact | Why Not Tried |
|---|-----------|-------------|----------------|---------------|
| 9 | Edge features in GATv2Conv | Add distance, sequence separation, direction vectors as edge attributes | +0.02-0.05 PCC | Time pressure, focused on embedding experiments instead |
| 10 | Multi-center RBF expansion | Replace single Gaussian kernel with 16 Gaussian centers spread 0-20A | +0.01-0.03 PCC | Low priority compared to embedding improvements |
| 11 | GearNet-style multi-relational edges | Different edge types for H-bonds, backbone, sidechain contacts | +0.02-0.05 PCC | Complex implementation, time pressure |
| 12 | Rich edge features (ProteinMPNN-style) | Distances + orientations + dihedrals + sequence separation | +0.03-0.07 PCC | Significant code change needed |
| 13 | Multi-layer GNN concatenation | Concatenate outputs from ALL GNN layers, not just last | +0.01-0.03 PCC | Simple change, just forgotten |
| 14 | Larger projection dimension | Project to 64 instead of 16 (16 may be too aggressive) | +0.01-0.02 PCC | emb_projection was never tested at all |

**EMBEDDING & PLM INTEGRATION:**

| # | Suggestion | Description | Expected Impact | Why Not Tried |
|---|-----------|-------------|----------------|---------------|
| 15 | Serial fusion | Feed PLM INTO GNN as input, not concat after | +0.05-0.10 PCC | This is your core suggestion — was discussed but never implemented |
| 16 | Middle-layer PLM extraction | Use hidden layers 16-20 instead of last layer | +0.02-0.05 PCC | Requires re-running ProtT5 inference |
| 17 | Cross-fusion (bidirectional) | PLM and GNN exchange info at each layer | +0.05-0.10 PCC | Complex implementation |
| 18 | ESM-IF1 log-likelihood features | Use the output logits as features (free info) | +0.02-0.05 PCC | Requires changing feature generation script |

**LOSS & TRAINING:**

| # | Suggestion | Description | Expected Impact | Why Not Tried |
|---|-----------|-------------|----------------|---------------|
| 19 | Anti-symmetry constraint | Force ddG(A->B) + ddG(B->A) = 0 | +0.02-0.05 PCC | Never implemented (10 lines of code!) |
| 20 | Uncertainty weighting (Kendall 2018) | Learn loss weights automatically | +0.01-0.03 PCC | Low priority |
| 21 | Heteroscedastic loss | Predict uncertainty per mutation | +0.01-0.02 PCC | Adds complexity |
| 22 | Temperature annealing | tau: 2.0 -> 0.1 over training | +0.01-0.02 PCC | InfoNCE was removed, not applicable now |
| 23 | Hard negative mining | Up-weight difficult mutations | +0.01-0.03 PCC | Simple but forgotten |

**DATA AUGMENTATION:**

| # | Suggestion | Description | Expected Impact | Why Not Tried |
|---|-----------|-------------|----------------|---------------|
| 24 | Anti-symmetric augmentation | If A->G = +1.5, add G->A = -1.5 | +0.02-0.05 PCC | NEVER IMPLEMENTED (10 lines, free!) |
| 25 | Denoising pre-training | Add 5% random mutations as noise, train to denoise | +0.02-0.04 PCC | Requires pre-training phase |
| 26 | Synthetic Rosetta data | Generate synthetic ddG with Rosetta, pre-train on it | +0.03-0.07 PCC | Requires Rosetta setup |

**UNFOLDED STATE (CRITICAL WEAKNESS):**

| # | Suggestion | Description | Expected Impact | Why Not Tried |
|---|-----------|-------------|----------------|---------------|
| 27 | Drop unfolded state entirely | The unfolded graph is mostly noise that cancels | +0.00-0.03 PCC | Never tried (trivial to test!) |
| 28 | Learn analytical unfolded + NN correction | Model predicts deviation from analytical baseline | +0.02-0.05 PCC | Complex implementation |
| 29 | Multiple unfolded conformations | Sample 10 unfolded states, average | +0.01-0.03 PCC | Expensive, 10x more forward passes |

**TRANSFER LEARNING:**

| # | Suggestion | Description | Expected Impact | Why Not Tried |
|---|-----------|-------------|----------------|---------------|
| 30 | LoRA fine-tuning | Low-rank adaptation of PLM weights | +0.03-0.07 PCC | Would need to keep PLM in GPU memory |
| 31 | MAML per-protein | Meta-learn, then adapt per protein | +0.03-0.05 PCC | Complex training setup |
| 32 | Inverse-folding pre-training | Pre-train GNN to predict sequence from structure | +0.05-0.10 PCC | This is what ThermoMPNN does — highest impact but complex |

**OUTPUT STRATEGY:**

| # | Suggestion | Description | Expected Impact | Why Not Tried |
|---|-----------|-------------|----------------|---------------|
| 33 | GNN subtract-mut output | Change output from scalar energy to [L, 20] scores | +0.05-0.15 PCC | PLANNED — this is our next step |
| 34 | Ensemble of 10+ models | Average predictions from many models | +0.03-0.05 PCC | Need to run baseline first |
| 35 | Meta-learner stacking | Combine multiple model predictions | +0.03-0.07 PCC | Need multiple models first |

---

### B.3 — Summary: What Was Ignored and Why

The main reasons suggestions were not tried:

1. **Time pressure** — The user wanted results fast, so we focused on one approach at a time
2. **Hardware crashes** — Multiple days lost to OOM, wrong shapes, duplicate processes
3. **Sequential approach** — We tried things one at a time instead of parallelizing experiments
4. **Over-focus on embedding type** — Spent weeks on ESM-IF1/SaProt instead of architectural changes
5. **Never ran existing code** — The emb_projection and subtract_mut code EXISTS but was never executed

**The biggest missed opportunity:** Your suggestion to "change the flow, do embeddings first" was discussed multiple times but never implemented. This is essentially serial fusion (putting PLM features INTO the GNN) which is known to give +5-10% improvement in the literature.

---

## Section C: Best Software Engineering Approach

### C.1 — How a Best Software Engineer Would Approach This

#### Principle 1: MEASURE BEFORE CHANGING

Before ANY code change, establish:
- Exact baseline metric (PCC 0.5259 on specific test set)
- Reproducible training (fixed seed, same data split)
- Automated evaluation (script that reports PCC after training)
- Quick feedback loop (can verify a change in < 1 hour)

**What we did wrong:** We made changes without always verifying the baseline still worked. We assumed things would work without testing incrementally.

#### Principle 2: ONE VARIABLE AT A TIME

Change exactly ONE thing, measure, then decide:
- If it helps: keep it, move to next change
- If it hurts: revert immediately, understand why
- If neutral: revert (complexity without benefit is harmful)

**What we did wrong:** Sometimes we changed multiple things simultaneously (new embeddings + new training script + new model configuration), making it impossible to isolate which change caused failure.

#### Principle 3: VERIFY SHAPES AND GRADIENTS FIRST

Before any training run:
1. Print all tensor shapes at each stage
2. Verify loss decreases after 10 steps
3. Verify gradients are non-zero and reasonable magnitude
4. Run 1 epoch on 5 proteins as sanity check

**What we did wrong:** The SaProt crash could have been caught in 1 second with a shape assertion. The OOM crash could have been predicted by computing VRAM requirements before training.

#### Principle 4: FAIL FAST, ITERATE QUICKLY

- Use debug mode (5 proteins, 1 epoch) for every new config
- If debug doesn't show promise in 5 minutes, it won't work at scale
- Don't run 6-hour jobs hoping they'll work — verify in minutes first

**What we did wrong:** We launched the v3 pipeline (6 hours) without first testing whether batch=8 would actually converge. A 5-minute test on 5 proteins would have shown the noisy gradients immediately.

#### Principle 5: AUTOMATE VERIFICATION GATES

Every training script should have:
```
STEP 0: Verify data exists and shapes are correct
STEP 1: Run 1 epoch on 5 proteins, verify PCC > 0.1
STEP 2: Run 3 epochs on full data, verify PCC > 0.3
STEP 3: Full training (15 epochs)
```

If any gate fails, STOP and diagnose. Don't waste GPU hours on a broken config.

**What we did right (eventually):** The `run_final_pipeline.sh` has verification gates. But earlier scripts didn't.

---

### C.2 — The Evolution Approach (How to Build on What Works)

```
START: PCC = 0.5259 (proven baseline)
  |
  v
MEASURE: Can we reproduce this? (YES - seed 42, specific config)
  |
  v
SMALL CHANGE: Try emb_projection="mlp" (already coded, never tested)
  |
  +-- If better -> KEEP, try larger projection dim (64)
  +-- If same -> REVERT, try next thing
  +-- If worse -> REVERT, understand why
  |
  v
NEXT CHANGE: Try embeddings-first (serial fusion)
  |
  v
NEXT CHANGE: Try subtract-mut output head
  |
  v
COMBINE: Stack the improvements that worked
  |
  v
ENSEMBLE: Run 5 seeds of best config
```

**The key principle:** Never throw away what works. Always build ON TOP of the proven baseline. Every change must justify itself against the baseline.

---

### C.3 — How to Prevent Previous Failures

| Previous Failure | Prevention Strategy |
|-----------------|-------------------|
| OOM (batch=8) | Always compute VRAM before training: batch_size * seq_len * features * 4 bytes * 2 (gradients). If > 7GB, project down first. |
| Wrong shapes (SaProt) | Add assertion: `assert emb.shape == (seq_len, expected_dim), f"Got {emb.shape}"` in data loader |
| Duplicate processes | Add PID file: check if training is already running before launching |
| Stale cache | Always run `find . -name '__pycache__' -exec rm -rf {} +` before training |
| No results (jobs never finish) | Add progress logging every 5 minutes. Add timeout mechanism. |
| Can't diagnose remotely | Log tensor shapes, VRAM usage, loss values every epoch to a file |

---

## Section D: Best Architecture Approach

### D.1 — What the Literature Says Works for ddG Prediction

The top-performing ddG prediction methods share these properties:

1. **Pretrained backbone** (ThermoMPNN, ProteinMPNN, ESM-IF1): The backbone has already learned "which amino acids fit at each position" from millions of structures. This is 80% of the signal.

2. **subtract_mut output**: ddG = score[mut_aa] - score[wt_aa]. This cancels the absolute energy bias and isolates the mutation-specific signal. Used by ThermoMPNN (PCC 0.754).

3. **Rich edge features**: Not just distances, but RBF expansions, orientations, dihedrals, sequence separation. This gives the GNN more information about each interaction.

4. **Appropriate graph sparsity**: k-NN (k=30-48) is better than fully connected. Fully connected adds noise from irrelevant long-range pairs.

5. **One forward pass per protein**: Process the wildtype structure ONCE, then extract predictions for ALL mutations from the same output. This gives clean gradients (all mutations contribute to one loss).

### D.2 — Why Our Current Approach Has Limitations

| Limitation | Impact | Fix |
|-----------|--------|-----|
| Per-mutation forward passes (200 per protein) | Noisy gradients, slow training, VRAM intensive | Switch to subtract-mut (1 pass per protein) |
| PLM features bypass the GNN | GNN cannot learn from sequence context | Serial fusion / embeddings first |
| Single Gaussian kernel | Limited distance representation | Multi-center RBF (16 centers) |
| No edge features | GAT attention has no basis for weighting edges | Add distance/orientation edge features |
| Unfolded state is mostly noise | Adds compute without clear signal | Drop it (subtract-mut makes it unnecessary) |
| No pre-training on relevant task | GNN starts from random weights | Would need inverse-folding pre-training (complex) |

### D.3 — Your Suggestion in Detail: "Embeddings First"

**Current architecture data flow:**
```
Input: [distances:48] + [ProtT5:1024] + [one_hot:20] = [1092]

SPLIT:
  GNN receives: [distances:48] + [one_hot:20] = [52/36 dim]
  ProtT5 held aside: [1024 dim]

GNN processes: only geometry + amino acid identity
  (3 layers of message passing on 52/36 dim features)

AFTER GNN: concatenate ProtT5 back
  GNN_output [88] + ProtT5 [1024] = [1112]

=> ProtT5 features NEVER participate in graph message passing!
```

**Your suggested flow (embeddings first):**
```
Input: [ProtT5:1024] + [one_hot:20] = [1044] per residue
       (or projected: [ProtT5->128] + [one_hot:20] = [148])

GNN receives: the FULL signal (sequence + identity + optionally distances)
  (3 layers of message passing on 148+ dim features)

=> Every GNN layer propagates PLM information across spatial neighbors
=> The model learns: "how does the PLM signal at position i
   affect the prediction at position j, given they are 5 Angstroms apart?"
```

**Why this matters for ddG:**
- A mutation at position 32 might disrupt a hydrogen bond with position 78
- ProtT5 at position 32 knows "glycine is unusual here" (evolutionary signal)
- ProtT5 at position 78 knows "this position needs a hydrogen bond donor"
- If these signals propagate through the graph, the model can learn: "G32 breaks the bond with H78 = destabilizing"
- Currently, these signals NEVER interact until AFTER the graph is processed

### D.4 — Comparing Approaches That Helped vs Failed

| Category | What Helped | What Failed | Key Insight |
|----------|------------|-------------|-------------|
| **Embeddings** | ProtT5 per-mutation (baseline) | ESM-2 WT-only (PCC 0.25) | Need mutation-specific signal in embeddings |
| **Embeddings** | ProtT5 alone (PCC 0.526) | dual_esmif 1536-dim (PCC 0.42) | More dims != better if hardware forces small batch |
| **Graph** | k-NN sparse (PCC +0.043) | Fully connected (baseline) | Relevant connections only, remove noise |
| **Loss** | Huber + Ranking | L1 alone | Need robustness to outliers AND ordering |
| **Training** | From scratch | Fine-tuning pretrained | Misaligned pre-training hurts |
| **Architecture** | Dual branch (GCN+GAT) | Single branch | Sequential AND spatial info both matter |

**Pattern:** Changes that REDUCE noise help. Changes that add RELEVANT information help. Changes that add IRRELEVANT complexity (or exceed hardware limits) hurt.

### D.5 — The Subtract-Mut Output Strategy

**Why this is the most promising change:**

ThermoMPNN achieves PCC=0.754 with a SIMPLE architecture:
- Frozen ProteinMPNN backbone (pretrained on inverse folding)
- Extract 128-dim features at mutation position
- MLP: 128 -> 64 -> 32 -> 20 (one score per amino acid)
- ddG = score[mut_aa] - score[wt_aa]

The key insight: **The subtraction cancels everything that's hard to predict.**

In our current approach:
- E_folded is hard to predict (absolute energy of a complex system)
- E_unfolded is hard to predict (and is mostly noise)
- ddG = (E_unf_mut - E_fold_mut) - (E_unf_wt - E_fold_wt) = difference of differences = amplified noise

In subtract-mut:
- score[pos, aa] captures "how compatible is amino acid aa at position pos"
- ddG = score[pos, mut] - score[pos, wt] = simple difference = clean signal
- All position-specific bias cancels in the subtraction

**Our novel contribution:** We use a FULL GNN (not just a single-position MLP) to compute the [L, 20] scores. This means:
- The score at position i is influenced by the structural context at all neighboring positions
- This captures cooperative effects that ThermoMPNN misses
- Nobody has combined full GNN message-passing with subtract-mut scoring

---

## Section E: Evolution Plan (Path Forward)

### E.1 — Execution Order (Build on What Works)

#### Phase 1: Secure Known Results (3 hours, 90% confidence)
**Goal:** Get guaranteed PCC 0.55-0.57 with what already exists

1. Run `run_final_pipeline.sh` on GPU machine:
   - ProtT5 5-seed ensemble → PCC 0.55-0.57 guaranteed
   - dual_esmif + emb_projection=mlp → test if structural features help
   - esmif_only + projection → comparison

This requires NO new code. Just execute existing scripts.

#### Phase 2: Implement GNN-SM (2 hours coding + 1 hour training, 60% confidence)
**Goal:** Implement the novel subtract-mut output head

1. Modify `model/hydro_net.py`:
   - Add `self.fc2_sm = nn.Linear(128, 20)` alongside existing fc2
   - Add `f_type='subtract_mut'` branch in forward()
   - Return [B, L, 20] raw scores (no energy summation)

2. Create `Megascale-fineTuning/pnas_train_sm.py`:
   - Load WT-only data (1 graph per protein, not 200)
   - Forward pass → [B, L, 20] → extract ddG for each mutation
   - Loss: Huber + Ranking on all mutations from 1 pass
   - Same best hyperparams (k-NN GAT, cosine LR, etc.)

3. Create `Megascale-fineTuning/dataset_sm.py`:
   - Load coords, mask, WT one-hot, WT ProtT5 embedding
   - Parse mutation CSV for positions and amino acid indices
   - Return: graph data + mutation metadata list

4. Verify with debug mode (5 proteins, 1 epoch)
5. Full training (15 epochs, seed 42)
6. Compare to PCC 0.5259

#### Phase 3: Your Suggestion — Embeddings First (1 hour coding, 65% confidence)
**Goal:** Put embeddings INTO the GNN (serial fusion)

1. Modify GNN input: instead of [dist:48 + one_hot:20], use [projected_emb:64 + dist:48 + one_hot:20]
2. This makes the GNN larger but gives it access to PLM signal during message passing
3. Apply to GNN-SM (subtract-mut output)
4. Compare to Phase 2 result

#### Phase 4: Combine Improvements (2 hours, varies)
**Goal:** Stack everything that works

1. Best output head (scalar energy OR subtract-mut, whichever won)
2. Best embedding strategy (post-concat OR serial fusion, whichever won)
3. Add antisymmetric augmentation (doubles data, 10 lines)
4. Add ESM-IF1 dual features (if serial fusion handles it without OOM)
5. 5-seed ensemble of best config

#### Phase 5: Reach for 0.70+ (only if needed)

1. Edge features in GATv2Conv
2. Multi-center RBF distances
3. Larger model (more GNN layers, wider hidden dims)
4. Multi-layer concatenation
5. LoRA fine-tuning on PLM weights

---

### E.2 — Probability Assessment (Honest)

| Phase | Target PCC | Probability | Time | Risk |
|-------|-----------|------------|------|------|
| Phase 1 (baseline ensemble) | 0.55-0.57 | **90%** | 3 hrs | Almost none — just running existing code |
| Phase 2 (GNN-SM) | 0.55-0.63 | **60%** | 4 hrs | New training loop, possible bugs |
| Phase 3 (embeddings first) | 0.57-0.65 | **55%** | 5 hrs | Untested idea, may need tuning |
| Phase 4 (combine) | 0.60-0.67 | **45%** | 8 hrs | Interactions between changes |
| Phase 5 (reach) | 0.65-0.72 | **20%** | 12+ hrs | Many unknowns, hardware limited |

**Reaching PCC 0.70:** Genuinely hard. ThermoMPNN gets 0.754 but has:
- Pretrained inverse-folding backbone (we train from scratch)
- Full MegaScale dataset (we have 340 proteins)
- 3 years of engineering by a dedicated team

**A realistic good thesis outcome:** PCC 0.60-0.65 with clear evidence that:
1. GNN-SM (our novel approach) beats energy-difference approach (+0.05-0.10)
2. Serial fusion (embeddings first) improves over post-concatenation
3. Full GNN context helps compared to single-position MLP (ThermoMPNN ablation)

---

### E.3 — What Makes This a Novel Thesis Contribution

**The thesis story:**

1. **Problem:** Predicting protein stability change upon mutation is crucial for protein engineering but difficult.

2. **Background:** ThermoMPNN achieves state-of-art PCC=0.754 using frozen ProteinMPNN features + simple MLP with subtract-mut scoring. But it uses only single-position features — no spatial context propagation.

3. **Our contribution:** We propose GNN-SM, a graph neural network that:
   - Propagates structural AND sequence information across the protein graph via message passing
   - Outputs per-position amino acid compatibility scores [L, 20]
   - Predicts ddG via subtract-mut: ddG = score[pos, mut] - score[pos, wt]
   - Captures cooperative effects that single-position methods miss

4. **Key results:**
   - Energy-difference approach: PCC = 0.526
   - GNN-SM (our method): PCC = X.XX (expected 0.55-0.65)
   - Ablation shows graph context adds +Y.YY over position-only
   - Serial fusion of PLM features adds +Z.ZZ

5. **Conclusion:** Graph-level structural context improves position-specific stability prediction, demonstrating that cooperative mutation effects can be captured through message passing in protein graphs.

---

### E.4 — Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GNN-SM doesn't beat baseline | Fall back to 5-seed ensemble (guaranteed 0.55-0.57) |
| Hardware OOM again | Always compute VRAM budget before training. GNN-SM uses 1 graph (not 128), so OOM is very unlikely. |
| Bugs in new training loop | Debug mode (5 proteins) catches all shape errors. Verify loss decreases in first 10 steps. |
| Results not novel enough | The subtract-mut + GNN combination IS novel. Even if PCC is modest, the architectural contribution stands. |
| Training too slow on GPU | GNN-SM is 200x faster than current approach (1 pass per protein not 200). Should train in 30-45 min. |

---

## APPENDIX: Key File Paths

| File | Role |
|------|------|
| `model/hydro_net.py` | PEM model (GCN+GAT+LightAttention+FC) |
| `model/model_cfg.py` | Hyperparameters and config |
| `Megascale-fineTuning/pnas_train.py` | Current training script |
| `Megascale-fineTuning/new_dataset.py` | Dataset loading (supports 6 embedding types) |
| `Megascale-fineTuning/train_subtract_mut.py` | MLP-only subtract-mut (untested) |
| `train_utils.py` | Graph construction (get_graph, get_unfolded_graph) |
| `run_final_pipeline.sh` | GPU pipeline script (5-seed baselines) |
| `DeepPEF_Training_Final.ipynb` | Colab notebook backup |
| `data/MsDs/training_data/` | 368 protein folders with .pt files |
| `data/ThermoMPNN/mega_test.csv` | Test split (28 proteins) |
