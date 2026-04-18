# DeepEF — Recent Changes Summary

## Committed Changes

### 1. Repo Cleanup (Mar 23, `0a89a3b`)
Removed dead files, fixed project structure, and updated `.gitignore`.

**Why:** The repository had accumulated unused scripts and artifacts from earlier experiments, making it harder to navigate and understand the active codebase. Cleaning up reduces confusion and establishes a clear baseline for the new training approach.

---

### 2. Replace Loss Functions with InfoNCE + DSM (Mar 24, `f9f704a`)

**Loss function overhaul:**
- Replaced the previous heuristic ranking loss with **InfoNCE (Boltzmann contrastive loss)**. The new loss computes:
  - *Primary term:* native folded energy should be lowest among all states (folded, unfolded, sequence-decoy, structure-decoy, cycle permutations)
  - *Secondary term:* native unfolded energy should be lower than decoy structure energy
- Replaced the explicit gradient penalty with **Denoising Score Matching (DSM)** as the regularizer — the model learns that the energy gradient at native structures should point "downhill" toward the native state.

**Why:** The previous loss used hand-tuned margin-based ranking terms that were fragile and didn't generalize well. InfoNCE is a principled contrastive objective grounded in statistical mechanics (Boltzmann distribution) — it naturally enforces a complete energy ranking without needing per-pair margins. DSM replaces the gradient penalty because it provides a learning signal about the *shape* of the energy landscape (the score function), not just the gradient magnitude. This encourages the model to learn a physically meaningful energy surface where native structures sit in energy minima.

**Training infrastructure:**
- Added **MPS (Apple Silicon) device support** — autocast, cache clearing, and GradScaler are now device-aware.
- Added **`compute_metrics()`** — tracks ranking accuracy (e.g., % of samples where E_native < E_decoy) and energy gap statistics during training and validation.
- Added per-step wandb logging of all energy values, ranking metrics, gradient norms, and DSM loss.

**Why:** MPS support enables local development and debugging on Mac without needing GPU cluster access. The new metrics (ranking accuracy, energy gaps) give direct visibility into whether the model is learning the correct energy ordering — something that was previously only inferable indirectly from the loss value.

---

### 3. Overhaul DSM to Holistic Noise + Sigma Tuning (Mar 25, `76503f5`)

- Rewrote DSM to noise **all feature components** (structure, embeddings, sequence) with per-component noise levels, then averaged the per-component losses equally.
- Tuned `sigma = 0.5` for the DSM noise scale.
- Added diagnostic plots for DSM behavior.

**Why:** The initial DSM implementation only noised the distance features, which meant the model only learned score matching in a 16-dimensional subspace. A holistic approach that corrupts structure, embeddings, and sequence features ensures the energy landscape is well-shaped across all input dimensions. Equal-weight averaging prevents the 1024-dim ProtT5 embeddings from dominating the loss.

**Supporting figure — `figures/dsm_diagnostic_plots.png`:**

![DSM Diagnostics](figures/dsm_diagnostic_plots.png)

This 6-panel diagnostic reveals *why* naive full-dimensional DSM fails:
- **Top-left:** Model gradient vs target score shows a 20,000× mismatch — the network can't produce gradients large enough to match DSM targets at 1092 dims.
- **Top-center:** DSM learning rate (% improvement) drops to zero beyond ~200 feature dimensions — at 1092 dims the loss is completely flat.
- **Top-right:** Parameter gradient norms from DSM are ~100,000× smaller than from the direct energy backward pass, meaning DSM contributes essentially nothing to learning.
- **Bottom-left:** Learning curves by dimension count — 16 dims learns well, 1092 dims stays flat.
- **Bottom-center:** Signal flow through DSM shows second-order gradients (Hessian) vanish exponentially with depth.
- **Bottom-right:** Confirms the concept is sound at low dims — DSM loss decreases steadily when restricted to 16 distance features.

---

---

### 13. Repo Cleanup (Apr 12, `cf97f2c`)

- Renamed `train-2_5_light_att.py` → `train.py`; updated all references in `DeePEF_train.sh`, `run_experiments.sh`, `CLAUDE.md`
- Moved `test_dsm_*.py` to `tests/` directory
- Fixed `.gitignore`: corrected `trianed_models` typo (×4), added `logs/`, `*.log`, scoped image ignores to `experiments/` and `figures/` dirs instead of globally
- Updated `CLAUDE.md`: replaced stale `old/` and `sbatch files/` entries with `tests/` and `Megascale-fineTuning/`

---

## Committed Changes (Previously "Uncommitted")

### 4. Embedding Projection into GNN (Mar 26, `da6376e`)

Added **learnable embedding projection modules** that compress ProtT5 embeddings (1024-dim) before feeding them into the GNN layers, rather than concatenating raw 1024-dim vectors after the GNN:

- **`MLPProjection`**: 1024 → 128 (hidden) → 16 (output), with ReLU + dropout
- **`LowRankProjection`**: 1024 → rank (4) → 16, factored linear projection
- Configurable via `--emb_projection` flag: `"none"` (original behavior), `"mlp"`, or `"low_rank"`

**Architectural change:** When projection is enabled, the 16-dim projected embeddings are concatenated into the GNN input features (GCN: 52→68 dims, GAT: 36→52 dims), so the GNN layers can learn joint structure-embedding representations. The raw 1024-dim concat after GNN is removed. When projection is `"none"`, the original architecture is preserved.

**Why:** In the original architecture, ProtT5 embeddings (1024 dims) are concatenated *after* the GNN layers, meaning the GNN only sees structural features and never learns to combine structure with sequence information. By projecting embeddings to a small dimension and feeding them *into* the GNN, the graph layers can learn interactions between local structure and sequence context. The low-rank option (only 4×16 = 64 + 16×16 = 320 parameters) is designed to test whether a minimal projection is sufficient, reducing overfitting risk on our limited training data.

### 5. Distance-Based GAT Edges (Mar 26, `da6376e`)

Added a **distance cutoff for GAT edges** (default 12 Å) using CA atom coordinates, replacing the fully-connected graph:

- `get_edge_index()` now accepts optional `ca_coords` and uses `torch.cdist` to build edges only between residues within the cutoff radius.
- CA coordinates are extracted before graph construction and passed through the training/validation pipeline.
- Cutoff is configurable via `CFG.gat_cutoff` (set to `None` for fully-connected fallback).

**Why:** The fully-connected GAT graph creates O(N²) edges for a protein of length N, which is both memory-expensive and physically unrealistic — residues 50 Å apart have negligible non-bonded interactions. A 12 Å cutoff roughly captures the first and second coordination shells of amino acid contacts, focusing attention on physically relevant pairwise interactions. This should reduce memory usage (enabling longer proteins) and may improve generalization by removing spurious long-range edges.

### 6. DSM Reverted to Distance-Only (D-Only) (Mar 26, `da6376e`)

Reverted DSM from holistic noise back to **noising only the 16 distance features (D)**, with the rationale documented in the code:

1. Only 16 dims → second-order gradient (Hessian) signal survives through the deep network
2. Distance features carry ~47% of the energy sensitivity
3. Physically meaningful — enforces ∂E/∂D ≈ 0 at native distances

**Why:** The holistic DSM (noising all 1092 dims) was found to produce vanishing gradients through the network. The Hessian (second-order gradient needed for DSM's `grad(grad(E))` computation) decays exponentially with input dimension in deep networks. By restricting to the 16 most energy-sensitive dimensions, the DSM signal remains strong enough to shape the energy landscape. This is a pragmatic compromise — we enforce smoothness where it matters most (distance features that define the protein's 3D structure).

**Supporting figure — `figures/ssm_vs_dsm_test.png`:**

![SSM vs DSM](figures/ssm_vs_dsm_test.png)

Comparison of Sliced Score Matching (SSM) variants vs full DSM vs D-only DSM on the actual PEM model:
- **Top-left (Loss curves):** Full DSM (1092 dims, blue) stays flat — no learning. DSM D-only (16 dims, green) shows steady decrease. SSM variants (K=1,4,8 random projections) also learn but are noisier.
- **Top-right (Param gradient norms):** D-only DSM produces the strongest and most stable parameter gradients. Full DSM gradients are orders of magnitude weaker.
- **Bottom-right (Single-step gradient strength):** Direct energy backward pass provides gradient norm ~53 — DSM D-only achieves 0.003, while full DSM is 0.0003 (10× weaker). SSM K=8 reaches 0.02 but with high variance.

This confirms D-only as the best practical choice: strongest signal, simplest implementation, and physically grounded in distance features.

**Supporting figure — `figures/dsm_alternatives.png`:**

![DSM Alternatives Architecture](figures/dsm_alternatives.png)

Architectural comparison of three alternative approaches we considered before settling on D-only DSM:
1. **Direct Score Prediction Head** — adds a separate `s(x)` head to predict the score directly (avoids second-order gradients entirely, but requires a second output head and changes the model architecture)
2. **Fisher Divergence with Hutchinson Trace** — uses random vector projections to estimate the trace of the Hessian stochastically (avoids explicit Hessian computation, but the Hutchinson estimator diverged in our tests)
3. **Noise Conditional Score Network (NCSN)** — trains a separate score network conditioned on noise level (powerful but requires a separate model, multi-scale noise, and fundamentally changes the training paradigm)

Verdict: Score Head is the most promising alternative for future work; Hutchinson diverged; NCSN is too large a departure from our current architecture.

**Supporting figure — `figures/dsm_alternatives_test.png`:**

![DSM Alternatives Test](figures/dsm_alternatives_test.png)

Empirical test of the alternatives on PEM:
- **Left (Raw loss):** Score Head (blue) and DSM D-only (green) both decrease steadily. Holistic DSM (red, 1092 dims) stays flat. Hutchinson estimator was excluded — it diverged to infinity.
- **Right (Relative change):** Score Head achieves ~30% loss reduction. DSM D-only achieves ~25%. Holistic DSM shows <5% change — confirming the vanishing Hessian problem.

**Supporting figure — `figures/fd_dsm_test.png`:**

![FD vs Autograd DSM](figures/fd_dsm_test.png)

Finite-difference (FD) validation of autograd DSM to rule out implementation bugs:
- **Left (Raw loss):** FD gradient penalty (light blue, high spikes) is noisy but matches autograd in trend. D-only variants (yellow/green) are smooth and low.
- **Right (Relative change):** Confirms autograd and FD produce consistent results — the flat holistic DSM is a genuine signal problem, not a code bug.

### 7. Replace Autograd DSM with Finite-Difference DSM (FD-DSM) (Mar 26, `da6376e`)

Replaced the `create_graph=True` autograd implementation of D-only DSM with a **finite-difference approximation** of the directional derivative:

```
fd_score = (E(x_noisy + ε·v) − E(x_noisy − ε·v)) / (2ε)
target_v = v · (−noise_D / σ²)
lossg    = mean_K[ (fd_score − target_v)² ]
```

where `v` is a random unit vector in the 16-dimensional distance-feature subspace, `ε = 0.1`, and `K = 1` direction per sample.

**Why autograd DSM stopped working:**

The original DSM computed `grad_d = ∂E/∂x_D` using `torch.autograd.grad(..., create_graph=True)`. For the DSM loss to train the model, its backward pass needs the **Hessian** `∂²E/(∂x_D ∂θ)` — the second-order derivative that links the score to model parameters. This Hessian vanishes through deep networks with normalization layers (InstanceNorm makes the model locally linear in distance features, so the second derivative w.r.t. parameters is ≈ 0). Measured directly:

```
|grad_d|   = 0.000027   (model's analytical gradient w.r.t. distance features)
|target_d| = 1.6085     (DSM target score)
ratio      = 0.000017   (gradient is 60,000× too small)
param grad after DSM backward = 0.00000379   (essentially zero)
```

As a result, `lossg` was permanently stuck at `1/σ² = 4.0` — the theoretical value when the model gradient is identically zero — with occasional Hessian explosions to 2000+ that destructively updated parameters without improving the score loss.

**Why FD-DSM works:**

FD-DSM estimates the directional derivative by evaluating the energy at two perturbed inputs (`x ± ε·v`). This is a **first-order** operation — no Hessian is needed. The backward pass computes `∂(fd_score)/∂θ = ∂E/∂θ` which flows cleanly through the model. Measured on the same protein:

```
autograd DSM:  lossg = 4.0043 → 4.0042 after 1 step  (no learning)
FD-DSM:        lossg = 9.2363 → 4.3482 after 30 steps  (−52.9%)
```

The initial FD-DSM loss exceeds 4.0 because the finite energy difference `(E⁺ − E⁻)/2ε` is not zero at initialization — unlike the analytical gradient which happened to be near-zero. This means there is real signal to drive learning.

**Why ε = 0.1 matters:**

The analytical gradient `∂E/∂x_D` is near-zero because the model is approximately linear at infinitesimally small perturbations (due to normalization). At `ε = 0.1` (10% of the distance feature scale), the perturbation is large enough to probe the nonlinear regime where the model's energy response is non-trivial and actually depends on its weights.

**Cost:** 2 forward passes per sample (for ε·v and −ε·v). With `K = 1` this is comparable to the previous autograd DSM cost. The clamp `lossg = min(lossg, 20.0)` and gradient clipping (`clip_grad_norm = True`, `max_norm = 10.0`) are retained as stability safeguards.

---

### 10. FD-DSM Dropout Fix — Batched Forward Pass in eval() Mode (Mar 28, `8fec666`)

Fixed a subtle bug where E⁺ and E⁻ were computed with **different dropout masks**, corrupting the finite-difference estimate.

**The problem:** `LightAttention` has a hardcoded `conv_dropout=0.25` that is independent of the `dropout_rate` passed to `PEM`. In `model.train()` mode, each forward call samples a fresh dropout mask — so `E(x+εv)` and `E(x-εv)` were evaluating two different stochastic functions. The fd_score `(E⁺ − E⁻)/2ε` was no longer a clean estimate of `v·∇E`; it included a `O(dropout/ε)` noise term that could dominate the signal.

**The fix (two parts):**

1. **`model.eval()` around the FD passes** — disables all dropout (including `LightAttention`'s hardcoded dropout), making E⁺ and E⁻ deterministic evaluations of the same function. Gradients still flow through `eval()` mode; only `torch.no_grad()` stops them.

2. **Single batched forward pass** — instead of two separate `model(...)` calls, `[x+εv, x-εv]` are stacked into a `[2, N, F]` batch and evaluated together. This halves the number of forward passes per direction.

```python
model.eval()
X_pair = torch.stack([X_noisy + epsilon * v, X_noisy - epsilon * v], dim=0)  # [2, N, F]
E = model(X_pair, ca_coords=ca_single.expand(2, -1, -1))
model.train()
fd_score = (E[0] - E[1]) / (2 * epsilon)
```

**Verified:** A convergence test on a single protein for 50 epochs shows loss dropping from ~3.1 → ~0.001 with the fix applied. Without it (dropout active), loss oscillates around the theoretical noise floor (~4.0) and does not decrease.

---

### 11. Gradient-Norm Scaling for DSM vs InfoNCE (Apr 10, `c7ca019`)

Added automatic per-step scaling of the DSM loss so its gradient contribution always matches InfoNCE's, replacing the fixed `reg_alpha` weight.

**The problem:** The unweighted combination `loss = lossd + lossg` let DSM gradients dominate or be drowned out unpredictably. Without scaling, `dsm_alpha` would need to be hand-tuned per experiment. Worse, when the DSM gradient norm is near-zero (small proteins, flat energy landscape early in training), naïve scaling exploded to millions — destabilizing training completely.

**The fix:**

1. **Two backward passes, no third:** `lossd.backward()` and `lossg.backward()` are each run once to measure their gradient norms. The final parameter gradient is then assembled manually as `grad_d + alpha * grad_g` — no third backward needed.
2. **Alpha clamped to [0.01, 10]:** Prevents explosion when `grad_norm_g ≈ 0` while ensuring DSM always contributes at least 1% of InfoNCE's gradient.
3. **Alpha logged per step and averaged per epoch** alongside `lossd`, `lossg`, and `fd_score`.

```python
dsm_alpha = torch.clamp(grad_norm_d / (grad_norm_g + 1e-8), min=0.01, max=10.0).detach()
# Manually combine gradients — no third backward
for n, p in model.named_parameters():
    if p.grad is not None and n in grads_d:
        p.grad.mul_(dsm_alpha).add_(grads_d[n])
```

**Observed:** `dsm_alpha` stabilises around 3–5 across epochs, confirming the two losses are naturally ~3–5x mismatched in gradient norm without scaling.

---

### 12. fd_score Averaged Over All Proteins per Epoch (Apr 10, `c7ca019`)

**Why `fd_score` is a better training signal than `lossg`:**

`lossg = (fd_score - target_v)²` where `target_v = -v·noise/σ²` is resampled randomly every step. Even if the model is learning perfectly, `lossg` will not decrease visibly because the target changes with every sample — it measures "how well did you answer this random question?" when the question changes each time.

`fd_score = (E(x+εv) - E(x-εv)) / 2ε ≈ v·∇E(x_noisy)` measures whether the model's energy function has developed a gradient in the direction of the noise. A model with `∇E ≈ 0` everywhere gives `fd_score ≈ 0`. A model that has learned the score function gives `fd_score ≈ target_v`. So **growing `fd_score` magnitude is the direct signal that DSM is working**, independent of the random noise draw.

Expected `lossg` for a model with a given `fd_score`:
```
E[lossg] = Var(target_v) + Var(fd_score) - 2·Cov(fd_score, target_v)
         ≈ 4  (when fd_score ≈ 0, i.e. early training)
```
`lossg` only starts dropping when `fd_score` reaches the same order of magnitude as `target_v` (~2), which requires much more than 20 epochs on 50 proteins.

**The fix:** `denoising_score_matching()` now returns `avg_fd_score` (the mean `|fd_score|` across K directions), accumulated across all proteins in the epoch and printed in the TRAIN SUMMARY alongside the other metrics. Previously it was logged for one random protein per epoch, making cross-epoch comparison meaningless.

---

### 8. Epoch-Level Training Summaries (Apr 10, `c7ca019`)

Added epoch-end summary printouts showing averaged loss components and ranking metrics for both training and validation, with the embedding projection configuration noted.

**Why:** Per-step metrics are noisy and hard to interpret at a glance. Epoch-level summaries provide a clear snapshot of training progress and make it easy to compare runs with different configurations (e.g., `emb_projection=mlp` vs `emb_projection=none`).

### 9. CLI & Debug Improvements (Apr 10, `c7ca019`)

- Added `--emb_projection`, `--emb_proj_rank`, `--emb_proj_dim`, `--emb_proj_hidden` command-line flags for experiment configuration.
- Debug mode now uses projection-specific output directories and smaller defaults (50 proteins, 10 epochs).
- wandb run names include the projection configuration tag.

**Why:** Enables running multiple embedding projection experiments from a single script via `run_experiments.sh`, with results automatically organized by configuration. Smaller debug defaults speed up local iteration on MPS.

---

## Multi-GPU Training Optimization (Apr 18)

A series of targeted changes to maximize training throughput on the RTX 6000 Ada cluster before adding multi-GPU DDP. The goal was to eliminate CPU/GPU idle time and reduce unnecessary overhead per training step.

---

### 14. Remove Per-Iteration GPU Cache Clearing (Apr 18, `bc39773`)

Removed `torch.cuda.empty_cache()` and `gc.collect()` calls that were running inside the inner training loop on every iteration.

**Why:** `empty_cache()` releases cached (but unused) memory back to the OS — it does not free memory that PyTorch is actively using. Calling it every step forces the CUDA allocator to re-request memory on the next allocation, serializing CPU↔GPU and adding ~10-100 ms of latency per step for no benefit. Similarly, `gc.collect()` at each step is pure overhead. Both were removed from the training loop and the validation loop; `empty_cache()` is still called once at the start of each epoch to reclaim truly idle memory.

---

### 15. Switch Default Embedding Projection to MLP (Apr 18, `433621a`)

Changed `emb_projection = "none"` → `emb_projection = "mlp"` in `model/model_cfg.py`.

**Architectural impact:**

| Mode | LightAttention input | LA params | Total params |
|------|---------------------|-----------|--------------|
| `none` | 1096 dims | 21.6M | ~22M |
| `mlp` | 104 dims | ~195K | ~704K |

With `emb_projection="none"`, the raw 1024-dim ProtT5 embeddings are concatenated after the GNN, making LightAttention `Conv1d(1096, 1096, kernel=9)` — 21.6M parameters in two conv layers alone. With `emb_projection="mlp"`, embeddings are projected 1024→128→16 before the GNN, and the GNN+LA only see 104-dim features — a 31× reduction in LA input width, dropping parameters from 22M to 704K.

**Why:** A 22M-parameter model on 103K training proteins without batching is massively overparameterized and will overfit. The MLP projection also allows the GNN to learn joint structure-sequence representations rather than treating them as independent streams, which is architecturally more principled.

---

### 16. DataLoader Tuning + GradScaler Fix (Apr 18, `6fc4b3f`)

**DataLoader changes (`model/model_cfg.py`, `model/data_loader.py`):**
- `num_workers`: 2 → 8 (parallel CPU prefetch workers)
- Added `persistent_workers = True` — keeps worker processes alive between epochs, avoiding fork/init overhead per epoch
- Added `prefetch_factor = 4` — each worker pre-fetches 4 batches ahead of GPU demand
- `fetch_dataloader()` refactored to use a `loader_kwargs` dict, with guards to disable `persistent_workers` and `prefetch_factor` when `num_workers=0` (debug mode)

**GradScaler fix (`train.py`):**
- Changed deprecated `torch.cuda.amp.GradScaler()` → `torch.amp.GradScaler("cuda")` in `_make_scaler()`
- Changed all `optimizer.zero_grad()` → `optimizer.zero_grad(set_to_none=True)` at 4 call sites — frees gradient memory immediately rather than zeroing in place, reducing memory pressure

**seq_len guard fix (`train_utils.py`):**
- Changed `if seq_decoy.shape[1] > config.seq_len` → `if Xjf.shape[1] > config.seq_len` — the original check used the decoy sequence tensor shape, which is wrong; the graph tensor `Xjf` is the object that actually stresses GPU memory.

**Why:** With 8 CPU workers and prefetch_factor=4, the DataLoader can prepare 32 batches in parallel while the GPU runs the forward pass — hiding protein loading latency behind GPU compute. `set_to_none=True` avoids writing zeros to gradient tensors (memory bandwidth waste). The GradScaler fix eliminates deprecation warnings that pollute logs.

---

### 17. SLURM: Allocate 8 CPU Cores per Task (Apr 18, `2abff7e`)

Added `#SBATCH --cpus-per-task=8` to `DeePEF_train.sh`.

**Why:** SLURM defaults to 1 CPU per task. With `num_workers=8` in the DataLoader, 8 workers compete for 1 CPU core — they serialize instead of running in parallel, negating the entire benefit of multi-worker loading. Allocating 8 cores gives each worker its own core for full parallel prefetch throughput.

---

### 18. Cache Valid Protein List (Apr 18, `50d1c11`)

`filter_corrupt_proteins()` in `model/data_loader.py` now saves and restores the validated protein list using a pickle cache keyed by an MD5 hash of the sorted protein paths.

**The problem:** On each training restart, `filter_corrupt_proteins()` was scanning all ~103K proteins — loading `crd_backbone.pt` and `mask.pt` for every protein to check for NaN coordinates. With ~206K file reads over a network filesystem, this took 20-40 minutes per startup before training began.

**The fix:**
```python
key = hashlib.md5("".join(sorted(self.data_dir)).encode()).hexdigest()[:16]
cache_path = os.path.join(self.data_path, f".valid_proteins_{self.set_type}_{key}.pkl")
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        self.data_dir = pickle.load(f)
    return  # skip scan entirely
```
The hash changes if proteins are added/removed, invalidating the cache automatically. On cache hit, startup is immediate.

---

### 19. Enable TF32 on Ampere/Ada GPUs (Apr 18, `bd32eba`)

Added two lines at module load time in `train.py`:
```python
if CFG.device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

**Why:** RTX 6000 Ada (Ada Lovelace architecture) has dedicated TF32 hardware. TF32 uses the same 8-bit exponent as fp32 but only 10-bit mantissa (vs 23-bit), providing ~1.5–2× matmul throughput at negligible precision loss for neural network training. PyTorch disables TF32 by default since PyTorch 1.12 to avoid surprising precision changes — these two lines opt back in explicitly. The GNN matmuls and LightAttention convolutions are the primary beneficiaries.

---

### 20. Vectorize Distance Matrix Computation — 9 Calls → 2 (Apr 18, `4f38b32`)

Refactored `get_noised_proteins()` in `train.py` to compute the O(N²) distance matrix **once per unique coordinate set** instead of once per graph representation.

**The problem:** The training loop builds 11 graph representations per protein (native folded/unfolded, decoy sequence/structure variants, 6 cycle permutations). Each call to `get_graph()` ran `get_dist_matrix()` — a `torch.cdist` over `[N×4, 3]` coordinates producing an `[N, N, 16]` tensor. With N=450, this is a 450×450×16 matrix computed 9 times per step, 8 of which used identical native coordinates.

**The fix:** Extract a new helper `_build_graph_features(D_base, mask, emb, one_hot, unfolded)` that takes a pre-computed raw distance matrix and builds the full feature tensor `[N, F]`. Then:
```python
D_native_raw = get_dist_matrix(Xjf_sq)  # computed ONCE — shared by 10 representations
D_decoy_raw  = get_dist_matrix(Xcd_sq)  # computed ONCE — only Xcd uses decoy coords

Xjf  = _build_graph_features(D_native_raw, mask_sq,      emb,        proT5_emb,       unfolded=False)
Xju  = _build_graph_features(D_native_raw, mask_sq,      emb,        proT5_emb,       unfolded=True)
Xd   = _build_graph_features(D_native_raw, mask_decoy,   emb_decoy,  proT5_emb_decoy, unfolded=False)
Xcd  = _build_graph_features(D_decoy_raw,  mask_crd_decoy, emb,      proT5_emb,       unfolded=False)
# ... cycle permutations all reuse D_native_raw with different embeddings
```

**Why:** The mask and embeddings (one-hot, ProtT5) differ per representation, but the coordinate geometry — and therefore the distance matrix — does not. Sharing the same `D_native_raw` across 10 representations reduces the most expensive per-step CPU operation by ~5×. Expected speedup: 15–25% reduction in total iteration time (distance matrix was ~30–40% of `get_noised_proteins` cost).

---

### 21. Fix `self.B`/`self.N` Instance Variables + Add `torch.compile` (Apr 18, `4f38b32`)

**Problem:** `PEM.forward()` stored batch size and sequence length as `self.B` and `self.N` (instance variables mutated during the forward pass), then read them in `forward_gat()` and `forward_gcn()` (sub-methods). This pattern prevents `torch.compile` from tracing a static graph — each forward call changes module state, forcing recompilation or falling back to eager mode.

**Fix — convert to local variables:** `self.B, self.N, _ = x.shape` → `B, N, _ = x.shape`, with `B` and `N` passed explicitly as arguments:
```python
# Before
x1 = self.forward_gcn(x_gcn, edge_index_gcn)  # reads self.B, self.N internally

# After
B, N, _ = x.shape
x1 = self.forward_gcn(x_gcn, edge_index_gcn, B, N)  # shape is local, not mutated state
```
All 12 occurrences of `self.B`/`self.N` converted; the `self.B = 0` / `self.N = 0` initializers in `__init__` removed.

**Add `torch.compile`:** With the mutation removed, the forward pass is now a pure function of its inputs and can be compiled:
```python
if CFG.compile_model and hasattr(torch, "compile") and CFG.device.type == "cuda":
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
```
`mode="reduce-overhead"` minimizes kernel launch overhead (the main bottleneck at batch_size=1). `fullgraph=False` allows partial compilation — ops that can't be traced (e.g., Python-level control flow in `get_edge_index`) fall back to eager without crashing. Controlled by `CFG.compile_model = True`; disabled automatically in `--debug` mode to avoid compilation overhead during fast iteration. Expected speedup: 10–20% on the GNN forward pass.
