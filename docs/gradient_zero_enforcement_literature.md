# Enforcing ∇E = 0 at Native Structures: Literature Review

**Context**: In our Protein Energy Model (PEM), native protein structures should be energy minima. This requires two conditions:
1. **E(native) < E(decoy)** — handled by InfoNCE ranking loss
2. **∇ₓE(native) = 0** — the subject of this review

Our challenge: E(x) takes a 1092-dim feature vector (16 distance + 32 bonded + 1024 proT5 + 20 one-hot), and enforcing ∇E = 0 via DSM requires Hessian ∂²E/∂x∂θ which vanishes at high dimensions.

---

## 1. Direct Force Supervision (Force Matching)

**Used by**: NequIP, MACE, SchNet, PaiNN, GemNet, TorchMD-Net

The standard approach in ML interatomic potentials. The model predicts scalar energy E(x), forces are derived via autograd `F = -∂E/∂x`, and a combined loss trains both:

```
L = λ_E * L_energy + λ_F * ||−∂E/∂x − F_target||²
```

- NequIP uses λ_F/λ_E = 1000 (forces dominate training)
- MACE uses Huber loss for robustness to outliers
- Ground truth forces come from DFT (quantum chemistry)
- To enforce ∇E = 0 at equilibrium: set F_target = 0

**Why it works for them**: Input x is 3D coordinates (3N dims for N atoms) — small enough that the Hessian ∂²E/∂x∂θ survives backpropagation.

**Why it fails for us**: Our x is 1092-dim features, not 3D coordinates. The Hessian vanishes at this dimensionality (empirically measured: parameter gradients from force matching are 150,000x smaller than from direct energy backward).

**References**:
- [NequIP: E(3)-equivariant GNN potentials](https://www.nature.com/articles/s41467-022-29939-5) (Batzner et al., Nature Comms 2022)
- [MACE: Higher Order Equivariant Message Passing](https://arxiv.org/abs/2206.07697) (Batatia et al., 2022)
- [TorchMD-NET: Equivariant Transformers for Neural Network Potentials](https://arxiv.org/abs/2202.02541) (Tholke & De Fabritiis, 2022)
- [Machine Learning Force Fields review](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c01111) (Chemical Reviews)
- [Practical Guide to ML Interatomic Potentials](https://arxiv.org/pdf/2503.09814) (2025)
- [Adaptive Loss Weighting for ML Interatomic Potentials](https://arxiv.org/html/2403.18122v1)

---

## 2. Denoising Score Matching (DSM)

**Used by**: ProteinEBM, DSMBind, Walk-Jump Sampling

Add noise to native structures x → x̃ = x + σε, then train the energy gradient to point back toward clean data:

```
L_DSM = ||∇ₓE(x̃) + (x̃ − x)/σ²||²
```

As σ → 0, this drives ∇E(x) → 0 at clean data points. DSM is mathematically equivalent to force matching against "noise forces" (x̃ − x)/σ².

**ProteinEBM** (bioRxiv 2025) is the most directly relevant: an energy-parameterized protein diffusion model trained with DSM where score = ∇E. Operates on 3D coordinates with multiple noise scales.

**DSMBind** applies DSM to SE(3)-equivariant binding energy prediction.

**Walk-Jump Sampling** (Frey et al., ICLR 2024) combines DSM-trained energy with Langevin MCMC ("walk") and one-step denoising ("jump") for protein generation.

**Multi-scale noise**: Using σ₁ > σ₂ > ... > σₙ helps — large σ provides easy targets (low-frequency landscape), small σ provides fine-grained gradient constraints.

**Why it works for them**: These models operate on 3D coordinates or SE(3) representations, where dimensionality is manageable.

**Our experience**: DSM on all 1092 features is flat (Hessian vanishes). DSM on 16 distance features only shows -46.8% loss reduction in 80 steps — dimensionality is small enough.

**References**:
- [ProteinEBM: Protein Diffusion Models as Statistical Potentials](https://www.biorxiv.org/content/10.64898/2025.12.09.693073v1) (bioRxiv 2025)
- [DSMBind: SE(3) Denoising Score Matching for Binding Energy](https://www.biorxiv.org/content/10.1101/2023.12.10.570461v1) (bioRxiv 2023)
- [Walk-Jump Sampling for Protein Discovery](https://arxiv.org/abs/2306.12360) (Frey et al., ICLR 2024)
- [Learning CG MD from Forces and Noise](https://arxiv.org/html/2407.01286v1) (Arts et al., 2024)
- [Yang Song's Score Matching Blog](https://yang-song.net/blog/2021/score/)
- [Score-Based Diffusion Models Explained](https://fanpu.io/blog/2023/score-based-diffusion-models/)
- [Denoising Score Matching Tutorial](https://johfischer.com/2022/09/18/denoising-score-matching/)

---

## 3. Contrastive Divergence (CD)

**Used by**: CG protein force fields (Tozzini et al.)

Maximum likelihood training of EBMs. The gradient is:

```
∂/∂θ log L = ⟨∂E/∂θ⟩_data − ⟨∂E/∂θ⟩_model
```

CD approximates the model expectation by running K short MCMC steps (Langevin dynamics or Metropolis) from data points. The energy is implicitly driven to have minima at data points because the Boltzmann distribution p(x) ∝ exp(−E(x)) is maximized when it matches the data distribution.

**Pros**: Principled ML framework; implicitly enforces native = energy minimum.

**Cons**: Short-run MCMC gives biased gradients; expensive; slow mixing for proteins with complex landscapes.

**References**:
- [CD for Coarse-Grained Protein Force Fields](https://pmc.ncbi.nlm.nih.gov/articles/PMC3966533/) (Tozzini et al., JCTC 2014)
- [Improved Contrastive Divergence Training of EBMs](https://energy-based-model.github.io/improved-contrastive-divergence/)

---

## 4. Variational Force Matching / Relative Entropy Minimization

**Used by**: Coarse-grained MD potential fitting

Two related "top-down" approaches:
- **Variational force matching**: Minimize MSE between all-atom forces (projected to CG space) and ∇E of the CG potential
- **Relative entropy minimization**: Minimize KL divergence between model's Boltzmann distribution and true CG distribution

Relative entropy achieves 100x data efficiency over force matching (Arts et al., 2024) but requires running simulations with the current model during training.

**Cons for us**: No ground-truth forces available for protein structures from PDB. Simulation during training is too expensive.

**References**:
- [Deep CG Potentials via Relative Entropy Minimization](https://arxiv.org/abs/2208.10330) (Wang et al., JCP 2023)
- [ML CG Potentials of Protein Thermodynamics](https://www.nature.com/articles/s41467-023-41343-1) (Nature Comms 2023)
- [Top-Down ML of CG Protein Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00638) (JCTC 2024)

---

## 5. Differentiable Simulation (DiffTRe)

**Used by**: TorchMD, JAX-MD based approaches

Run differentiable MD simulation with the learned energy, backprop through the entire trajectory to match experimental observables (RDFs, NMR shifts, folding rates). DiffTRe avoids full trajectory backprop by using reweighting.

**Cons**: Exploding gradients through long trajectories; massive memory; very slow.

**References**:
- [Learning NNPs from Experimental Data via DiffTRe](https://www.nature.com/articles/s41467-021-27241-4) (Nature Comms 2021)
- [Differentiable Molecular Simulation for Protein Force Fields](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256990) (PLoS ONE 2021)

---

## 6. Hessian / Curvature Regularization

Beyond ∇E = 0, enforce that the Hessian ∂²E/∂x² is positive semi-definite (true minimum, not saddle point):

```
L_hessian = λ * max(0, −λ_min(H))
```

**Cons**: Computing full Hessian is O(D²). Even Hutchinson trace estimator is expensive through a GNN. Rarely used in practice.

---

## 7. Physics-Informed Energy Loss (OpenMM-Loss)

Use an external physics engine (OpenMM) to compute potential energy U(x) of predicted structures as additional loss. Applied to OpenFold (Doerr et al., 2024).

**Cons**: Requires all-atom coordinates including hydrogens; not applicable to our backbone-only GNN energy model.

**References**:
- [Interpreting Forces as Deep Learning Gradients](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393680/) (2024)

---

## Key Insight: Why Our Problem Is Different

All successful approaches (NequIP, MACE, ProteinEBM, DSMBind) compute ∂E/∂x where **x is 3D coordinates** — a small, physically meaningful space (3N dims for N atoms). The Hessian works fine there.

**Our model is different**: E(x) takes a 1092-dim feature vector, not raw coordinates. The graph construction pipeline (coordinates → distance matrix → Gaussian kernel → bonded features → concatenate with embeddings) creates a high-dimensional intermediate representation. This is why the Hessian vanishes.

### Feature layout
```
[D(0:16) | Fb(16:48) | proT5(48:1072) | one_hot(1072:1092)] = 1092 dims
```

- D (16 dims): inter-residue distances — physically meaningful, DSM works here
- Fb (32 dims): bonded features from adjacent residues
- proT5 (1024 dims): frozen pretrained embeddings — ∇E = 0 w.r.t. these has no physical meaning
- one_hot (20 dims): amino acid identity

---

## Practical Options for DeepEF

### Option A: DSM D-only (current, proven)
Score matching on only the 16 distance features. Works because:
- 16 dims → Hessian survives
- Distance features carry ~47% of energy sensitivity
- Showed -46.8% loss reduction in 80 steps

### Option B: Coordinate-space DSM (like ProteinEBM)
Noise the actual 3D backbone coordinates (available in `native_info`), rebuild graph features from noised coords, compute ∂E/∂coords. This is what ProteinEBM does.
- Dimensionality: 4 atoms × 3 coords × N residues (small per-residue)
- Physically grounded — perturbing actual atomic positions
- Challenge: need to differentiate through the graph construction pipeline

### Option C: Multi-scale DSM D-only
Use multiple noise levels (σ = 2.0, 1.0, 0.5, 0.1) on D features:
- Large σ: easy targets, shapes global landscape
- Small σ: fine-grained gradient constraints near native

### Option D: Combined approach
InfoNCE (ranking) + DSM D-only (gradient) + explicit gradient penalty at native. Three complementary signals.

---

## Our Experimental Results

| Approach | Dims | Loss change (100 steps) | Status |
|----------|------|------------------------|--------|
| DSM D-only (16 distance) | 16 | -46.8% | Works |
| Holistic DSM σ=0.5 | 1092 | 0% (flat) | Broken |
| Holistic DSM σ=2.0 | 1092 | 0% (flat) | Broken |
| SSM K=1 | 1092→1 | -82% but noisy | Unstable |
| SSM K=8 | 1092→1 | +10.9% | Worse |

**Root cause**: DSM parameter gradient ∂L/∂θ requires Hessian ∂²E/∂x∂θ. Measured at 150,000× smaller than direct energy gradients. Gradient cancellation scales as O(1/√D) where D = feature dimensionality.
