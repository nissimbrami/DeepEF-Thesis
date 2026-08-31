# DeepPEF v5 — Gap-Fill Teaching Content

Authoritative reference notes to fill gaps identified in the teaching materials.
Each section is written as self-contained, teaching-quality prose. Formulas use
inline notation acceptable in Markdown (e.g. `r = cov(x,y) / (σx·σy)`).

Scope note: numbers such as the ~0.5259 baseline PCC are reported only as the
project's own historical result; no new accuracy figures are invented here.

---

## 1. Evaluation Metrics

We evaluate ddG / dG predictions against experimental values with three metrics.
Two measure *trend/ranking* (correlation), one measures *absolute error* (RMSE).

Let `y = (y_1, ..., y_n)` be the true stability values and `p = (p_1, ..., p_n)`
the predicted values over `n` mutations (or proteins).

### Pearson correlation coefficient (PCC)

**Formula.** Pearson measures the *linear* association between predictions and
truth:

```
r = cov(p, y) / (σ_p · σ_y)
  = Σ_i (p_i - p̄)(y_i - ȳ) / sqrt( Σ_i (p_i - p̄)²  ·  Σ_i (y_i - ȳ)² )
```

where `p̄`, `ȳ` are the means and `σ` are standard deviations.

**Range.** `r ∈ [-1, +1]`.
- `+1` = predictions rise perfectly linearly with truth.
- `0` = no linear relationship.
- `-1` = perfectly anti-correlated (predicts the opposite direction).

**What it measures for ddG.** Whether, across many mutations, the model's
predicted stability change tracks the real stability change. It is
*shift- and scale-invariant*: adding a constant to every prediction, or
multiplying all predictions by a positive factor, does not change `r`. So PCC
answers "does the model get the *pattern* right?" and ignores systematic offset
or units.

**Why PCC is our primary grading metric.** For downstream use — ranking mutation
candidates, deciding which variant is more stabilizing — the *relative ordering*
and linear trend matter far more than the exact kcal/mol value. Our energy model
outputs an energy in arbitrary/relative units (it is not calibrated to absolute
kcal/mol), so a metric that is invariant to overall shift and scale is the fair
way to score it. The project's historical best on the PNAS-filtered Megascale
setup is a PCC of ~0.5259, which is the reference bar the v5 improvements are
measured against.

### Spearman rank correlation (ρ, "rho")

**Formula / intuition.** Spearman is simply Pearson computed on the *ranks* of
the values instead of the raw values. Replace each `p_i` with its rank
`R(p_i)` (1 = smallest, ... n = largest), do the same for `y_i`, then:

```
ρ = Pearson( R(p), R(y) )
```

For distinct ranks with no ties this reduces to the classic form
`ρ = 1 - 6·Σ d_i² / (n(n² - 1))`, where `d_i = R(p_i) - R(y_i)` is the rank
difference for item `i`.

**Range.** `ρ ∈ [-1, +1]`, same interpretation as Pearson but for *monotonic*
(not necessarily linear) agreement.

**Why rank matters for stability ranking.** Spearman only asks "does the model
put mutations in the correct order?" It is robust to non-linear-but-monotonic
distortions and to outliers, because a huge numeric error only moves an item by
its rank position, not by its raw magnitude. For prioritizing which variants to
test experimentally, correct *ordering* is exactly the quantity of interest, so
Spearman is a natural companion to PCC.

### RMSE (Root-Mean-Square Error)

**Formula.**

```
RMSE = sqrt( (1/n) · Σ_i (p_i - y_i)² )
```

**Units.** Same units as the target. When predicting dG/ddG this is
**kcal/mol** — but note this only holds when the model output is calibrated to
that scale. Because our energy is in relative units, RMSE here should be read as
a magnitude-of-error diagnostic rather than a physically calibrated kcal/mol
number.

**What it measures / caveats.** Unlike the correlations, RMSE penalizes
*absolute* deviations and squaring makes it sensitive to large outliers. A model
can have excellent PCC (right trend) but poor RMSE (systematic offset or wrong
scale), or vice versa. That is why we report all three: correlation for the
pattern, RMSE for the calibration.

---

## 2. Loss Functions

Losses are the *training* objective; metrics are the *evaluation* objective. The
fine-tuning loss in `pnas_train_v5.py` is a weighted sum:

```
loss = primary_loss + reg_loss + energy_reg + rank_loss + corr_loss
```

- `primary_loss` — point-wise regression (Huber or L1) on predicted vs true dG.
- `reg_loss`     — L2 weight decay on model parameters (`REG_LAMBDA · Σ ‖θ‖²`).
- `energy_reg`   — L2 penalty on the raw energies (see below).
- `rank_loss`    — margin ranking loss (optional, weight `RANKING_LAMBDA`).
- `corr_loss`    — Pearson or CCC correlation loss (v5, optional, `corr_weight`).

Each component is defined below.

### Huber loss (delta = 1)

**Formula.** For residual `e = p - y` and threshold `δ`:

```
L_δ(e) = 0.5 · e²                    if |e| ≤ δ
       = δ · (|e| - 0.5·δ)           if |e| >  δ
```

With `δ = 1` (our default) the switch happens at an absolute error of 1.

**Why robust to outliers vs MSE / L1.** Huber is a hybrid:
- Near zero it behaves like **MSE** (`0.5·e²`) — smooth, with a gradient that
  shrinks as the error shrinks, giving stable, fine-grained convergence.
- Far from zero it behaves like **L1** (`δ·|e|`, constant slope `δ`) — the
  gradient *saturates* instead of exploding, so a single mislabeled or extreme
  mutation cannot dominate the update.

MSE alone squares large errors and lets noisy labels hijack training; L1 alone
has a non-smooth kink at zero and a constant gradient that hurts precise
convergence. Huber gets the best of both, which matters because experimental
ddG labels contain measurement noise.

### Margin ranking loss

**Formula (pairwise).** For a pair `(i, j)` with truth `y_i, y_j` and
predictions `p_i, p_j`, let `s = sign(y_i - y_j)` and `Δp = p_i - p_j`:

```
L_rank(i, j) = max( 0, -s · Δp + margin )
```

In our code (`ranking_loss`), random pairs are sampled from the mini-batch (up
to 128 pairs), `sign` is taken from the *target* difference, and the loss is the
mean over pairs, scaled by `RANKING_LAMBDA` (default 0.1).

**Why it helps ranking.** The regression loss only cares that each individual
prediction is close to its own label. The ranking term explicitly says: "if
mutation `i` is truly more stable than `j`, then `p_i` should exceed `p_j` by at
least `margin`." If the order is already correct with enough separation, the
`max(0, ·)` clamps the term to zero — no penalty. If the order is wrong or too
close, it pushes the pair apart in the correct direction. Because our primary
grading metric is a correlation (a trend/ordering metric), directly optimizing
pairwise order aligns training with evaluation.

### Energy regularization (`0.001 · ‖E‖²`)

**Formula.** With the concatenated folded + unfolded energies `E`:

```
energy_reg = E_REG_LAMBDA · MSE(E, 0) = E_REG_LAMBDA · mean(E²),   E_REG_LAMBDA = 0.001
```

**Why bound the extensive energy.** The model's energy is *extensive*: it is a
sum of per-residue contributions (`E = Σ_residues Σ_terms Fh`), so it grows with
protein length and can drift to large positive/negative magnitudes that are only
weakly constrained by the ddG signal (a *difference* of two energies). Without a
bound, the raw energies can wander far from zero while their difference stays
correct, causing numerical scale issues and unstable gradients. The small
`0.001` L2 penalty gently anchors the absolute energy scale near zero without
distorting the *difference* that carries the ddG signal — a soft
"keep the energies well-behaved" prior, deliberately tiny so it never competes
with the primary loss.

### Correlation losses: Pearson loss and CCC loss (v5)

These directly optimize the *trend* the PCC metric rewards. Two predictions can
share the same Huber loss yet very different PCC; a correlation loss closes that
gap. They are opt-in via `--corr_weight` (default 0 = off).

**Pearson loss.**

```
L_pearson = 1 - r
```

where `r` is the batch Pearson correlation (Section 1). `r = +1` gives loss 0;
lower correlation gives higher loss. It is invariant to shift and scale, so it
purely rewards getting the linear trend right. Implemented in `losses_v5.py` with
centering and an `eps` for numerical safety; returns 0 for batches smaller than 2
or with zero variance.

**CCC loss (Lin's Concordance Correlation Coefficient).**

```
CCC = 2·cov(p, y) / ( σ_p² + σ_y² + (μ_p - μ_y)² )
```

so `L_ccc = 1 - CCC`. Equivalently `CCC = r · C_b`, where `r` is Pearson and
`C_b` (the *bias correction factor*) is a penalty in `(0, 1]` that is largest
(= 1) only when the means match **and** the variances match.

**How CCC differs from Pearson.** Pearson is invariant to shift and scale — it
only checks the *pattern*. CCC additionally penalizes:
- a **location shift** — through the `(μ_p - μ_y)²` term in the denominator; if
  predictions are systematically offset from truth, CCC drops even when Pearson
  is perfect.
- a **scale mismatch** — through `σ_p² + σ_y²`; if the spread of predictions does
  not match the spread of the targets (wrong dynamic range), CCC drops.

So CCC demands agreement about the **1:1 line**, not merely a linear
relationship. It is therefore stricter than Pearson: `CCC ≤ |r|` always, with
equality only under identical means and variances. Use Pearson loss when only the
trend matters (the metric we are graded on); use CCC loss when we also want the
predicted dG to agree in absolute location/scale with experiment. Both are
implemented in `losses_v5.py`.

### Denoising MSE (Lever E)

**Idea.** Lever E adds a denoising objective at fine-tuning time as a
regularizer/auxiliary signal. Structural coordinates (or their derived distance
features) are perturbed with Gaussian noise and the model is trained to be
consistent with the clean structure — an MSE between the noised-path prediction
and the clean target. This is the fine-tuning-time echo of the pretraining
Denoising Score Matching objective (Section 4): it teaches the energy surface to
behave sensibly under small structural perturbations rather than overfitting to
exact coordinates. In v5 it is carried through `get_deltaG` as `denoise_loss`,
gated by its lever flag so that with the lever off the behavior is exactly the
baseline. Because the perturbation is a controlled noise, the objective is a mean
squared error term, i.e. `denoise_MSE = mean( (prediction_noised - target)² )`.

---

## 3. Dataset Biology

### The Tsuboyama et al. (2023) mega-scale stability dataset

The training/validation data derive from the Tsuboyama et al. mega-scale study
(published in *Nature*, 2023), which measured folding stability for on the order
of hundreds of thousands of protein domains and point mutants using a
high-throughput **cDNA-display proteolysis** assay. This scale is far beyond
what classical, one-protein-at-a-time calorimetry could produce, which is why it
became a workhorse dataset for learning ddG models.

### The K50 cDNA-display proteolysis assay (how dG is obtained)

The assay converts a protein's *resistance to being cut by a protease* into a
folding free energy. The logic:

1. Each protein variant is physically linked to the cDNA that encodes it
   (cDNA display), so sequencing counts act as a readout of how much of each
   variant survived a treatment.
2. Variants are exposed to a **protease** (e.g. trypsin / chymotrypsin) at a
   *range of concentrations*. A folded protein hides its backbone inside the
   structure and **resists cleavage**; an unfolded protein exposes cut sites and
   is rapidly digested.
3. As protease concentration rises, more of the population is cut. **K50** is the
   protease concentration at the assay's *midpoint* — where half the molecules
   have been cleaved. A more stable (more folded) protein resists longer, so its
   K50 sits at a higher protease concentration.
4. Using a thermodynamic model that links this proteolysis midpoint to the
   folded↔unfolded equilibrium (with an unfolded-state reference), the measured
   K50 is converted into a **folding free energy dG** for each variant. A
   mutation's effect is then `ddG = dG_mutant - dG_wildtype`.

Intuition: **K50 is a stability thermometer read through the lens of protease
resistance** — higher K50 ⇒ harder to digest ⇒ more folded ⇒ more stable.

### Why clamp dG to [-1, 5] kcal/mol

The `--dg_ml` option clamps target dG to the range **[-1, 5] kcal/mol** (see the
`threshold = [-1.0, 5.0]` in the dataset code). This reflects the **assay's
dynamic range**. Proteolysis assays can only resolve stability where the
folded↔unfolded transition actually moves within the accessible protease
concentrations:
- Below the low end, a protein is effectively always unfolded/always cut — the
  assay saturates and cannot distinguish "unstable" from "even more unstable."
- Above the high end, a protein is effectively always folded/never cut — the
  assay saturates in the other direction.

Values pushed outside this window are extrapolations, not measurements. Clamping
to `[-1, 5]` keeps the labels inside the region the assay can actually measure,
preventing the model from chasing noisy, saturated pseudo-values.

### Why single mutations + PNAS quality curation (quality > quantity)

Two filters are treated as essential, not optional:
- `--one_mut` keeps only **single-point** mutations. A single substitution has a
  clean, attributable stability effect; multi-mutants confound several changes
  and their interactions (epistasis), muddying the learning signal.
- The **PNAS-filtering** curation selects high-quality, well-measured single-site
  variants (filtering noisy measurements, extreme outliers, and low-quality
  proteins).

The project's own experiments showed that *adding* the full unfiltered Megascale
data **hurt** performance versus the curated subset. The lesson recorded in the
project is blunt: for noisy-label ddG regression, **data quality beats data
quantity** — the curation removes label noise that a bigger-but-dirtier set would
inject.

### ThermoMPNN and why test proteins are held out

**ThermoMPNN** is the benchmark/competitor: a ddG predictor that fine-tunes on
top of **ProteinMPNN** inverse-folding representations. Because ProteinMPNN was
pretrained on large amounts of structure→sequence data, ThermoMPNN carries a
strong learned prior about which residues fit a given fold — which is a large
part of why inverse-folding pretraining is the single biggest known lever in this
field.

The proteins used to define the ThermoMPNN benchmark are **held out** of DeepPEF
training (the dataset code reads a ThermoMPNN protein list and removes those
proteins, along with homologs, from the training pool). This prevents *test
leakage*: if a benchmark protein (or a close homolog) appeared in training, the
reported score would be inflated and not reflect true generalization to unseen
proteins. Holding them out — and removing homologs — makes the comparison against
ThermoMPNN a fair, out-of-sample test.

---

## 4. EBM Pretraining Theory

This is implemented in the root `train.py` (the pretraining stage) and is the
conceptual foundation the fine-tuning stage builds on.

### Energy-based models: native structure = energy minimum

An **energy-based model (EBM)** assigns a scalar energy `E(x)` to a
configuration `x` (here, a protein structure + sequence). We *define* correctness
by low energy: the true, native folded structure should have the **lowest
energy**, and everything else — wrong structures, wrong sequences, unfolded
chains — should have higher energy. Training does not regress to a target number;
it *shapes the energy landscape* so that its minimum sits at the native state.
This is a physics-flavored inductive bias: proteins fold to their free-energy
minimum, so we teach the model to make the native the minimum.

### InfoNCE / Boltzmann contrastive loss

To carve out that minimum we use a **Boltzmann / InfoNCE contrastive** loss. Turn
each energy into an unnormalized Boltzmann probability `exp(-E/τ)` (temperature
`τ`), then make the native the "correct class" among a set of negatives:

```
L = -log[ exp(-E_native/τ) / Σ_k exp(-E_k/τ) ]
  =  E_native/τ + log Σ_k exp(-E_k/τ)
```

In `train.py` (`lossd_fucntion`) there are two contrastive terms:
1. **Primary:** the native folded energy `Ejf` must be the lowest among a pool
   of negatives: decoy *sequence* on native structure, decoy *structure*, native
   *unfolded*, and several cyclic-permutation graphs `Ecy1..Ecy4`.
2. **Secondary:** the native *unfolded* energy must still be lower than a *decoy
   structure* energy (native chain beats a wrong fold even when unfolded).

The negatives are constructed to cover distinct failure modes:
- **shuffled sequence** (`mix_A_acid`) — right structure, wrong sequence;
- **wrong / decoy structure** (`crd_decoy`) — right sequence, wrong fold;
- **cyclic permutations** — same residues rearranged along the chain, forcing the
  model to respect actual sequence order and contacts rather than a bag of
  residues.

The `logsumexp` form **saturates to ~0** once the native is already well below
all negatives, so training focuses effort on the still-confused cases. `τ`
controls how strict the separation must be.

### Denoising Score Matching (DSM)

Contrastive loss shapes *where* the minimum is; **DSM** shapes the *slope around
it* — it teaches the energy **gradient** to point back toward the native
structure. Add Gaussian noise `noise ~ N(0, σ²)` to the (distance) features of
the native and train the model's score `∇E` so that, along a random direction
`v`:

```
v · ∇E  ≈  v · ( -noise / σ² )
```

i.e. the negative gradient (the "force") points from the noised configuration
back toward the clean native, with magnitude set by how far the noise pushed it.
The implementation (`denoising_score_matching`) uses a **finite-difference**
estimate of the directional derivative to avoid a vanishing second-derivative
(Hessian) problem through normalization layers:

```
v · ∇E  ≈  ( E(x + ε·v) - E(x - ε·v) ) / (2ε)
loss     =  mean_over_directions[ ( FD_score - target_v )² ],   target_v = -Σ(v·noise)/σ²
```

The result is an energy surface that not only *ranks* the native lowest but has a
smooth basin whose gradients push perturbed structures back toward native — a
learned, differentiable force field.

### How DSM connects to Lever E (denoising) at fine-tuning time

Lever E (Section 2) is the fine-tuning-time reflection of DSM. At pretraining,
DSM teaches the energy gradient to point toward native under noise. At
fine-tuning on mutations, the same denoising idea reappears as an auxiliary
**denoising MSE** term: perturb the structure, require the prediction to stay
consistent with the clean structure. This preserves the "well-shaped basin"
learned during pretraining and prevents the ddG fine-tuning from flattening or
distorting the energy landscape around the native. In short: DSM builds the force
field; Lever E keeps that force field intact while the model learns mutation
effects.

---

## 5. Why Physics-Grounded (the thesis framing)

The central research bet of DeepPEF is **architectural**: instead of a generic
ML regressor that maps (structure, sequence) → ddG through arbitrary layers, we
build **thermodynamics into the model itself** and let the network fill in only
the parts physics does not pin down. Three design choices encode this bias:

1. **Folded − unfolded energy difference.** Stability is *by definition* a
   difference of free energies, `dG = E_unfolded - E_folded`. The model computes
   two energies from the *same* coordinates — a folded graph with full pairwise
   contacts, and an unfolded graph reduced to the linear chain — and takes their
   difference. This mirrors the actual thermodynamic cycle and means the network
   is rewarded specifically for recognizing the stabilization gained by forming
   3D contacts, not for memorizing an absolute number. A mutation changes the
   sequence embedding and one-hot encoding while the coordinates stay fixed, so
   `ddG = dG_mut - dG_wt` isolates the sequence-driven change.

2. **Extensive (summed) energy.** Energy is built as a **sum of per-residue
   contributions** — an extensive quantity, exactly as physical energies are
   additive over a system's parts. This is a deliberate constraint: the model
   must express stability as a sum of local terms, which is interpretable
   (per-residue energies) and physically motivated, even though it introduces a
   length dependence that v5's optional `length_norm` lever can convert to an
   intensive per-residue energy when desired.

3. **Flory / unfolded-state reference.** The unfolded state is not treated as an
   afterthought but as an explicit **reference state** (in the spirit of Flory's
   random-coil model of the denatured chain). Measuring folded energy *relative
   to* this reference is what makes the difference meaningful — stability is
   always a comparison to the unfolded ensemble, and the architecture bakes that
   comparison in.

**Why make this bet?** A generic regressor with enough capacity could in
principle learn ddG, but it would need to *rediscover* thermodynamics from
limited, noisy labels and would generalize poorly off-distribution. By encoding
the folded/unfolded difference, the additive energy, and the Flory reference
directly into the architecture, we shrink the hypothesis space to physically
plausible functions. That inductive bias should mean better data efficiency, more
interpretable outputs (energies and per-residue terms that mean something), and
more trustworthy generalization to unseen proteins. The trade-off — and the honest
part of the thesis framing — is that these constraints can cap raw accuracy if a
purely data-driven competitor (e.g. inverse-folding-pretrained ThermoMPNN) has a
stronger prior. The research question is precisely whether a physics-grounded
energy function can approach or match such models while remaining interpretable
and mechanistically faithful.
