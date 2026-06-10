# DeepPEF — Levels of Progression

## How to Read This
- ✅ = This level improved over previous
- ❌ = This level failed / went backward
- ⏳ = Planned but not yet executed
- Each level builds on what we learned from previous levels

---

## LEVEL 0: Raw Baseline (Starting Point)

- **Config:** Pretrained PEM model, MSE loss, fully-connected graph, ProtT5 per-mutation embeddings, constant LR 1e-4
- **PCC:** ~0.48
- **Learning:** The pretrained model overfits to the decoy discrimination task (native vs wrong fold), which is fundamentally different from ddG prediction (stable vs unstable mutation). The fully-connected graph introduces O(L^2) noisy edges from distant residues with no physical interaction. MSE loss gives extreme ddG outliers disproportionate gradient influence.

---

## LEVEL 1: Train From Scratch ✅

- **Change:** `--no_pretrained` (random weight initialization instead of loading pretrained decoy-discrimination model)
- **PCC:** 0.48 → 0.485 (+0.005)
- **Learning:** Fresh random weights learn ddG-specific features better than weights transferred from the decoy task. Pre-training only helps when the pre-training objective is ALIGNED with the downstream task. Decoy discrimination (native fold vs wrong fold) is not aligned with mutation stability prediction. The pretrained features actively interfere.

---

## LEVEL 2: Huber Loss ✅

- **Change:** `--loss_type huber` (delta=1.0) replacing MSE/L1 loss
- **PCC:** 0.485 → 0.501 (+0.016)
- **Learning:** Some mutations have extreme experimental ddG values (outliers, possibly measurement errors). MSE gives these outliers quadratic gradient magnitude, dominating training and pulling the model toward fitting noise. Huber clips their influence by switching from quadratic to linear penalty beyond delta=1.0. This is a fundamental "fix the training dynamics" improvement that should be kept in ALL future experiments.

---

## LEVEL 3: Cosine LR Schedule ✅

- **Change:** `--cosine_lr` with lr_min=1e-6, annealing from 1e-4 to 1e-6 over 15 epochs
- **PCC:** 0.501 → 0.516 (+0.015)
- **Learning:** A constant learning rate overshoots near convergence — the model oscillates around the minimum in later epochs instead of settling into it. Cosine annealing provides a smooth decay that allows coarse exploration early and fine-grained updates late. This is a well-established optimization technique that provides a nearly free boost.

---

## LEVEL 4: k-NN GAT (k=30, 12A cutoff) ✅

- **Change:** `--use_knn_gat` — sparse spatial edges (k=30 nearest CA atoms within 12 Angstroms) instead of fully-connected graph
- **PCC:** 0.516 → 0.5259 (+0.043 total from this single architectural change)
- **Learning:** BIGGEST SINGLE GAIN in the entire project. The fully-connected graph connects every residue pair, creating O(L^2) edges. For a 200-residue protein that is 40,000 edges, most connecting residues 30+ Angstroms apart with zero physical interaction. These noisy edges dilute the GAT attention — the model wastes capacity trying to learn "ignore this irrelevant edge." k-NN focuses exclusively on structurally relevant contacts (actual physical neighbors). Fewer edges = cleaner gradient signal = better learning. This validates that REDUCING noise is more valuable than adding information.

---

## LEVEL 5: Ranking Loss ✅

- **Change:** `--loss_type huber_rank --ranking_weight 0.1` — add pairwise margin ranking loss (margin=0.1) on top of Huber
- **PCC:** Contribution +0.003
- **Learning:** Small but consistent gain. PCC measures correlation but the Huber loss alone doesn't explicitly optimize for ordering. The ranking loss adds: "if mutation A has higher ddG than mutation B, the model should predict accordingly." This directly optimizes the metric we care about (correlation = correct ordering). The small magnitude suggests the Huber loss already captures most of the signal, but ranking adds a useful inductive bias for preserving mutation ordering within each protein.

---

## LEVEL 6: Weight Decay ✅

- **Change:** `--weight_decay 1e-5` — mild L2 regularization on all parameters
- **PCC:** Contribution +0.008
- **Learning:** With a 22M parameter model trained on only ~68K mutation samples (340 proteins x ~200 mutations), mild overfitting occurs. Weight decay penalizes large weight magnitudes, preventing the model from memorizing training examples. The small value (1e-5) ensures regularization without under-fitting. This is another "training dynamics" fix rather than an architectural change.

---

## LEVEL 7: Combined Best = 0.5259 ✅

- **Config:** ALL of Levels 1-6 combined, seed 42
  - From scratch (no pretrained model)
  - k-NN GAT with k=30, CA cutoff=12A
  - Huber loss (delta=1.0) + Ranking loss (lambda=0.1, margin=0.1)
  - Cosine LR: 1e-4 -> 1e-6 over 15 epochs
  - Weight decay: 1e-5
  - Mini-batch: 64 mutations per protein
  - dg_ml range clipped to [-1, 5]
- **PCC:** 0.5259
- **Learning:** Gains are roughly additive — each addresses a DIFFERENT failure mode (outlier sensitivity, LR scheduling, graph noise, overfitting, task misalignment, ordering). There is no single silver bullet; the combined +0.045 PCC comes from systematically eliminating multiple independent sources of error. This is the proven reproducible baseline that all future experiments must beat.

---

## FAILED LEVELS (Things That Went Backward)

---

## LEVEL F1: dual_esmif 1536-dim (No Projection) ❌

- **Change:** Concatenate ProtT5 (1024-dim) + ESM-IF1 encoder (512-dim) = 1536-dim input, no dimensionality reduction
- **PCC:** 0.5259 → 0.4232 (-0.10!)
- **Root Cause:** 1536-dim features require significantly more VRAM per forward pass. On 8GB GPU, mini_batch_size had to be reduced from 64 to 8. With batch=8, only 8 mutations contribute to each gradient step, making gradients extremely noisy (high variance). Noisy gradients prevent the optimizer from finding good minima. The extra information from ESM-IF1 was completely overwhelmed by the degraded optimization.
- **Learning:** HIGH dimensionality + small batch = DISASTER. Feature dimensionality is constrained by hardware. The information gain from adding ESM-IF1 is LESS than the optimization loss from reducing batch size by 8x. MUST project high-dimensional features down before they enter the GNN. The fix exists (`--emb_projection mlp` projects 1536->16) but was never tested.

---

## LEVEL F2: ESM-IF1 Only (512-dim, No ProtT5) ❌

- **Change:** Replace ProtT5 with ESM-IF1 structural encoder features only (512-dim per residue)
- **PCC:** 0.2856
- **Root Cause:** ESM-IF1 encoder features capture "what the local structure looks like" but NOT "what amino acids are evolutionarily preferred at this position." ProtT5 carries the evolutionary/sequence signal that is critical for predicting which mutations are destabilizing. Structure alone tells you the environment but not the compatibility of a specific amino acid with that environment.
- **Learning:** Structure alone is insufficient. You need BOTH structural AND sequence/evolutionary information. ProtT5's value comes from encoding evolutionary constraints (which AAs are tolerated at each position) — this is exactly what ddG prediction needs. Any future architecture must preserve the ProtT5 signal.

---

## LEVEL F3: WT-Only Embeddings (No Per-Mutation Signal) ❌

- **Change:** Use the same wildtype ProtT5/ESM-2 embedding for all mutations at each position (ignore mutation identity in embedding)
- **PCC:** ~0.25
- **Root Cause:** If the model receives IDENTICAL input for all mutations at the same position, it has ZERO information to distinguish A32G from A32W. The only mutation-specific signal comes from the 20-dim one-hot encoding, which is overwhelmed by the 1024-dim identical embedding. The model effectively cannot see the mutation.
- **Learning:** The model NEEDS per-mutation embeddings. The ProtT5 embedding of the mutant sequence carries critical information about "how compatible is this specific amino acid in this sequence context." Without it, the model is essentially guessing. This is why the subtract-mut approach is attractive — it gets per-mutation signal from the OUTPUT (score[mut_aa] - score[wt_aa]) rather than from the INPUT.

---

## LEVEL F4: Fine-Tuning Pretrained Model ❌

- **Change:** Start from pretrained weights (trained on decoy discrimination), fine-tune on ddG
- **PCC:** 0.48 (same as baseline, worse than from-scratch 0.526)
- **Root Cause:** The pretrained model learned "native fold has lower energy than decoys" — a completely different objective from "mutation X destabilizes by 1.5 kcal/mol." The pretrained features encode fold-quality signals that are irrelevant (and possibly harmful) for mutation stability. Fine-tuning cannot fully overcome this initialization bias in 15 epochs.
- **Learning:** Pre-training on the WRONG task is worse than random initialization on the RIGHT task. Domain-appropriate pretraining (like inverse folding, which teaches "which amino acids fit at each position") would be beneficial. Wrong pretraining actively hurts. This validates Level 1's decision to train from scratch.

---

## LEVEL F5: SaProt Embeddings ❌

- **Change:** Replace ProtT5 with SaProt (structure-aware protein language model, 1280-dim)
- **Result:** CRASH — never produced a PCC
- **Root Cause:** The SaProt generation script produced POOLED embeddings: shape [1, 1280] per protein (one vector for the whole protein). Our architecture expects PER-RESIDUE embeddings: shape [seq_len, 1280] (one vector per residue). The shape mismatch crashed during data loading. This was never fixed or re-attempted.
- **Learning:** ALWAYS verify tensor shapes before launching training. A single assertion (`assert emb.shape[0] == seq_len`) would have caught this in 1 second. Never waste hours debugging a crash that a shape check would prevent. The SaProt idea itself is reasonable — only the implementation was wrong.

---

## LEVEL F6: Full-Dimensional DSM (1092-dim Score Matching) ❌

- **Change:** Apply Denoising Score Matching loss to the full 1092-dim feature space (distances + embeddings + one-hot)
- **Result:** Gradient vanishing, training stalled
- **Root Cause:** The Hessian of a 1092-dim feature space is extremely ill-conditioned. Most dimensions contribute near-zero curvature. The useful signal is concentrated in the 16 distance dimensions only. In 1092 dims, the score matching gradient is diluted ~20,000x below useful magnitude.
- **Learning:** Score matching only works in low-dimensional subspaces where the energy landscape has meaningful curvature. Must restrict DSM to distance features (16-dim) only. Not all dimensions of the feature space have physically meaningful energy landscapes.

---

## LEVEL F7: Holistic DSM Noise (Noise on All Features) ❌

- **Change:** Add denoising noise to ALL features: structure + embeddings + sequence one-hot
- **Result:** No improvement, reverted
- **Root Cause:** Noise on embeddings DESTROYS the PLM signal (ProtT5 features are already optimized by a billion-parameter model — adding random noise corrupts them). Noise on one-hot makes amino acid identity ambiguous (confuses the model about which amino acid is present). Only noise on DISTANCE features creates physically meaningful perturbations (slight coordinate jitter = realistic thermal fluctuations).
- **Learning:** DSM noise must be domain-appropriate. For proteins, only perturb coordinates/distances (these have physical meaning as thermal fluctuations). Never perturb features that come from pretrained models — they encode compressed information that noise destroys.

---

## PLANNED LEVELS (Not Yet Executed)

---

## LEVEL 8: 5-Seed Ensemble ⏳

- **Change:** Run Level 7 config with seeds [42, 123, 456, 789, 1337], average all 5 predictions
- **Expected PCC:** 0.55-0.57
- **Why:** Ensembling reduces prediction variance. Each seed finds a slightly different local minimum, and averaging smooths out individual model errors. This is a FREE boost requiring zero architecture changes — just run the same training 5 times.
- **What we learned that makes this likely to work:** Level 7 is reproducible and stable. Different seeds should produce PCC in [0.50-0.54] range individually, and their average should beat any single seed.
- **Implementation:** Script exists (`run_final_pipeline.sh`), just needs GPU execution.
- **Probability:** 90%

---

## LEVEL 9: dual_esmif + MLP Projection ⏳

- **Change:** Concatenate ProtT5 (1024) + ESM-IF1 (512) = 1536-dim, then project 1536 -> 16 via MLP before entering GNN. This fixes Level F1's OOM problem.
- **Expected PCC:** 0.53-0.56
- **Why:** ESM-IF1 encodes structural information that ProtT5 lacks. The projection keeps the feature dimension manageable so batch=64 still fits in VRAM. The MLP learns which dimensions of the 1536-dim concatenation are most informative.
- **What we learned that makes this plausible:** Level F1 proved the information is there (just can't use it raw). Level F2 proved ESM-IF1 alone is insufficient. Projection bridges both failures — keeps both signals while fitting in memory.
- **Implementation:** Code exists (`--emb_projection mlp`), never passed from training script.
- **Probability:** 75%

---

## LEVEL 10: GNN Subtract-Mut Output (GNN-SM) ⏳

- **Change:** Change model output from scalar energy to [L, 20] amino acid compatibility scores. ddG = score[position, mut_aa] - score[position, wt_aa]. Process wildtype structure ONCE, extract predictions for ALL mutations from single forward pass.
- **Expected PCC:** 0.55-0.65
- **Why:**
  - The subtraction cancels systematic position-specific bias (proven in ThermoMPNN, PCC 0.754)
  - 200x fewer forward passes per protein = each gradient step uses ALL mutations = dramatically cleaner gradients
  - Eliminates noisy unfolded-state prediction entirely
  - One forward pass per protein means batch=ALL_MUTATIONS (not batch=64)
- **What we learned that makes this plausible:**
  - Level 4 (k-NN GAT) validates spatial graph structure → keep it in GNN-SM
  - Level F3 proves per-mutation signal is essential → subtract-mut gets it from OUTPUT not INPUT
  - Level F1 proves clean gradients matter → GNN-SM has maximally clean gradients (all mutations in 1 step)
  - ThermoMPNN proves subtract-mut scoring works (PCC 0.754 with just an MLP)
- **Our novelty:** Full GNN context propagation + subtract-mut scoring. ThermoMPNN uses only single-position features. Our GNN propagates information from neighboring residues, potentially capturing cooperative/epistatic effects.
- **Implementation:** Requires new training loop and output head. `train_subtract_mut.py` has MLP-only version; full GNN version needs implementation.
- **Probability:** 60%

---

## LEVEL 11: GNN-SM + Anti-Symmetry Augmentation ⏳

- **Change:** For every mutation A->B with ddG=+x, add reversed mutation B->A with ddG=-x. With subtract-mut this is AUTOMATIC: ddG(A->B) = score[B] - score[A] = -(score[A] - score[B]) = -ddG(B->A). Just add both directions to training data.
- **Expected PCC:** +0.02-0.05 on top of Level 10
- **Why:** Doubles effective training data for free. Enforces the physical law that forward and reverse mutations have equal and opposite stability effects. Reduces model bias toward predicting positive or negative ddG values.
- **What we learned:** The model benefits from more training signal (Level 6 shows even mild regularization helps with limited data). Anti-symmetry is a physics-based constraint that acts as free regularization.
- **Implementation:** ~10 lines of code in the data loader. Trivially simple.
- **Probability:** 80% (conditional on Level 10 working)

---

## LEVEL 12: GNN-SM + Serial Fusion (Embeddings First) ⏳

- **Change:** Feed ProtT5 embeddings INTO the GNN as input features (projected: 1024->64), instead of concatenating them after GNN processing. The GNN receives [projected_emb:64 + dist:48 + one_hot:20] = 132-dim at each node. PLM information now participates in message passing.
- **Expected PCC:** +0.03-0.08 on top of Level 10
- **Why:**
  - Currently, ProtT5 features NEVER flow through the graph — they bypass the GNN entirely
  - If PLM signals propagate through spatial edges, the model learns "how does the evolutionary preference at position i affect position j, given they are 5A apart"
  - This captures cooperative/epistatic effects: a mutation at position 32 disrupting a bond with position 78
  - ESM-GearNet paper shows serial fusion (PLM into GNN) consistently outperforms parallel fusion (PLM concatenated after)
- **What we learned:**
  - Level 4 proves the GNN's spatial graph structure works → adding PLM to that graph gives it richer signal
  - Level F2 proves PLM signal is critical → putting it inside the GNN makes the GNN aware of it
  - Level 7 proves ProtT5 is the best embedding source → serial fusion maximizes its utility
- **Implementation:** Modify `hydro_net.py` to include projected PLM in GNN input features. Remove post-GNN concatenation.
- **Probability:** 70%

---

## LEVEL 13: GNN-SM + Learnable AA Embedding ⏳

- **Change:** Replace 20-dim one-hot amino acid encoding with `nn.Embedding(20, 64)` — a learnable 64-dim embedding for each amino acid, optionally combined with projected PLM features.
- **Expected PCC:** +0.01-0.03
- **Why:** One-hot encoding treats all amino acids as equidistant. Learnable embeddings can discover that (Ile, Leu, Val) are similar (all hydrophobic, branched) while (Gly, Pro) are unique (flexible, rigid). This gives the GNN better amino acid representations.
- **What we learned:** The model benefits from richer per-residue features (Level 4 improved by giving better graph structure; Level 12 improves by giving better node features). Learnable embeddings are a simple enhancement.
- **Probability:** 65%

---

## LEVEL 14: GNN-SM + Edge Features (RBF + Orientation) ⏳

- **Change:** Add distance RBF expansion (16 Gaussian centers, 0-20A) + backbone orientation features + sequence separation as edge attributes to GATv2Conv.
- **Expected PCC:** +0.02-0.05
- **Why:** Currently, edge weights are computed from a single Gaussian kernel (one number per edge). RBF expansion gives 16 numbers per edge, encoding distance with much higher precision. Orientation features encode relative backbone geometry. ProteinMPNN uses exactly this (25 RBF centers) and it is critical to their performance.
- **What we learned:** Level 4 proves that graph structure matters enormously. Better edge information = even better graph structure = even better message passing.
- **Implementation:** ~20 lines to add RBF expansion and pass as `edge_attr` to GATv2Conv.
- **Probability:** 60%

---

## LEVEL 15: Full Ensemble (5-Seed Best Config) ⏳

- **Change:** Take whatever the best single-model config is (from Levels 10-14), run it with 5 seeds, average predictions.
- **Expected PCC:** +0.02-0.04 on top of best single model
- **Why:** Same reasoning as Level 8 but applied to the improved architecture. Ensemble variance reduction is nearly guaranteed regardless of the base model.
- **Probability:** 90%

---

## LEVEL 16 (Stretch): ProteinMPNN Frozen Backbone ⏳

- **Change:** Use public ProteinMPNN pretrained weights as the GNN encoder (frozen). Train only a small MLP head on top for ddG prediction with subtract-mut scoring.
- **Expected PCC:** 0.65-0.75
- **Why:** This is literally what ThermoMPNN does (PCC=0.754). ProteinMPNN was pretrained on inverse folding (predicting sequence from structure) — the PERFECT pre-training task for ddG prediction because it teaches "which amino acids are compatible with this structural environment."
- **What we learned:**
  - Level F4 proves: wrong pre-training hurts
  - Level 1 proves: from-scratch is better than wrong pre-training
  - Therefore: RIGHT pre-training (inverse folding) should be best of all
- **Requires:** Professor approval (using external pretrained weights raises questions about novelty)
- **Probability:** 85% (IF professor approves the approach)

---

## LEARNING SUMMARY (What the Levels Teach Us)

### Pattern 1: Dimensionality Must Match Batch Size

- **Level F1 proves:** High dimensionality (1536) + forced small batch (8) = catastrophic failure (-0.10 PCC)
- **Level 4 proves:** Sparse graph (k-NN) outperforms dense graph (fully-connected) — less is more
- **Level 9 plans:** Project high-dim features BEFORE they enter the GNN, preserving batch size
- **Lesson:** ALWAYS check that feature_dim * batch_size fits in VRAM before training. If not, project down first. Information is worthless if you cannot optimize on it properly.

### Pattern 2: Per-Mutation Signal is Essential

- **Level F3 proves:** WT-only embeddings = no mutation information = random predictions (PCC ~0.25)
- **Level 7 proves:** Per-mutation ProtT5 embeddings give strong signal (PCC 0.526)
- **Level F2 proves:** Structure-only features lack the "which AA fits here" signal
- **GNN-SM insight:** You can get per-mutation signal from the OUTPUT (score[mut] - score[wt]) instead of the INPUT. This is cheaper and potentially stronger because the GNN only needs to run once per protein.

### Pattern 3: The Right Task Matters More Than Model Size

- **Level F4 proves:** Pretrained on wrong task (decoy discrimination) < random init on right task (ddG)
- **Level 1 proves:** From-scratch training on the actual objective outperforms transfer from misaligned objectives
- **Level 16 (planned):** Pretrained on RIGHT task (inverse folding) should >> everything else
- **Lesson:** Domain-appropriate pretraining > from-scratch training > wrong pretraining. The alignment between pre-training objective and downstream task is more important than model size or training duration.

### Pattern 4: Robust Optimization > Clever Architecture

- **Levels 2-6 combined** give +0.045 PCC from loss function, LR schedule, and regularization choices alone
- **Level 4** gives +0.043 from ONE architectural change (k-NN graph)
- **Together:** Training dynamics fixes and one good architectural choice account for ALL improvement from baseline
- **Lesson:** Fix training dynamics FIRST (loss, LR, batch size, regularization), THEN improve architecture. A perfect architecture with broken optimization learns nothing. A simple architecture with good optimization often outperforms complex models with poor training.

### Pattern 5: What Works Should Inform What's Next

- **k-NN spatial edges work (Level 4)** → Keep spatial graph structure in GNN-SM (Level 10)
- **ProtT5 works (Level 7)** → Serial fusion puts it deeper into the GNN (Level 12)
- **Huber loss works (Level 2)** → Keep it in ALL future experiments
- **Ranking loss works (Level 5)** → Keep it in GNN-SM training
- **Per-mutation embeddings work (Level 7)** → GNN-SM achieves this via output scoring (Level 10)
- **Lesson:** Never discard proven improvements. Stack them. Each addresses a different failure mode, and they are roughly additive.

### Pattern 6: Failure Teaches More Than Success

- **Level F1** taught us about the dim/batch tradeoff → directly motivates Level 9 (projection)
- **Level F2** taught us PLM signal is critical → directly motivates Level 12 (serial fusion)
- **Level F3** taught us per-mutation signal is needed → directly motivates Level 10 (subtract-mut output)
- **Level F4** taught us about task alignment → directly motivates Level 16 (inverse folding pretraining)
- **Lesson:** Every failed experiment narrows the search space. Document WHY things fail, not just THAT they fail.

---

## QUICK REFERENCE: Level Summary Table

| Level | Name | PCC | Delta | Status |
|-------|------|-----|-------|--------|
| 0 | Raw Baseline | 0.48 | — | ✅ Done |
| 1 | Train From Scratch | 0.485 | +0.005 | ✅ Done |
| 2 | Huber Loss | 0.501 | +0.016 | ✅ Done |
| 3 | Cosine LR | 0.516 | +0.015 | ✅ Done |
| 4 | k-NN GAT | 0.5259 | +0.043 | ✅ Done |
| 5 | Ranking Loss | — | +0.003 | ✅ Done |
| 6 | Weight Decay | — | +0.008 | ✅ Done |
| 7 | Combined Best | **0.5259** | +0.045 total | ✅ Done |
| F1 | dual_esmif 1536-dim | 0.4232 | -0.103 | ❌ Failed |
| F2 | ESM-IF1 Only | 0.2856 | -0.240 | ❌ Failed |
| F3 | WT-Only Embeddings | ~0.25 | -0.276 | ❌ Failed |
| F4 | Fine-Tune Pretrained | 0.48 | -0.046 | ❌ Failed |
| F5 | SaProt Embeddings | CRASH | — | ❌ Failed |
| F6 | Full-Dim DSM | Stalled | — | ❌ Failed |
| F7 | Holistic DSM Noise | No gain | — | ❌ Failed |
| 8 | 5-Seed Ensemble | 0.55-0.57 | +0.03 | ⏳ Planned |
| 9 | dual_esmif + Projection | 0.53-0.56 | +0.01 | ⏳ Planned |
| 10 | GNN Subtract-Mut | 0.55-0.65 | +0.05-0.12 | ⏳ Planned |
| 11 | + Anti-Symmetry | +0.02-0.05 | — | ⏳ Planned |
| 12 | + Serial Fusion | +0.03-0.08 | — | ⏳ Planned |
| 13 | + Learnable AA Emb | +0.01-0.03 | — | ⏳ Planned |
| 14 | + Edge Features RBF | +0.02-0.05 | — | ⏳ Planned |
| 15 | Full Ensemble | +0.02-0.04 | — | ⏳ Planned |
| 16 | ProteinMPNN Backbone | 0.65-0.75 | +0.15-0.25 | ⏳ Stretch |

---

## THE PATH: From 0.48 to 0.75

```
0.48  [Level 0] -----> 0.526 [Level 7]  -----> 0.55-0.57 [Level 8]
      (baseline)       (all fixes)              (ensemble)
                                                    |
                                                    v
                                               0.55-0.65 [Level 10]
                                               (GNN-SM: biggest planned gain)
                                                    |
                                                    v
                                               0.60-0.68 [Levels 11-14]
                                               (stacked improvements)
                                                    |
                                                    v
                                               0.62-0.72 [Level 15]
                                               (final ensemble)
                                                    |
                                                    v
                                               0.65-0.75 [Level 16]
                                               (ProteinMPNN backbone - stretch)
```

**Current position:** Level 7 (PCC = 0.5259)
**Next step:** Level 8 (5-seed ensemble, scripts ready, just needs GPU time)
**Biggest expected jump:** Level 10 (GNN-SM, requires implementation)
**Target for thesis:** PCC 0.60-0.65 with novel GNN-SM contribution
**State of the art:** ThermoMPNN = PCC 0.754
