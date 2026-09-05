# DeepEF — Orientation for the Cluster Agent

**You have no prior context. This document is the context. Read it fully before running
anything, and save sections 2, 3 and 7 to memory before your first action.**

Owner: Nissim Brami, M.Sc. student, Ben-Gurion University, supervisor Prof. Chen Keasar.
Predecessor work: Shahar Cohen's M.Sc. thesis (BGU, 2025) and the repository
`github.com/shaharec/DeepPEF`.

---

## 1. What the project is, and what we are actually trying to prove

DeepEF is a graph neural network that acts as a **free-energy function** for proteins. Given a
sequence and a structure it returns one scalar energy. Stability is computed as a difference,
`ΔG = E_unfolded − E_folded`, and the effect of a mutation as
`ΔΔG = ΔG_mutant − ΔG_wild-type`.

This formulation is the point of the project. Most competing methods regress ΔΔG directly and
therefore have no notion of absolute stability at all; an energy function returns a value for
any sequence and structure, and can rank alternative conformations of the same protein. That is
what makes it worth defending.

**The problem.** The model ranks mutations *within* a protein well (per-protein Pearson ≈ 0.71)
and pools poorly *across* proteins (≈ 0.61). Fitting `pred ≈ a_p · true + b_p` for each protein
`p` and removing both parameters with an oracle lifts **every one of 52 checkpoints ever
trained** to 0.77–0.81, independent of raw score. So the ranking information is present; what is
lost is calibration.

- `b_p` is the **offset**. It is exactly the model's error on the wild-type ΔG, because
  `ΔΔG_pred − ΔΔG_true = e(mut) − e(wt)` and `e(wt)` is constant within a protein.
- `a_p` is the **slope**. It is compression: the model under-reacts to mutations, collapsing to
  `a ≈ 0.07–0.18` on de-novo designed and fragile folds. Slope predicts per-protein performance
  better than any other measured quantity (+0.60), better than mean ΔΔG (+0.43) or length (−0.18).

**The thesis: make `a_p` and `b_p` learnable during training, rather than corrected afterwards.**
Post-hoc correction was tried and *hurts* (0.655 → 0.573). Every experiment is judged by whether
it moves these two quantities without destroying within-protein ranking.

That is the whole goal. Not "beat ThermoMPNN". Not "reach 0.80" — that figure is an oracle
ceiling and is not achievable; never quote it as a target.

---

## 2. Settled facts — do not re-derive these, do not re-litigate them

Each row has a source. If you doubt one, open the source. Do not guess.

| Fact | Source |
|---|---|
| **The honest reference is `calib_ctrl`: pooled ΔΔG PCC 0.606, per-protein 0.711** | `Megascale-fineTuning/sbatch_wtanchor.sh` line 7 |
| **0.655 is test-peeked and was retracted by its author.** "proper val-selection gives ~0.63". Also 0.655 (epoch 2) and 0.738 (epoch 11) come from *different checkpoints* | `RESEARCH_LOG.md` lines 386–390; defence talk slide |
| **The reference runs through `Megascale-fineTuning/train.py`**, not `pnas_train.py` | `sbatch_wtanchor.sh` line 42 |
| `train.py` has `--val_frac`, working `--epochs`, `--no_pretrain`, `--no_freeze`, `--wt_anchor_weight`, `--designed_weight`, `--affine_calib`, `--run_tag` | its argparse |
| `train.py` does **not** accept `--one_mut` or `--dg_ml`. Both filters are hardcoded: `ddG_ML == '-'` removal at lines 244 and 284, multi-mutation removal at 247. Passing those flags is an argparse error, not a bug | `train.py` |
| **Freezing is the default** in `train.py`. `--no_freeze` is mandatory together with `--no_pretrain` — its own help text calls it "the correct baseline for random init". Random init **plus** freeze trains only `fc1`/`fc2` on random GNN features and produces a flat, low curve | `train.py` line 63 |
| Mini-batch size is **hardcoded 64** in `train.py` line 40. A `--mini_batch_size` flag existed briefly (commit `2ee331d`) and was removed the next day (`6001848`). **All reference numbers were produced at 64** | git history |
| Checkpoints are written every epoch as `kf_{fold}_epoch_{N}.pt` under `Megascale-fineTuning/models/<MODEL_NAME>/`, where `MODEL_NAME` ends in `kf_<run_tag>` | `train.py` lines 46–47, 103–104, 448 |
| The **WT anchor was already run** at weights 0.3 and 1.0. Offset std: control 0.29 → 0.26 → 0.25. Adding designed-fold reweight ×3: **0.19**. But in that arm the slope distribution collapses — maximum falls from ~1.5 to 0.85 | `eval_results/figures/plotB_wtanchor_*.png` |
| **No numeric results for those runs were ever committed** — only the figures. The PCC values are unknown | repo contents |
| Across-protein variance in ΔG sits **77–88% in `E_unfolded`**; `corr(E_u, wt_err) = 0.865`; `var(E_u)` is 6–10× `var(E_f)`; `corr(wt_err, length) ≤ 0.12` | our analysis of `analysis/wt_dg_error/task6_gat_gcn_energy.csv` |
| The **spatial (GAT) branch carries 98–99%** of ΔG; the chain (GCN) branch carries −0.063 with std 0.007 | same file |
| **AFRC ≡ analytic Flory coil**: correlation 0.99995, 0.38% mean difference, fitted `b` = 5.82. Run **one** coil configuration, not two | our measurement over 10 wild-type sequences |
| Non-standard amino acids: **0 of 19,615 residues**. The unknown→alanine fallback never fires | our count |
| Zero-shot pretraining lifting ΔG 0.65 → 0.82 is a **validation** figure (thesis Table 3.3), not comparable to the 0.51 held-out figure. Separately, fine-tuning from the decoy-discrimination checkpoint *hurts* ΔΔG (0.48 vs 0.526 random init) | thesis; `RESEARCH_LOG.md` |
| **Already tested and negative:** pooled-correlation loss, length normalisation, learned readout aggregation, post-hoc offset correction, additional MegaScale data, FireProtDB/ProTherm (leakage against S669), ESM-2 fusion, graph-transformer backbone | `RESEARCH_LOG.md` |
| **Contact-order head is dead:** descriptor R = 0.40 on both epochs and upward-biased at n=28; and subtracting one per-protein constant from every variant cancels exactly in `output − output[0]`, so it cannot move ΔΔG at all | our analysis |
| Near-native decoys have **no data** for the fine-tuning proteins; only the pretraining corpus and 3DRobot have decoys | `RESEARCH_LOG.md` |

---

## 3. Hard rules

1. **`github.com/shaharec/DeepPEF` is READ-ONLY.** Clone, fetch and read only. Never push,
   commit, tag, or open a pull request there. This is absolute.
2. **Do not push to `github.com/nissimbrami/DeepEF-Thesis`** without explicit, per-action
   approval from Nissim. Work on a local branch, report, wait.
3. **Every run gets a unique `--run_tag`.** Never overwrite a checkpoint directory.
4. **Never report a number from in-training `validate()`.** Results come only from
   `Megascale-fineTuning/run_calib_eval.sh` → `validation/score_runs.py`. The in-training ΔΔG in
   `pnas_train.py` subtracts `val_dg[0]` — one global wild-type reference for every protein —
   and is not a ΔΔG at all.
5. **Every number carries three qualifiers:** pooled or per-protein · ΔG or ΔΔG · which split
   and which selection. A number without all three is not a result.
6. **Selection on validation, never on test.** `--val_frac 0.1` exists for this. The test set
   may be touched exactly once per run, as a reproduction check, described in §5.
7. **A stopped run is cheaper than a finished wrong run.**

---

## 4. Why the metric definitions matter more than they look

Four numbers circulate in this project and they are **not comparable to each other**:

| Number | Quantity | Split | Note |
|---|---|---|---|
| 0.82 | ΔG | validation | thesis Table 3.3, protocol table |
| 0.51 | ΔG | held-out (DeepEF1 split) | thesis Table 3.6 |
| 0.54 | ΔΔG | held-out | thesis Table 3.4 |
| 0.655 | ΔΔG pooled | held-out | **test-peeked, retracted** |
| **0.606 / 0.711** | **ΔΔG pooled / per-protein** | **held-out, val-selected** | **the reference** |

Most of the confusion in this project has come from comparing two of these. Before you compare
any two numbers, check that all three qualifiers match.

The correct evaluation path is:

```
train.py                          → checkpoints kf_all_epoch_N.pt
run_calib_eval.sh <log> <dir> <tag>
   ├─ picks best epoch by VALIDATION ddG PCC parsed from the log
   ├─ evaluate.py --trained_model_path <ckpt> --model_name eval_results/abl_<tag>_e<N>
   │     → CSV with columns protein, deltaG, pred_deltaG, ddG, pred_ddG
   └─ validation/score_runs.py <tag>
         pooled = pearson(df.ddG, df.pred_ddG)
         PP     = mean over df.groupby('protein') with len(g) >= 3
```

---

## 5. The procedure

Six steps. Each has a gate. **Do not proceed past a failed gate — report it and stop.**

### Step 0 — environment

Verify SLURM access, the `keasar` QOS, the `esm2_env_py38` conda environment, CUDA visibility,
and which partitions accept your jobs. Nodes on this cluster include `cs-4090-*` (preferred),
`cs-6000-*`, `cs-3090-*`, `cs-2080-*`, `cs-1080-*`, and `cs-cpu-*` (CPU only).

Shahar's launchers target `--partition rtx6000 --qos keasar --gpus=1 --cpus-per-task=8` and
exclude `cs-6000-01..04` and `cs-cpu256-01`. If `rtx6000` is unavailable to you, the 4090
partition is the better card anyway; adjust and record the change.

**Gate:** you can submit a trivial GPU job and it runs.

### Step 1 — data

`train.py` reads:

```
./data/Processed_K50_dG_datasets/training_data/<protein>/
      coords_tensor.pt, deltaG.pt, mask_tensor.pt, one_hot_encodings.pt,
      prott5_embeddings/prott5_embedding_<i>.pt
./data/Processed_K50_dG_datasets/mutation_datasets/<protein>.csv
./data/Processed_K50_dG_datasets/Pnas_filtering/{train_proteins.csv, pnas_mutations.csv}
./data/ThermoMPNN/mega_test.csv
```

**`calib_ctrl` was produced on this cluster**, so these tensors existed here under Shahar's
account. Search for them before downloading anything. If they are present and readable,
symlink `./data` and you are done — no conversion, no adapter.

If they are not available, the dataset also exists on Hugging Face as
`nissimb/deepef-megascale` (~83 GB) in a **different layout** (`crd_backbone.pt`,
`seq_one_hot.pt`, `proT5_emb.pt`, `mask.pt`, mutations under `mutation_files/`). In that case
write a subclass of `train.py`'s `AllProteinValidationDataset` that overrides **only file I/O** —
never the filtering, the test-split logic, or the returned dictionary — and then prove
equivalence on three proteins (one small, one large, one designed `HEEH_*`) against
`new_dataset.MSDataset(one_mut=True, dg_ml=True)`:

1. identical mutation count after filtering
2. `delta_g` tensors equal
3. `coords` shapes and values equal
4. embeddings `[n_mut, L, 1024]`, `torch.allclose`
5. one-hot equal
6. row 0 is the wild type in both

**All six must pass. If one fails, that difference is the finding — report it, do not patch
around it.**

**Gate:** 368 protein directories reachable, 28 proteins in `mega_test.csv`, and either the
native layout or all six equivalence assertions passing.

### Step 2 — smoke test

The exact reference flags with `--epochs 1`. Not `--max_folds 1` — `--full_data` bypasses
k-fold, so that flag does nothing.

Before training starts, assert:

1. protein counts: 28 test, ~34 validation, ~306 train
2. `batch['delta_g'][0, 0]` — the wild-type ΔG — present and finite for **every** training
   protein. Both the ΔΔG objective and the WT anchor subtract it
3. `not (no_pretrain and freeze)` — see §2; this combination is a known dead configuration
4. the first GCN layer's weight standard deviation looks like random initialisation, not a
   loaded checkpoint, confirming `--no_pretrain` took effect
5. on the first minibatch: `ddg_loss` is nonzero and `l1_loss` is not driving `data_loss`
6. parameter count ≈ 703,805, within 5%
7. the epoch count in effect equals what you passed
8. after epoch 0, `kf_all_epoch_0.pt` loads in a fresh process and runs one forward pass

**Gate:** all eight pass, one epoch completes on GPU.

### Step 3 — reproduce the reference

```
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze --loss_mode ddg \
  --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
  --wt_anchor_weight 0 --designed_weight 1 \
  --val_frac 0.1 --epochs 15 --run_tag calib_ctrl_repro
```

Chain `run_calib_eval.sh` as a dependent job with `--dependency afterok:<jobid>`, exactly as
`sbatch_wtanchor.sh` does.

**Monitor the live run** and stop it on any of:

| Signal | Stop condition |
|---|---|
| training `ddg_loss` | flat or rising through epoch 2 |
| NaN or Inf anywhere | first occurrence |
| validation ΔΔG PCC-PP | below 0.30 at the end of epoch 2 |
| within-protein `std(pred)/std(true)`, median over proteins | below 0.15 — slope collapse |
| epoch wall-clock | more than twice the first epoch |
| `cov(): degrees of freedom <= 0` warnings | more than 20% of minibatches — the ranking term is dead |

**One reproduction check at epoch 2.** Once `kf_all_epoch_2.pt` exists, evaluate it on the
28-protein test set through `evaluate.py` and read the pooled PCC. Shahar's honest runs sit near
0.64 at epoch 2. **Below 0.55, stop** — the run will not reach 0.606, and finding that out at
epoch 2 costs one hour instead of ten. This is a reproduction check, **not** epoch selection;
selection stays on validation.

**Gate:** final pooled ≈ 0.606, PP ≈ 0.711. If not, report the number with mini-batch size,
seed, and all three metric qualifiers, and stop. The regime difference must be found before
anything else runs.

### Step 4 — seed variance

`RANDOM_SEED = 42` is hardcoded at `train.py` line 33. Add a `--seed` flag that sets Python,
NumPy and Torch seeds, then run Step 3's configuration at five seeds.

**This is the most important unmeasured number in the project.** Lever effects are expected in
the range 0.01–0.05. If σ ≥ 0.02, single-seed comparisons are meaningless and the experiment
plan must narrow to three levers at five seeds each rather than eight at one.

**Gate:** mean ± σ for pooled and PP, reported.

### Step 5 — the first lever

`--wt_anchor_weight` at 0.3, 1.0 and 3.0. The first two were run by Shahar; 3.0 was not, and the
trend in offset std (0.29 → 0.26 → 0.25) has not saturated.

**The deliverable is not a single score. It is the trade-off curve:** offset std `std(b)`, the
distribution of per-protein slopes `a`, and per-protein PCC, each as a function of anchor weight.
That curve does not exist anywhere and is a result in its own right, because the one arm that
reduced offset most (`w=1.0` with designed-reweight ×3, std 0.19) also collapsed the slopes.

**Gate:** three weights complete, trade-off table reported.

---

## 6. What comes after Step 5

Do not start any of these before Step 4's σ is known and Nissim has decided how many levers fit
the remaining time.

| Lever | What | Why |
|---|---|---|
| L2 | `--designed_weight` 3 and 5, alone and crossed with the best anchor weight | Separates "the anchor works" from "reweighting works" from "they compose". Designed folds are 121 of 368 training proteins |
| L3 | **Slope term**: penalise `abs(std(pred_ddG) − std(true_ddG))` within each protein | Nothing on anyone's list addresses slope, yet slope predicts per-protein performance best. The anchor fixes the first moment; this fixes the second. This is the thesis's own contribution |
| L4 | **Analytic Flory coil** for the unfolded reference state | The unfolded state is currently built from the *folded* coordinates with non-local contacts deleted — a folded protein with holes, not an unfolded chain. And 77–88% of the across-protein variance lives there. Replacing it with `d = b·abs(i−j)^ν`, `b ≈ 5.8`, `ν = 0.5`, makes the reference depend only on sequence separation |
| L5 | Structural features from the group's biologist: burial, metal coordination sphere, secondary structure, disulfides, oligomeric state | The only items that add information the model does not have in any form. Two requirements from bugs already found: burial must be **zero in the unfolded state**, or it cancels in `E_u − E_f` and cannot express the hydrophobic driving force, which *is* the change in buried area; and normalise by a constant, not by chain length |

**L4 carries a falsifiable mechanism prediction, and it should be reported whichever way it comes
out.** Today the unfolded energy is carried almost entirely by the spatial branch (98–99%) while
the chain branch contributes nothing — backwards for a state in which only chain-local terms
should survive. The coil makes the unfolded distances a function of sequence separation alone, a
chain-local quantity. **Predict: after the coil, `corr(E_u, GAT-only)` falls, `corr(E_u,
GCN-only)` rises, and `var(E_u)` shrinks.** Regenerate the branch-energy columns from the new run
and check.

**Its honest caveat:** the coil changes only the distance and bonded feature blocks. The one-hot
encoding and the ProtT5 embedding still feed the unfolded pass, and since length explains almost
none of the variance, the remainder is sequence content through exactly those two channels. If
the coil barely moves `var(E_u)`, the follow-up is to zero the ProtT5 block in the unfolded pass
and measure again.

---

## 7. Standing orders for how you work

**Order 1 — premise check before any run longer than an hour.** Write one line: entrypoint ·
reference number · source file and line · date of that source. If a newer artifact in the repo
touches the same quantity, the premise is stale. Thirty seconds. This exact check would have
prevented a wasted seven-hour run earlier in this project.

**Order 2 — newest artifact wins.** When the research log, a launcher script and a code comment
disagree, the most recently committed one is current.

**Order 3 — name the metric before reading the number.** See §4.

**Order 4 — investigate and decide; do not present menus.** You have the repository and the
cluster. When you hit a fork, read the code, decide, and state the decision with its evidence. A
menu is appropriate only when the choice is genuinely about risk appetite rather than facts.

**Order 5 — close every loop.** If a diagnostic command fails, fix it and re-run before moving
on. A failed script that is silently abandoned costs more than it saves.

**Order 6 — report format.** After each step write a short report containing: the exact command
line, the SLURM job ID, node and wall-clock, the three-qualifier metric table, `std(b)` and the
slope distribution, and anything that surprised you. Commit locally. Do not push without
approval.

---

## 8. Anticipated questions

**Why not use `pnas_train.py`? It runs.** It has no validation split, its `--epochs` flag is dead
(the schedule is hardcoded at 20 frozen plus 60 unfrozen epochs), it has no anchor, and its
in-training ΔΔG uses one global wild-type reference. No honest number ever came from it.

**Can I select the epoch on the test set?** No. Selection is on validation. The test set is
touched once per run as a reproduction check.

**What if the reference does not reproduce?** That is not a failure, it is a finding. Report the
number with mini-batch, seed and all three qualifiers, and stop.

**Should I regenerate the K50 tensors if they are missing?** Only as a last resort. That means
running ProtT5 over every variant to reproduce numbers that already exist in another layout.
Prefer finding the existing tensors, then the adapter with its equivalence gate.

**Why five seeds and not one?** Because the effects we are chasing are the same size as the
seed noise, and nobody has measured that noise. Step 4 exists to find out.

**What is the expected final outcome?** From the 0.606 reference: pooled ΔΔG around 0.68, range
0.63–0.74; per-protein around 0.76. The offset levers overlap heavily — they attack one quantity
— so their effects do not add. The slope term is the one that could add rather than overlap.
**And the number is not the deliverable.** A trade-off curve showing how offset and slope respond
to the anchor, plus a mechanism test on the unfolded reference, is a stronger result than a
higher number that cannot be attributed to anything.
