# LORO readout: descriptors vs one-hot, tryptophan held out of TRAINING

**Status: PARTIAL — the descriptor arm has not produced a valid model. The comparison
cannot be scored as designed, and the reason is a confound in the descriptor arm, not an
answer about descriptors.**

## What was run

Two arms differing in exactly one flag, seed 42, W removed from TRAIN only (present in TEST):

| arm | flag | job |
|---|---|---|
| `gld_loroW_onehot` | `--aa_descriptors none` | 21084179 |
| `gld_loroW_desc`   | `--aa_descriptors mordred_pca16` | 21107689 |

Both verified genuinely different, not a silent no-op:

- **Holdout is real in both.** `[LORO]` per-protein tallies are non-zero in both logs
  (e.g. onehot `2K5N 1278->1193 removed 85`; desc `2AME 1221->1157 removed 64`).
  `train.py:loro_report()` raises if the holdout removes zero rows, so a no-op flag could
  not have produced these runs.
- **Descriptors are real in the desc arm.** `[W6] mode=mordred_pca16 K=16
  md5=6c3a2b0a58a8729977809b1eadae3c16`, and the first linear layer widened by exactly
  +16 columns (`fc1_gcn` 52->68 in desc vs 52 in onehot) with healthy non-zero column
  norms. The descriptor block is wired in and receiving gradient.

A first desc attempt (21084180) died instantly on `--aa_descriptors pca16`, a retired
alias; 21107689 is the valid resubmission. This is why the desc arm is ~5h behind.

## The result so far

Validation ddG PCC, per epoch (the metric `run_calib_eval.sh` uses for epoch selection):

| epoch | onehot | desc |
|---|---|---|
| 0 | **0.548** | 0.090 |
| 1 | 0.641 | **-0.028** |
| 2 | 0.640 | 0.022 |
| ... | ... | (still running) |
| 12 | 0.712 | — |

**The descriptor arm has collapsed to a constant prediction.** Its ddG RMSE takes exactly
one distinct value, 2.541, across all three epochs, while onehot's takes ten distinct
values (1.801-1.903). A frozen RMSE equal to the target's spread is what a constant
output produces. The one-hot arm had already reached PCC 0.548 at **epoch 0**, so this is
not a slow start.

The model is not frozen — it is training: weight drift is 0.47/epoch in desc vs
0.45-0.52 in onehot, and desc train loss moves 124 -> 141 -> 304. So the collapse is an
**optimization failure**, not a dead feature.

Three independent confirmations of the constant-prediction reading:
1. RMSE takes exactly one value (2.541) over three epochs; onehot takes ten (1.801-2.142).
2. 2.541 **exceeds** onehot's worst signal-bearing epoch (2.142 at PCC 0.548), which is
   where a constant predictor must land: RMSE = RMS(true ddG).
3. ddG PCC is ~0 (0.090, -0.028, 0.022) while `run_calib_eval.sh` computes it from the
   actual predictions.

**What it is NOT** (checked, so the mechanism is not overstated): there is no numerical
blow-up. No NaN/Inf and no all-zero tensors in any desc checkpoint; the largest parameter-
norm inflation vs onehot is only **x1.10**; and the output head is healthy in both arms
(`fc2` |W| 3.457 vs 3.493, bias/weight ratio 0.129 vs 0.139). The collapse is upstream of
the head and is not divergence-to-infinity.

## Most likely cause: the descriptor block is unstandardized

*(This is the leading hypothesis, established by measurement of the inputs. It is not yet
proven causal — that needs the z-scored re-run below. It is stated as a candidate, not a
finding.)*

`data/aa_descriptors_mordred_pca16.csv` is mean-centered but never scaled, and nothing in
`aa_descriptors.py` or `scripts/build_aa_descriptors_mordred.py` rescales it:

- PC1 std = **2.726**, range **-5.887 .. +6.035**
- the one-hot columns it is appended to are exactly **0/1**
- descriptor block per-residue L2 norm: median **3.71** vs one-hot's exactly **1.0**
  -> the block carries **~13.8x the input energy** of the one-hot block
- PC1 alone injects **~30x** the variance of a one-hot column, into the same layer at the
  same learning rate

`mordred_pca16` has **never** trained successfully in any run on this cluster — this is
its first outing, so there is no precedent that the path can learn at all.

Scale imbalance is a well-established cause of exactly this failure, and it is the only
input-side difference between the arms. But the causal link is **not proven** here: I have
shown the imbalance exists and that the arm collapsed, not that the first caused the
second. The z-scored re-run is what would close that gap.

## Verdict against the four pre-recorded outcomes

The prediction on record (CONTEXT.md CHECKPOINT 26) was *desc will not beat one-hot by
much*, because ProtT5 already predicts held-out-residue hydropathy at R^2 0.704 and the
embedding is in both arms.

**That prediction is NOT yet confirmed, and must not be scored as confirmed.** The
observed outcome is closest to #4 (collapse), but *asymmetric* — only the descriptor arm
collapsed, and for a reason unrelated to the descriptor hypothesis. A collapsed arm is
uninformative about whether descriptors carry information one-hot lacks: it lost to a
learning-rate/scaling mismatch before the hypothesis was ever tested.

Reporting "desc did not beat one-hot, as predicted" from this run would be **confirming
the prediction with a broken arm** — exactly the failure mode the project guards against.

## Honest bottom line

- **The test did not run.** The descriptor arm never produced a usable model.
- **The prediction is neither confirmed nor overturned.** It remains open.
- **The redundancy conclusion stands untouched** on its existing evidence (the ProtT5
  R^2 0.704 probe), and gains no support from this run.

## What would make it a real test

Re-run the desc arm with the descriptor block **energy-matched to the one-hot block**.

A subtlety worth stating, because the obvious fix is the wrong one: plain z-scoring makes
things *worse* on the metric that matters. Z-scoring equalises the columns among
themselves (killing PC1's dominance) but K=16 unit-variance columns give the block a norm
of ~sqrt(16)=4, so block-vs-one-hot energy goes **13.8x -> 17.8x**. The block must be
z-scored **and then scaled by 1/sqrt(K)**.

Both candidate matrices are prepared (in `data_fixed/`, the read-only tree untouched):

| file | per-residue L2 | vs one-hot (1.000) |
|---|---|---|
| `data/aa_descriptors_mordred_pca16.csv` (current) | 2.27 / **3.71** / 6.53 | ~13.8x energy |
| `data_fixed/aa_descriptors_mordred_pca16_z.csv` | 3.33 / **4.22** / 4.36 | ~17.8x — worse |
| `data_fixed/aa_descriptors_mordred_pca16_unit.csv` | 0.83 / **1.06** / 1.09 | **matched** |

No code change is needed — `--aa_descriptor_csv` already overrides the path:

```
--aa_descriptors mordred_pca16 --aa_descriptor_csv data_fixed/aa_descriptors_mordred_pca16_unit.csv
```

**Not submitted.** Launching a new GPU arm is the user's call, and 13 GPU jobs are already
in flight. Note the override was smoke-tested on CPU: it loads at shape (20,16), md5 2396e12cc3605e118c512a7902cfc208,
provenance accepted (header check relaxed by the override, column count still enforced).

Until that runs, the descriptor hypothesis is **untested on LORO, not refuted**.

## Note: a real bug found in `validate()` (affects both arms equally)

`Megascale-fineTuning/train.py:882` — `val_loss += batch_loss` sits **outside** the
per-protein loop (indent 12, same as the `for` at line 850), so the printed
`Validation Loss:` is the **last protein's** loss / `len(val_ds)`, not a mean over the
val set. This is why desc's printed val looked frozen at 0.12675 to five decimals.

Not fixed here: the tree is read-only, both arms are affected identically, and
`run_calib_eval.sh` selects epochs by **ddG PCC**, which `validate()` accumulates
correctly — so epoch selection is not contaminated. But the printed `Validation Loss`
should not be used for model selection anywhere.
