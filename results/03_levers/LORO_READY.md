# LORO — leave-residue-out harness. BUILT, CPU-VALIDATED, NOT SUBMITTED.

Everything below was run. Nothing here was submitted to SLURM. The sbatch lines in
section 6 are written out and deliberately NOT executed.

---

## 0. Why a proxy at all — the zero, verified

Ofir Ezrielev's contribution is that the physicochemical descriptor space is CONTINUOUS,
so a model trained on canonical residues can place a NON-CANONICAL one by interpolation.
The obvious test — predict non-canonical effects on MegaScale — is impossible, and this
was verified rather than assumed:

    python scripts/count_noncanonical.py

    mutation CSVs scanned : 862
    rows                  : 733945
    mutation codes parsed : 451514
    distinct chars in aa_seq  : ACDEFGHIKLMNPQRSTVWY
    distinct WT-side letters  : ACDEFGHIKLMNPQRSTVWY
    distinct MUT-side letters : ACDEFGHIKLMNPQRSTVWY
    TOTAL NON-STANDARD RESIDUE OCCURRENCES: 0

**Zero. Confirmed over 451,514 parsed mutation codes and 700,553 sequences.** The alphabet
closure has been removed and a 25-row open table (SEP/TPO/PTR/MSE/HYP) loads, but there is
no data on this corpus to score it against. The claim is a GENERALISATION claim, and the
only runnable test is the canonical proxy: hold a canonical residue out of training
entirely, then predict mutations to it from descriptor space alone.

One-hot cannot represent a held-out residue at all — its column never fires in training,
so its coefficient is never updated. Descriptors can, because the residue is still a point
among trained neighbours. That contrast is the experiment.

---

## 1. PREDICTIONS, STATED BEFORE RUNNING

A test that predicts success everywhere is not a test. These are committed in advance.

| arm | residue | prediction | why |
|---|---|---|---|
| **W** | tryptophan | **descriptors BEAT one-hot** | Large, aromatic, distinctive. Its 5 nearest descriptor neighbours (Y 22.7, F 25.6, H 34.5, R 38.2, K 43.1) are ALL in training, so interpolation has real support. Aromaticity, ring atoms, pi-electrons and volume are axes trained on F/Y/H — W is inside their convex hull. |
| **P** | proline | **descriptors do NOT beat one-hot** | Proline's effect is BACKBONE GEOMETRY — a pyrrolidine ring locking phi and eliminating the amide NH. That is a topological constraint, not a position in a physicochemical descriptor space. |

The P prediction was sharpened by the measured geometry, and this is the strongest part of
the control: **P's nearest descriptor neighbours are V (26.3), I (27.8), T (28.0), L (29.5)
— aliphatics.** The descriptor space asserts "P is a small aliphatic", which is exactly
wrong about the only thing that makes proline special. So P is not merely a residue we
expect to do worse on; it is one where the descriptor space is *confidently misleading*.
If descriptors help on P as much as on W, the W result is not interpolation — it is some
generic regularisation benefit, and the claim is not supported.

**Falsification condition, stated in advance:** if dSpearman(W) <= 0, the continuity claim
fails on this corpus and is reported as failed. If dSpearman(P) >= dSpearman(W), the W
result does not demonstrate interpolation and must not be written up as such.

---

## 2. The metric: Spearman rho on held-out-residue mutations only

**Why ddG is legitimate here, under the project's metric rule.**

The rule is: ddG cancels anything identical between WT and mutant, so reference-state and
whole-protein levers must be scored on dG or b_p. A mutation *to* tryptophan genuinely
differs between WT and mutant — the mutant residue IS the held-out object, and it is
present on exactly one side of the difference. It therefore does NOT cancel, and ddG is
the correct target. (This is the same reason a_p is legitimately measurable on ddG: it is
a within-protein quantity, not a shared offset.)

**Why Spearman and not PCC.** This is a within-protein RANKING question: can the model
order mutations into an unseen residue? Spearman is invariant to any monotone transform of
the predictions, so an arm that gets the ORDER right but the SCALE wrong still scores. That
is the right separation, because the descriptor arm's already-measured effect is on the
OFFSET (b_p), and we must not let an offset improvement masquerade as a ranking one.

**A known structural trap, carried forward from the earlier ridge probe.** With a SINGLE
held-out residue, every held-out row has the same mutant residue, so the mutant-side
feature block is CONSTANT across them. A constant block shifts predictions uniformly; it
cannot re-rank them. Correlation is invariant to that shift, so **a single-residue holdout
scored on correlation alone is structurally blind to the mutant encoding**, and rho ~ 0 is
the expected reading, not a null result.

This is why the harness reports rho **per protein** and pools across proteins, and why the
multi-residue arm (W,F,Y) exists. Within one protein the WT side still varies across
positions, so per-protein rho is not degenerate; but the honest reading of a single-residue
holdout is that **b_p is the sensitive metric and Spearman is the conservative one**. Both
are reported. Anyone quoting a single number from this experiment should quote both.

---

## 3. The holdout — implemented and PROVEN to remove what it claims

Flag added to `Megascale-fineTuning/train.py` (the real fine-tuning entry point):

    --holdout_residues W

Semantics, each one deliberate:

* **TRAINING set only.** The filter sits in `load_protein_data` and NOWHERE else.
  `load_test_protein_data` has a near-identical block that is deliberately left unpatched,
  so held-out rows survive in test and there is something to score. Verified by parsing the
  installed source and asserting which loader body contains the guard.
* **Both directions.** Removes mutations INTO the residue AND mutations FROM it. Removing
  only "into" would leave X->W rows in training, showing the model tryptophan as the
  residue being mutated away from — the "never seen" premise would be false.
* **The WT row is NEVER removed.** `mut_type == 'wt'` does not match the substitution
  regex. Row 0 is the per-protein reference that b_p and a_p are defined against; dropping
  it would silently change the calibration of every protein and confound the arm.
* **Empty default = byte-identical to baseline.**
* **Zero-removal tripwire.** `loro_report()` RAISES if the holdout removed 0 rows. A flag
  that claims to remove a residue and removes nothing would train an arm silently identical
  to baseline and be written up as a null result — this project's signature failure mode.

### Counted before and after, on the real CSVs

    python scripts/loro_cpu_validate.py --holdout W
    python scripts/loro_cpu_validate.py --holdout P

| holdout | rows before | rows after | removed | INTO | FROM | proteins bitten | leaked | WT kept |
|---|---|---|---|---|---|---|---|---|
| **W** | 733,945 | 703,699 | **30,246 (4.12%)** | 23,531 | 6,715 | **368/368** | **0** | 2,149 |
| **P** | 733,945 | 695,750 | **38,195 (5.20%)** | 23,068 | 15,127 | **368/368** | **0** | 2,149 |

"Leaked = 0" is the complement check: after the holdout, ZERO surviving training rows
mention the held-out residue on either side. Both validations report **ALL PASS**.

**A denominator that had to be corrected.** The first run reported "368/862 proteins" and
failed its own check. Investigation: **494 of the 862 mutation CSVs are EMPTY (0 rows)**.
They are not proteins lacking tryptophan — they have no rows at all and no corresponding
`training_data` tensor directory, so the training loader never sees them. The 368 non-empty
CSVs are exactly the 368 `training_data` directories, and the holdout bites in **368 of
368 = 100%** of them. Scoring against 862 would have reported a spurious 43% and hidden a
real no-op behind arithmetic. The denominator is now the non-empty set, and the check
asserts the holdout bites in EVERY one.

---

## 4. CPU validation on random tensors — ALL PASS

`scripts/loro_cpu_validate.py` — no GPU, no checkpoint, no training.

* **V1** holdout removes a non-zero count, in every non-empty protein — PASS
* **V2** removes both directions, never the WT row, nothing leaks — PASS
* **V3** guard is inside `load_protein_data`, NOT inside `load_test_protein_data`; the
  zero-removal tripwire is defined and raises — PASS
* **V4** Spearman scorer on random tensors: matches `scipy.stats.spearmanr` to 1e-10
  (0.663307 vs 0.663307); invariant under a monotone transform; sign flips under negation;
  returns 1.0 on perfect rank agreement; handles ties by average ranks; returns NaN rather
  than crashing on constant input and on n<3; the MegaScale `'-'` sentinel becomes NaN and
  is dropped, never 0.0 — PASS
* **V5** end-to-end scoring on RANDOM predictions of the real shapes: 332 scorable
  proteins, 18,906 held-out rows, **median rho = −0.0032** for a random predictor (it MUST
  be ~0, and is — the harness is not leaking the label), and the scorer returns exactly 1.0
  when handed the labels as predictions, proving it reads the array it is given — PASS

The V5 pair is the guard against this project's signature failure mode: a harness that
"runs, completes and reports a number" while never reading the feature. A random predictor
scoring ~0 AND a self-prediction scoring 1.0 together show the number tracks the input.

### V6 — the PATCHED LOADER ITSELF, end-to-end (the anti-"feature never read" guard)

The checks above validate the split logic. This one executes the **actual patched
`load_protein_data`** by exec'ing train.py's module body under `--holdout_residues W` and
calling the real dataset method. This project's signature failure mode is code that runs,
completes and reports a number while the feature is never read; that cannot be ruled out by
testing a reimplementation.

    [LORO] holding out of TRAINING (both directions): W
    [LORO] 2PTL           rows   3549 ->   3363  (removed   186)
    [LORO] 2RU9           rows   1209 ->   1147  (removed    62)
    [LORO] EEHEE_rd4_0256 rows    817 ->    756  (removed    61)
      2PTL           delivered 3363 rows | tensors aligned OK | W-touching remaining: 0
      2RU9           delivered 1147 rows | tensors aligned OK | W-touching remaining: 0
      EEHEE_rd4_0256 delivered  756 rows | tensors aligned OK | W-touching remaining: 0
    TOTAL W-touching rows delivered to TRAINING: 0   (must be 0)
    TEST 2K5H: 3549 rows, W-touching STILL PRESENT: 240   (must be >0)

Three things are established at once, on the real code path:

* **training receives ZERO W-touching rows** — the holdout is actually applied;
* **`one_hot`, `delta_g` and the embedding tensors stay row-aligned with the mutation
  list** (asserted, not eyeballed) — the filter does not silently desync the tensors from
  the labels, which would corrupt every target without raising;
* **the TEST protein still has 240 W-touching rows** — the eval set survives, so there is
  something to score.

**Both control states also verified:**

* **Zero-removal tripwire FIRES:** forced to a 0-removal state, `loro_report()` raises
  `--holdout_residues W removed ZERO rows across 7 proteins. The flag is a no-op...`
* **Flag OFF is inert:** no flag -> `HOLDOUT_RESIDUES = set()`, `loro_report()` is a no-op,
  and the same protein delivers its full **3,549 rows including 186 W-touching** — i.e.
  unfiltered baseline behaviour.

---

## 5. THE GPU GATE: is the descriptor block already inside ProtT5? — NO.

`scripts/loro_redundancy.py`. Pure CPU, seconds. This gates whether the GPU spend is
justified at all: if a linear map from the ProtT5 embeddings already in the feature vector
reproduces the descriptor vector for a residue type it never saw, the descriptor block is
redundant and LORO would only be re-encoding information already present.

**The split must be residue-disjoint, and this is the methodological core.** Every
occurrence of tryptophan shares ONE descriptor vector — the W row of the table. A ROW-RANDOM
split therefore puts the same target in train and test, and the regression only has to
recognise "this is a W" (trivial: ProtT5 certainly encodes residue identity) and emit the
memorised vector. So the honest split holds out a residue TYPE: fit on 19, predict the
20th, whose descriptor vector was never seen.

Residue identity is taken from `one_hot_encodings.pt`, not from parsing `aa_seq`, because
`aa_descriptors.py` guarantees descriptor row *i* == one-hot index *i* for *i* < 20 — the
join is exact and needs no letter mapping that could silently mis-align.

### Held-out R², descriptor target = PCA-16 block, 1024-dim ProtT5, 40 proteins

    MEAN   R^2 (residue-disjoint) : -0.0283
    MEDIAN R^2 (residue-disjoint) : -0.0440
    W  (a LORO arm)               : -0.1641
    P  (a LORO arm)               : -0.0972

    R^2 (row-random, CONTAMINATED): +1.0000
    contamination gap             : +1.0283

**Held-out R² is approximately ZERO, and slightly negative** — a linear map from ProtT5
predicts an unseen residue's descriptor vector *worse than predicting the average amino
acid*. Only 4 of 19 residues exceed +0.17.

Stable across three orders of magnitude of ridge regularisation:

| lambda | mean R² | median R² | W | P |
|---|---|---|---|---|
| 0.1 | −0.0213 | −0.0507 | −0.1695 | −0.0830 |
| 1.0 | −0.0283 | −0.0440 | −0.1641 | −0.0972 |
| 10.0 | −0.0197 | −0.0480 | −0.1665 | −0.0815 |
| 100.0 | −0.0089 | −0.0303 | −0.1436 | −0.0705 |

**VERDICT: NOT redundant. The descriptor block is a genuinely new channel and LORO is
worth the GPU.**

### Two honest caveats on this number

1. **The row-random +1.0000 is memorisation, not a code bug.** Verified explicitly: with 19
   distinct feature vectors replicated into 400 rows, every test residue type also appears
   in train (`TEST types NOT in TRAIN: []`), so each test row shares its exact feature
   vector AND target with a train row. A 1024-dim ridge interpolates 19 points exactly.
   **The +1.03 contamination gap is the measurement of how badly a row-random split would
   have misled us** — it would have reported "descriptors are perfectly redundant, do not
   run LORO" and killed a valid experiment.
2. **Cysteine is missing (19 of 20 types).** C occurs once in 40 proteins — MegaScale's
   designed mini-domains largely avoid it. That is a real property of the corpus, not a
   bug, but it means the probe covers 19 residues and C cannot be a LORO arm here.
3. This probe tests LINEAR decodability of a context-free per-residue mean. A nonlinear
   readout, or one using context, could do better. "Not linearly redundant" is the claim;
   "carries no shared information" is not.

---

## 6. THE SBATCH LINES — WRITTEN, NOT SUBMITTED

**DO NOT RUN THESE HERE.** Five trainings and ~94 queued jobs are in flight. These are the
exact lines, for someone else to submit deliberately.

Each arm is two runs differing ONLY in `--aa_descriptors`, with the data split held fixed
by an identical `--holdout_residues`. That single-variable contrast is the whole experiment.

```bash
# Run from the repo root. Conventions (partition/QOS/exclude/module load, train.py invoked
# from the repo root, the --no_pretrain --no_freeze --val_frac --epochs recipe) are copied
# from cluster_run/scripts/05_anchor_sweep.sh so these cells are comparable to the
# factorial that is already running. Only --holdout_residues and --aa_descriptors differ.

cd /home/nissimb/DeepPEF
PARTITION=rtx6000; QOS=keasar; GPU_SPEC=--gpus=1
EXCLUDE="--exclude cs-6000-01,cs-6000-02,cs-6000-03,cs-6000-04,cs-cpu256-01"
SEED=42
mkdir -p logs

# HOLD is the arm: W = the positive prediction, P = the negative control.
# DESC is the single variable under test: none = one-hot, pca16 = descriptors.
for HOLD in W P; do
  for DESC in none pca16; do
    RUN_TAG="loro_${HOLD}_${DESC}_s${SEED}"
    sbatch --parsable --job-name "DeepEF_${RUN_TAG}"       --partition $PARTITION --qos $QOS $GPU_SPEC --cpus-per-task=8 --time 3-00:00:00 $EXCLUDE       --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out"       --wrap "module load anaconda; source activate esm2_env_py38;               export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;               python Megascale-fineTuning/train.py                 --full_data --no_pretrain --no_freeze --loss_mode ddg                 --pooled_corr_weight 0 --dg_length_norm none --affine_calib                 --wt_anchor_weight 0 --designed_weight 1                 --holdout_residues ${HOLD} --aa_descriptors ${DESC}                 --val_frac 0.1 --epochs 15 --seed ${SEED} --run_tag ${RUN_TAG}"
  done
done
```

Four cells: `loro_W_none`, `loro_W_pca16`, `loro_P_none`, `loro_P_pca16`. Within an arm the
two cells differ ONLY in `--aa_descriptors`; the data split is pinned identical by
`--holdout_residues`, so the contrast is single-variable.

Note `--wt_anchor_weight 0`: the anchor is held OFF here on purpose. The replication in
CHECKPOINT 8 showed the anchor suppresses the exposure/slope coupling monotonically
(0.3 -> 0.587, 1.0 -> 0.471, 3.0 -> 0.400), so switching it on would move a second thing
at once and confound the encoding contrast this experiment exists to measure.

Optional third arm, only if the single-residue Spearman comes back structurally flat
(section 2): holding out three residues makes the mutant side vary again so rank
correlation can discriminate. Same block with `HOLD` replaced by the literal `W,F,Y`
(quote it: `--holdout_residues "W,F,Y"`, and use a tag without commas).

**Before submitting, confirm on the login node (all CPU, seconds):**

```bash
cd /home/nissimb/DeepPEF && conda activate esm2_env_py38
python scripts/loro_cpu_validate.py --holdout W    # must end ALL PASS
python scripts/loro_cpu_validate.py --holdout P    # must end ALL PASS
python scripts/gate_g4_cpu.py                      # must print dG=-0.0030 width=1092
```

**Reading the result when it lands** — decided in advance so the answer is not fitted after
the fact:

| outcome | conclusion |
|---|---|
| dSpearman(W) > 0 AND dSpearman(P) <= 0 | **The continuity claim is supported.** Descriptors interpolate a residue they never saw, and fail exactly where descriptor space is uninformative about the mechanism. |
| dSpearman(W) > 0 AND dSpearman(P) > 0, comparably | NOT interpolation. Some generic benefit of a richer encoding. Do not claim the continuity result. |
| dSpearman(W) <= 0 | The claim is not supported on this corpus. Report as failed. |
| both ~0 | Check b_p before concluding — a single-residue holdout is structurally blind on correlation (section 2). |

---

## 7. Files

| path | what |
|---|---|
| `Megascale-fineTuning/train.py` | `--holdout_residues` flag, train-loader-only filter, zero-removal tripwire |
| `Megascale-fineTuning/train.py.pre_loro` | pre-patch backup (md5 71dd176cd7810ad53605dcb864346231) |
| `scripts/patch_loro.py` | the patch, verbatim anchors, each asserted to match exactly once, idempotent |
| `scripts/loro_cpu_validate.py` | V1–V5 CPU validation |
| `scripts/loro_redundancy.py` | residue-disjoint descriptor/ProtT5 redundancy probe |
| `scripts/count_noncanonical.py` | the zero-count census |

## 8. Gate

`python scripts/gate_g4_cpu.py` after the patch: **`baseline ... dG=-0.0030 width=1092`**,
ALL PASS — unchanged from before the patch. The feature vector was not touched: the holdout
filters ROWS in the training loader and adds no columns.
