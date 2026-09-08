# NEXT_ARMS — what the golden lane runs next, and why
### 2026-09-08 · analysis only · NOTHING was submitted by this document
### Basis for every number: 27 test proteins (2K5H excluded), ddG metric, per-protein fits, val-selected epoch.

---

## 0. THE SITUATION, MEASURED

`squeue` at the time of writing: **8 golden cards RUNNING** (~11h left) and **7 golden jobs ALREADY
PENDING** under `MaxGRESPerAccount`, submitted by `scripts/fill_golden2.sh`. The lane is **not** at
risk of idling — it is already oversubscribed by seven arms. **The decision in front of us is not
"what do we add", it is "is the queued wave the right wave".**

| state | jobs |
|---|---|
| RUNNING (wave 1) | `gld_slope1.0_s1/s2/s3`, `gld_loroWdesc2_s42`, `gld_w5_dg_s1/s2`, `gld_slope_w5_s42`, `gld_slope_anchor_s42` |
| **PENDING (wave 2)** | `gld_w12_s42/s1/s2`, `gld_w12_dg_s42`, `gld_w12_slope_s42`, `gld_w5_dg_s3`, `gld_slope_w5_s1` |

**Two findings below change what that wave should be.** Both are measurements made for this
document, not opinions.

---

## 1. THE POWER CALCULATION THAT ORDERS EVERYTHING

Five identical-config seeds (`abl_sigma_seed{1,2,3,4,42}`), 2K5H dropped, recomputed here:

| metric | mean | **SD across seeds** | CV |
|---|---|---|---|
| pooled ddG PCC | 0.5781 | **0.0344** | 6.0% |
| a_p (median) | 0.3789 | **0.0445** | 11.7% |
| std(b_p), dG | 1.1172 | **0.1408** | 12.6% |

**Correction to a natural assumption:** a_p and std(b_p) are *relatively noisier* than pooled PCC,
not quieter. The reason to score a lever on them is **the metric rule** — they measure the channel
the lever acts on — **not** better statistics. Anyone who justifies them as "less noisy" is wrong.

**Minimum detectable effect** (two-sided, alpha=0.05, 80% power, two arms of n seeds):

| metric | n=1 | n=2 | **n=3** | n=4 |
|---|---|---|---|---|
| pooled ddG PCC | 0.136 | 0.096 | **0.079** | 0.068 |
| a_p | 0.176 | 0.125 | **0.102** | 0.088 |
| std(b_p) | 0.558 | 0.394 | **0.322** | 0.279 |

**Read this against the effects we actually have:**

| known effect | size | detectable? |
|---|---|---|
| `--slope_weight 1.0` on **a_p** | **+0.242** | **YES at n=2** |
| `--slope_weight 1.0` on **pooled** | +0.0116 | **NO, not even at n=4** |
| slope arm's **std(b_p)** cost | +0.756 | **YES at n=1** |
| best pooled gain in the whole project | +0.075 | **borderline at n=3** |

> **The governing conclusion: pooled ddG PCC is under-powered for everything we can currently
> build.** Our best-ever gain (+0.075) sits at the n=3 detection limit. **An arm justified only by
> an expected pooled gain is an arm that cannot return a result.** Arms must be justified by a
> **channel-specific** prediction on a_p or std(b_p), where the effects are large relative to noise.
> This, not enthusiasm, is what ranks the candidates below.

---

## 2. FINDING A — THE nu SWEEP IS DEAD. IT MUST NOT GET A CARD.

Agent B's sweep completed (`results/08_data/nu_sweep.json`, frozen checkpoint, n=28, absolute WT dG).
P1's own success criterion was **std(b_p) below the 0.9967 baseline**. Recomputed here:

| arm | std(b_p) | vs base | std_pred |
|---|---|---|---|
| **base** | **0.9972** | — | 0.9125 |
| nu=0.500 b=5.82 | 1.1233 | **+0.1261** | 1.0467 |
| nu=0.550 b=5.82 | 1.1066 | +0.1095 | 1.0146 |
| nu=0.588 b=4.97 | 1.1043 | +0.1072 | 1.0102 |
| **nu=0.620 b=4.97 (best)** | **1.1036** | **+0.1064** | 1.0026 |
| nu=0.650 b=5.82 | 1.1416 | +0.1444 | 1.0424 |

**0 of 10 cells beat baseline. The best is 10.7% WORSE.** The physics argument for nu≈0.588 was
sound and the sweep was the right, cheap way to test it — and it returned a clean negative. The
sweep cost minutes on a frozen checkpoint, exactly as P1 specified, which is why this negative cost
us nothing.

**Verdict: P1 step 4 ("only if a nu beats baseline on dispersion, train one arm") is NOT triggered.
No nu arm is scheduled. The reference-state-via-coil programme now rests entirely on P6.**

This is a real result for the thesis: the coil's rejection in §5.1 was not an artifact of using the
theta exponent. **Giving the coil its physically correct shape does not rescue it.**

---

## 3. FINDING B — THE QUEUED W12 ARMS CANNOT ISOLATE W12

`scripts/fill_golden2.sh` submits:

    sub gld_w12_s42 --loss_mode ddg --sidechain_features --burial_features --seed 42

**W12 is bundled with W5 (`--burial_features`).** The matched control it would need is a
`--loss_mode ddg --burial_features` arm at the same seeds. **That arm does not exist.** Every W5 arm
ever scored or running is `--loss_mode dg`:

    scored:  gld_w5_dg_s42        (loss_mode dg)
    running: gld_w5_dg_s1, s2     (loss_mode dg, + --flory_unfolded --coil_b fixed)
    queued:  gld_w5_dg_s3         (loss_mode dg)

So `gld_w12_s42` minus the nearest control differs in **two** flags at once. Whatever it shows is
unattributable — **the same defect as factor A in P4**, which we already paid for once.

**Two of the five queued W12 arms are, however, clean and should stand:**

- `gld_w12_slope_s42` (ddg + sidechain + burial + slope) has an exact matched control **already
  running**: `gld_slope_w5_s42` (ddg + burial + slope). The contrast is **W12 alone**. OK
- `gld_w12_dg_s42` is a dG-channel arm; its control is `gld_w5_dg_s42` (scored, a_p 0.664), matched
  except for `--flory_unfolded --coil_b fixed` — **partially confounded**, worth fixing but usable.

**The fix costs nothing extra in wall-clock:** drop `--burial_features` from the three `gld_w12_s*`
arms, making them **W12-alone against the `sigma_seed*` control population we already have five
seeds of.** That is a clean, fully-powered contrast for free.

> **Recommendation R1 (highest value, zero GPU cost): before those pending jobs start, re-specify
> `gld_w12_s42/s1/s2` as W12-alone.** They are PENDING, not RUNNING — no safety rule is touched by
> replacing a queued job, and no running job is ever cancelled.

---

## 4. FINDING C — THE LORO RE-RUN HAS COLLAPSED AGAIN

`gld_loroWdesc2_s42`, the P2 re-run with `--aa_descriptors mordred_pca16_only`:

| epoch | ddG PCC | PCC-PP | **ddG RMSE** |
|---|---|---|---|
| 1 | 0.085 | 0.028 | **2.541** |
| 2 | 0.050 | 0.016 | **2.541** |

**RMSE frozen at exactly 2.541 to four significant figures — the identical constant the first LORO
arm froze at for ~11 epochs.** For contrast, healthy arms on the same lane at epoch 1:

    gld_slope1.0_s1  PCC 0.682   RMSE 1.504
    gld_w5_dg_s1     PCC 0.594   RMSE 1.808

**The `_only` fix did not work.** P2's diagnosis was scale competition between the descriptor block
and one-hot; `mordred_pca16_only` removes one-hot by zeroing it, so **if the collapse survives that,
the cause was never the scale clash.** A model whose RMSE is bit-stable across epochs while PCC
drifts toward zero is predicting a constant.

**Also noted: `scripts/gate_descriptor_scale.py` (P2 step 2) was never written.** It is the check
whose absence cost 8 GPU-hours the first time, and its absence has now cost a second card.

**Recommendation R2:** do **not** queue a third LORO training arm. Diagnose on CPU first — is the
descriptor table degenerate under `_only`, and does zeroing one-hot leave the mutated position with
no identity signal at all, which would make the collapse *correct behaviour* rather than a bug?
**Kill criterion, stated now:** if RMSE is still exactly 2.541 at epoch 5, the arm is dead and its
card is better released than run to 15. That is a judgement about a *running* job, for the human to
make — this document cancels nothing.

---

## 5. THE RANKING — expected information per GPU-hour

Information per GPU-hour = (probability the arm returns a *decisive* answer) x (value of that
answer) / cost. **An arm that cannot fail scores zero regardless of its cost.**

| # | arm | GPU-h | primary metric | why it ranks here |
|---|---|---|---|---|
| **1** | **W12-alone, 3 seeds** (R1 respec) | 24 | **a_p**, + std(b_p) | Largest measured deficit; mechanism verified at **r=-0.922**; effect predicted on a_p where MDE(n=3)=0.102; **wiring already verified end-to-end**. Highest value, already funded. |
| **2** | **W12 x slope** | 8 | **a_p**, + pooled | Matched control already running. Tests whether *information* and *rescaling* compose — the one question whose answer changes the architecture story either way. |
| **3** | **W12 on dG** | 8 | **std(b_p)**, dG MAE | Burial columns are zero unfolded, so folded-minus-unfolded IS the driving force; the only W12 arm that can move b_p. |
| **4** | **severing, RETRAINED (P6)** | 8 (+8 control) | **std(b_p) at matched PP-PCC** | Decides the whole reference-state programme, which nu has just left with no other support. |
| **5** | **more seeds on slope+W5** | 8–16 | **a_p** | Cheap; the current best combination arm is n=1. |
| **6** | clean factor-A design (P4) | **96** | std(b_p) | **Deferred — see §6. It cannot afford its own question.** |
| X | **nu trained arm** | — | — | **DEAD (§2). 0/10 cells beat baseline.** |
| X | third LORO arm | — | — | **Diagnose on CPU first (§4).** |

---

## 6. EACH ARM: METRIC, AND WHAT WOULD MAKE IT A FAILURE

### Arm 1 — W12 side-chain, alone, 3 seeds · 24 GPU-h

**Metric, per the metric rule.** W12's four columns split cleanly:

- cols 0,2 (identity: dG_transfer, volume) are **state-independent** — identical folded and
  unfolded — and change only at the mutated position. They **survive ddG**. → score on **a_p and
  per-protein PCC**, which are within-protein and legitimately ddG-measurable.
- cols 1,3 (burial-weighted) are **exactly zero in the unfolded pass** (verified: checks 3a/3b).
  They are a folded-state quantity acting on absolute stability. → **must also be scored on dG /
  std(b_p)**.

**Scoring it on ddG alone would repeat the W5 mistake** (MASTER §2.2 item 3), which is why both are
named.

**Primary: a_p (median), 3 seeds vs the 5 `sigma_seed*` controls.** Secondary: std(b_p), pooled.

**What makes it a FAILURE — declared before it runs:**

1. **a_p moves by less than 0.102** (the n=3 MDE) → no detectable effect; the deficit is not fixed
   by giving the model transfer free energy.
2. **a_p rises but the per-destination-residue slope gap does not close** — i.e. `corr(slope, KD)`
   stays near -0.73. This is the *specific* prediction: W12 should raise the hydrophobic-destination
   slopes (C 0.267, L 0.282, I 0.311) toward the polar ones (N 0.565), **not** lift all twenty
   equally. A uniform lift is a rescaling, and would mean W12 is acting as generic capacity, not as
   the claimed mechanism.
3. **a_p rises entirely through s with r flat.** By `a_p = r·s`, that is exactly what
   `--slope_weight` already does for free. **W12's whole claim is that it recovers LOST INFORMATION,
   so it must move r.** If r does not move, W12 is a redundant slope lever and must be reported as
   one.
4. std(b_p) degrades by more than the slope arm's known +0.756 cost → the trade is worse than the
   lever we already have.

**Criterion 3 is the one that matters.** It separates "another calibration knob" from "the model can
now see hydrophobicity", and it is measurable at n=3.

### Arm 2 — W12 x slope · 8 GPU-h

**Metric.** **a_p** primary (both levers act there, and it is ddG-measurable). Pooled secondary,
with the explicit caveat that pooled MDE at n=1 is 0.136, so **this arm cannot resolve a pooled
difference** — stated up front so a null is not misread as a refutation.

**Control:** `gld_slope_w5_s42`, already running, differs by `--sidechain_features` only.

**FAILURE:** a_p under the combination is **not above** the better of the two singles by more than
the n=1 a_p MDE (0.176) → the levers do not compose. Also a failure if a_p exceeds r — impossible
under `a_p = r·s` with s at its ceiling — **that would indicate a scoring bug, not a win**, and must
be investigated rather than reported.

### Arm 3 — W12 on the dG channel · 8 GPU-h

**Metric.** **std(b_p)** primary. This is the arm the metric rule *forces* onto dG: the
burial-weighted columns are a whole-folded-state quantity that partially cancels in ddG.

**FAILURE:** std(b_p) not below the `gld_w5_dg_s42` control by more than the n=1 MDE (0.558 — a very
weak test, so **this arm is exploratory and must be labelled so**), **or** the degenerate signature
appears: `std(pred WT dG)` collapsing toward `std(true WT dG) = 0.9239`, which is the
`--dg_length_norm` failure and voids the number. **Report std_pred alongside, always.**

### Arm 4 — severing, RETRAINED (P6) · 8 GPU-h + 8 for a matched control

    --unfolded_emb zero --flory_unfolded --coil_b fixed --coil_edges

**Metric.** **std(b_p), compared at MATCHED per-protein PCC.** Unmatched, the comparison is
confounded, because D1 (`--unfolded_emb zero`) alone destroys the model at 14.4 sigma (a_p 0.0136).

**FAILURE — and this arm has an unusually clean failure condition:** if std(b_p) does **not** shrink
in a model that *learned* with a protein-independent reference state, **the reference-state
explanation of b_p is closed for good** — a genuine, publishable negative that retires a whole
programme. If it *does* shrink but per-protein PCC collapses to D1 levels (~0.15), the result is
**uninterpretable in exactly the way the frozen version was** (`UNINTERPRETABLE_OOD`) and must be
reported as such, not as a win: **a_p ~ 0.014 means the model is flat, and a flat model has a small
offset spread trivially.**

**Preconditions:** print the OOD verdict line first, keep all four triggers active, and report
`corr(E_u, E_f)` for the trained model against the frozen 0.5430 [0.294, 0.736].

### Arm 5 — more seeds on the current arms · 8–16 GPU-h

**Which ones, and why only these.** Seeds buy MDE, and MDE only matters where the expected effect
sits near the threshold. From §1:

- `gld_slope_w5_s42` is n=1 and is the **best combination arm we have** → `gld_slope_w5_s1` (already
  queued) is correctly prioritised. **Keep it.**
- `gld_w5_dg_s3` completes 3 seeds of the best single-flag a_p (0.664). **Keep it.**
- **Do NOT add seeds to `--slope_weight 1.0`.** It is already n=4 (s42 scored + s1/s2/s3 running)
  and its a_p effect (+0.242) is detectable at n=2. A fifth seed lowers the a_p MDE from 0.102 to
  0.088 on an effect of 0.242 that is already unambiguous. **That card buys nothing.**

**FAILURE:** the three slope seeds disagree in *sign* on a_p, or their spread exceeds the +/-0.060
band on pooled — either would mean the "only proven lever" was a seed artifact, and everything
downstream of MASTER §3.2 would need re-examination.

### Arm 6 — clean factor-A design (P4) · 96 GPU-h · **DEFER**

**Is it worth it versus W12? Explicitly: no, not now.** Both sides of the argument:

**For A:** it is the one lever aimed squarely at **b_p**, which gates the **+0.14** the oracle says
is available — larger than every gain in the project combined. Factor A is currently *unestimable*
(perfectly aliased with seed and epoch), so this is not a re-run, it is the first real measurement.
And the exposure→a_p decay across anchor weights (0.587 → 0.471 → 0.400) is a real signal.

**Against A, and this is decisive:**

1. **P4's own primary metric is std(b_p), whose MDE at n=3 is 0.322 — 29% of its own mean.** The
   design as specified (4 levels x 3 seeds) **cannot detect anything smaller than a 29% change in
   offset dispersion.** P4 already concedes "only an effect above ~0.07 is detectable" on pooled;
   the std(b_p) picture is worse. **96 GPU-hours buys a test underpowered for its own primary
   question.**
2. **b_p has resisted EIGHT explanations, and the ninth clue points away from A.** The strongest
   remaining evidence (P5) is that the **top-2 |b_p| proteins carry 78% of the oracle gain and
   neither is a structural outlier**. b_p is concentrated in a few proteins for an unidentified
   reason — **not** the kind of thing a global anchor weight addresses. Spending 96 GPU-h on a
   global lever against a locally-concentrated phenomenon is the wrong instrument.
3. **P5's cheap CPU work has not been done yet.** Characterising the two outliers, and testing
   whether low per-protein-PCC proteins carry high |b_p| — i.e. whether **b_p is a symptom, not a
   cause** — costs **zero GPU-hours** and could invalidate the entire premise of the A design. If
   b_p is merely where the model's failures are absorbed, no b_p lever will work and A is moot.
4. **12 arms = 96 GPU-h = the whole golden lane for half a week**, during which nothing else runs.

**Verdict: run P5's CPU characterisation FIRST. It is free, and it is the thing most likely to tell
us whether the 96 GPU-hours are worth spending at all.** Revisit A only if P5 shows b_p is a cause
rather than a symptom, and then only with a power calculation attached.

**W12 wins the comparison on all three terms:** larger measured effect (a_p, MDE 0.102, mechanism at
r=-0.922) / smaller cost (24 vs 96 GPU-h) x higher probability of a decisive answer (a_p is
ddG-measurable and levers have moved it before; **std(b_p) has never responded to anything**).

---

## 7. EXACT SBATCH LINES — ready to paste

Every line carries `--partition=rtx6000 --qos keasar` (both required; `--qos keasar` alone is
rejected and the `rtx6000` partition has `DenyQos=normal`), `--gres=gpu:rtx_6000:1` (never select a
GPU by partition name — a lua plugin rewrites partition-selected GPU requests onto `gpu`),
`WANDB_MODE=disabled` (one omission destroyed a 7h53m run), `--resume`, and a unique `run_tag`.

```bash
cd /home/nissimb/DeepPEF
SUB="module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled;"
BASE="--full_data --no_pretrain --no_freeze --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --resume"
G="--partition=rtx6000 --qos keasar --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 12:00:00"

sub () {
  local N="$1"; shift
  squeue -u nissimb -h -o "%j" | grep -qx "$N" && { echo "skip $N (already queued)"; return; }
  ls eval_results/abl_${N#gld_}_e*.csv >/dev/null 2>&1 && { echo "skip $N (already scored)"; return; }
  sbatch --parsable --job-name "$N" $G --output logs/${N}_%j.out --error logs/${N}_%j.out \
    --wrap "$SUB python Megascale-fineTuning/train.py $BASE $* --run_tag ${N#gld_}"
}
```

**R1 — W12 ALONE, 3 seeds (replaces the confounded `gld_w12_s42/s1/s2`).**
Control = the five `sigma_seed*` runs. Differs from control by `--sidechain_features` ONLY.

```bash
sub gld_w12only_s42 --loss_mode ddg --sidechain_features --seed 42
sub gld_w12only_s1  --loss_mode ddg --sidechain_features --seed 1
sub gld_w12only_s2  --loss_mode ddg --sidechain_features --seed 2
```

**Already queued and correctly specified — leave them alone:**

```bash
# gld_w12_slope_s42 : control gld_slope_w5_s42 (running), differs by --sidechain_features only
# gld_w12_dg_s42    : dG channel, std(b_p) primary
# gld_w5_dg_s3      : completes 3 seeds of the best single-flag a_p
# gld_slope_w5_s1   : second seed of the best combination arm
```

**R3 — severing retrained (P6), when a card frees.** Two arms — the severed model and its matched
D0 control — because the comparison must be made at matched per-protein PCC.

```bash
sub gld_sever_s42     --loss_mode dg --unfolded_emb zero --flory_unfolded --coil_b fixed --coil_edges --seed 42
sub gld_severctrl_s42 --loss_mode dg --flory_unfolded --coil_b fixed --coil_edges --seed 42
```

**NOT submitted, deliberately:**

```bash
# nu trained arm          -- DEAD: 0/10 sweep cells beat baseline on std(b_p) (see section 2)
# third LORO arm          -- diagnose on CPU first; RMSE frozen at 2.541 again (section 4)
# 12-cell factor-A design -- underpowered for its own metric; run P5 free CPU work first (section 6)
```

**Before any of these start, two CPU jobs that cost no GPU time:**

1. `scripts/gate_descriptor_scale.py` — P2 step 2, **never written**, and its absence has now cost
   two cards.
2. Extend `scripts/gate_g4_cpu.py` to assert **width=1096 with `--sidechain_features` on** (P3 step
   6). The gate currently covers 1092 and 1095 only. W12's width is verified by
   `_w12_bak/w12_verify.py` (check 4g) but is **not** in the standing guard.

---

## 8. WHAT WAS VERIFIED FOR THIS DOCUMENT, WITH ARTIFACTS

Per the signature failure mode — *code runs, prints a success line, feature never read* — every
claim here was checked against an artifact, not a DONE message.

| claim | how it was verified |
|---|---|
| **G4 guard intact** | `python scripts/gate_g4_cpu.py` → **`baseline PASS dG=-0.0030 width=1092`**, ALL PASS. Unmoved. |
| W12 is genuinely wired | `_w12_bak/w12_verify.py` ALL PASS. Decisive check **4h: perturbing ONLY cols 48:52 moves the output** (max abs delta = 2.28e-3) — the block is *read*, not merely present. |
| W12 sizing is correct | 4a/4b `fc1_gat` 36→40, `fc1_gcn` 52→56; 4c/4d/4e `fc2_*`, `fc_in_dim`, `inst_norm*` **unchanged**; 4g forward finite at **width=1096**. |
| W12 unfolded zeroing | 3a/3b unfolded and flory-coil burial cols **exactly zero**; 3e folded-minus-unfolded **is** the burial-weighted term. |
| W12 mechanism | `gate_sidechain.py` **ALL PASS**: `corr(dG_transfer, slope) = -0.922` vs volume -0.408. |
| PEM guard present | 6a `graph_transformer` **raises** on `--sidechain_features` (it slices left-anchored and would otherwise silently ignore the block — hole #1 in MASTER §6.5). |
| nu sweep is negative | `results/08_data/nu_sweep.json`, 10 cells, std(b_p) recomputed; **0 beat 0.9972**. |
| LORO collapsed again | `logs/gld_loroWdesc2_s42_21136228.out`: RMSE **2.541** at both scored epochs vs 1.504/1.808 for healthy arms. |
| W12/W5 confound | `scripts/fill_golden2.sh` line for `gld_w12_s42` includes `--burial_features`; **no `--loss_mode ddg --burial_features` arm exists** in `eval_results/` or `squeue`. |
| seed dispersion | recomputed directly from the 5 `abl_sigma_seed*` CSVs, 2K5H dropped. |

**Nothing was submitted. Nothing was cancelled. No running job was touched.**

---

## 9. THE ONE-PARAGRAPH ANSWER

**The golden lane is already full for the next ~11 hours and has seven arms queued behind it, so the
decision is corrective, not additive.** Three of the seven queued arms (`gld_w12_s42/s1/s2`) bundle
W12 with W5 and have no matched control, so they cannot attribute their result to W12 — respecify
them as W12-alone against the five-seed `sigma` population, which costs nothing and buys a clean,
fully-powered contrast. **The nu arm is dead on its own pre-declared criterion** (0 of 10 cells beat
baseline on std(b_p)), and the coil's rejection therefore does not depend on having used the wrong
exponent. **The LORO re-run has collapsed a second time**, so the next LORO step is CPU diagnosis,
not a third card. **The 96-GPU-hour factor-A design should wait**, because it is underpowered for
its own primary metric (std(b_p) MDE at n=3 is 0.322, 29% of its mean) and because the free CPU work
in P5 — is b_p a cause or a symptom? — could invalidate its premise entirely. **W12 wins on every
term of information-per-GPU-hour: a verified mechanism (r=-0.922), a channel where effects are
detectable at n=3, and a quarter of A's cost.** Its failure condition is declared in advance and is
not the vague "no improvement" but something sharper: **if a_p rises through s with r flat, W12 is a
redundant slope lever and must be reported as one.**
