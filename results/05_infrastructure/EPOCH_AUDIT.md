# EPOCH-SELECTION AUDIT (OPEN_PROBLEMS P0)
### 2026-09-08 · scripts/gate_epoch_selection.py · results/trajectory_runs.txt

**The question this answers.** Is our headline `0.6382` the epoch chosen on VALIDATION, or the best
of 15 epochs on TEST? If the latter, we did exactly what we criticise in the prior work.

---

## 0. VERDICT

| claim | status |
|---|---|
| The headline `sigma_seed2_e10` = **0.6382** is val-selected, scored once | ✅ **CLEAN** |
| The control `calib_ctrl_repro2_e14` = **0.5635** is val-selected, scored once | ✅ **CLEAN** |
| Therefore the **+0.075 headline gain is not a peeking artifact** | ✅ **CLEAN** |
| **30 of 32** single-epoch runs scored exactly at their val argmax | ✅ **CLEAN** |
| `gld_slope1.0_s42` is quoted at a **non-canonical** epoch everywhere it appears | ❌ **DEFECT 1** |
| `OPEN_PROBLEMS.md` P0 asserts canonical `e14` for both trajectory runs | ❌ **DEFECT 2 (wrong)** |
| The `a_p = 0.740` headline mixes two different scoring bases | ❌ **DEFECT 3** |
| 2 runs scored at an epoch that is not their val argmax | ⚠️ **DEFECT 4 (minor)** |
| `validation/score_runs.py` prints "BEST PCC" over scored TEST epochs | ⚠️ **DEFECT 5 (hazard)** |
| Canonical `e10` now scored: **a_p 0.721**, vs the quoted 0.740 (**+0.019 inflation**) | ✅ **MEASURED** |
| The gate now **PASSES**; G4 unmoved at dG=-0.0030 width=1092 | ✅ **GREEN** |
| 7 inherited `$SHAHAR` baselines are quoted; 5 have 4-15 scored epochs | ⚠️ **BLIND SPOT (§6c)** |
| …but **none** of those quotes is its run's test maximum | ✅ **CLEAN** |

**Nothing in the audit moves the headline. The defects are all confined to the two trajectory runs
and to two near-zero-scoring arms. But DEFECT 1 means the single most-quoted secondary number in the
thesis — `a_p 0.740`, "the only proven lever" — is a best-of-six TEST pick.**

**One-line answer to the question P0 asks: we did NOT peek on the headline, and we DID peek on
`a_p`.** The peek was accidental (a max-over-scored-epochs line in a helper script, §7), it is
confined to one arm, and that arm's qualitative claim survives — but the number must be requoted.

---

## 1. HOW SELECTION ACTUALLY WORKS (verified in code, not assumed)

`Megascale-fineTuning/run_calib_eval.sh` lines 14-28:

    # 1) best epoch by VAL overall ddG-PCC (validate() prints one line per epoch, in epoch order)
    m = re.findall(r'ddG PCC=(-?\d+\.\d+)\s+ddG PCC-PP=(-?\d+\.\d+)\s+ddG RMSE=(-?\d+\.\d+)', t)
    best = max(range(len(m)), key=lambda i: float(m[i][0]))

The log it parses is the **training** log, whose `ddG PCC=` lines `validate()` emits from
`self.val_ds`. The test set is then scored **once**, at that epoch (lines 45-48). **We did not peek.**

Two properties of this parser matter downstream and are reproduced exactly by the gate:
- the log prints **3 decimals**, so ties are real and common;
- `max(range(n), key=...)` returns the **first** maximum, so ties break to the **earlier** epoch.

---

## 2. THE SCORED-EPOCH CENSUS

34 run tags across 44 `eval_results/abl_*.csv` (one factorial cell landed mid-audit).
**32 tags have exactly one scored test epoch.**
Two have six each — and only two:

    gld_slope1.0_s42   e0, e4, e8, e12, e13, e14
    gld_dg_coil_s42    e0, e4, e8, e12, e13, e14

These were scored deliberately by `scripts/score_arms.sh` to measure a **trajectory** (does `r` stay
flat while `s` rises). That is a legitimate use of multiple epochs. Quoting the best of six is not.

---

## 3. DEFECT 1 — `gld_slope1.0_s42` IS QUOTED AT A NON-CANONICAL EPOCH (the real finding)

**The canonical epoch was never checked. It is not e14, and it is not e13.**

Val ddG PCC from `logs/gld_slope_21080862.out` (SLURM 21080862), all 15 epochs:

| e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | **e10** | e11 | **e12** | e13 | e14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| .623 | .674 | .660 | .703 | .713 | .724 | .717 | .719 | .723 | .737 | **.740** | .710 | **.740** | .730 | .722 |

**Val argmax = 0.740, attained at e10 and e12.** The tie is genuine at the log's 3 decimals, and
`run_calib_eval.sh`'s first-max rule resolves it to **e10**.

    CANONICAL EPOCH FOR gld_slope1.0_s42  =  e10
    e14 has val PCC 0.722 — the 8th-best of 15 epochs.
    e13 has val PCC 0.730 — the 6th-best of 15 epochs.

**e10 was not among the six scored epochs.** The trajectory was scored at e0/4/8/12/13/14, so the
canonical checkpoint had **no test score at all** until this audit scored it (SLURM 21137818, CPU,
COMPLETED 00:59:29, 28315 rows). e12 — the other half of the val tie — was already scored.

### THE CANONICAL NUMBER, now measured

    gld_slope1.0_s42 @ e10  (27-protein basis, the basis of record)
        pooled ddG PCC  0.5798
        a_p (median)    0.7210      <-- THIS is what tables must quote, not 0.740
        r   (median)    0.7878
        s   (median)    0.9760
        ddG PCC-PP      0.7250

**The honest canonical value is a_p = 0.721, not 0.740.** The quoted figure was inflated by
**+0.019** — small in absolute terms, and it does not change any conclusion, but it was obtained by
taking the maximum over six test epochs rather than the one validation chose.

### What the docs actually quote

| file:line | quotes | that epoch's val rank | verdict |
|---|---|---|---|
| `MASTER.md:59` | `gld_slope1.0_s42_e13` — a_p **0.740**, s **1.042** | **6th of 15** | ❌ best-of-six TEST pick |
| `MASTER.md:141` | "`gld_slope1.0_s42`, epoch 14" — a_p 0.7396 | **8th of 15** | ❌ non-canonical |
| `MASTER.md:685`, `STATUS_FULL.md:23,40,97` | "**best a_p = 0.740**", "the only proven lever" | inherits e13 | ❌ inherits the pick |
| `CONTEXT.md:2696` | "`gld_slope1.0_s42` at epoch 14" — a_p 0.7396 | **8th of 15** | ❌ non-canonical |

**The measured test a_p across the six scored epochs (27-protein basis, per-protein median):**

    e0  0.4796      e12  0.6945
    e4  0.5748      e13  0.7397   <-- the quoted headline: the MAXIMUM of the six
    e8  0.7039      e14  0.7117
                   [e10  0.7210]  <-- THE CANONICAL EPOCH, scored by this audit

**`a_p = 0.740` is the argmax over six scored TEST epochs.** That is the definition of test-set
peeking. The direction of the error favours our own claim, which is the worst direction.

---

## 4. DEFECT 2 — `OPEN_PROBLEMS.md` P0 STATES THE WRONG CANONICAL EPOCH

P0 step 3 reads: *"For both runs that is `e14` (the val log's argmax)"*. **That was assumed, not
parsed.** It is correct for `gld_dg_coil_s42` and **wrong for `gld_slope1.0_s42`** (argmax e10,
val PCC 0.740 vs e14's 0.722). Had the allowlist been seeded with the asserted value, the gate would
have blessed a non-canonical epoch and closed P0 while the defect stood.

For the record, `gld_dg_coil_s42` from `logs/gld_dg_coil_21080861.out` (SLURM 21080861):

| e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | e11 | e12 | e13 | **e14** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| .541 | .579 | .569 | .603 | .607 | .607 | .621 | .648 | .700 | .702 | .737 | .743 | .758 | .747 | **.769** |

**Canonical = e14, uniquely.** `MASTER.md:60` and `MASTER.md:304` quote e14. ✅ **Correct.**
This arm is a rejection (pooled 0.2212 vs control 0.5635), so its epoch does not flatter it either.

---

## 5. DEFECT 3 — THE `a_p = 0.740` NUMBER MIXES TWO SCORING BASES

The scoring basis of record is **27 proteins (2K5H excluded)**. Recomputing every candidate:

| quoted number | epoch | basis | a_p (median) | s (median) | r (median) |
|---|---|---|---|---|---|
| `MASTER.md:59` a_p **0.740**, s **1.042** | e13 | **27-protein** | 0.7397 | 1.0422 | 0.7904 |
| `MASTER.md:141` / `CONTEXT.md:2696` a_p **0.7396**, s **0.9901**, r **0.7925** | e14 | **28-protein (2K5H INCLUDED)** | 0.7396 | 0.9901 | 0.7925 |

The two `0.740`s in the docs are **not the same measurement**. They agree to 4 decimals by
coincidence — one is e13 on 27 proteins, the other e14 on 28. `CONTEXT.md:2696` and `MASTER.md:141`
therefore violate the 27-protein basis as well as quoting a non-canonical epoch.

---

## 6. DEFECT 4 — TWO RUNS SCORED OFF THEIR VAL ARGMAX (minor)

The gate counts scored epochs; it does not check that a *single* scored epoch is the right one.
Auditing all 31 single-epoch runs against their training logs:

**30 of 32 MATCH their val argmax exactly.** Two do not:

| run | scored | val argmax | val PCC at each | pooled ddG PCC | consequence |
|---|---|---|---|---|---|
| `loroW_onehot_s42` | e12 | **e14** | .712 vs .726 | — | LORO arm; see P2 |
| `p3_a1_d1_s0_D1_uemb_seed42` | e12 | **e14** | .689 vs .711 | **-0.0741** | factorial D1 cell |

In both cases the `kf_all_epoch_14.pt` checkpoint **already existed** when the e12 CSV was written
(loroW: ckpt 02:57 vs CSV 04:15; p3 cell: ckpt 19:24 vs CSV 19:53), so this is genuine
mis-selection, not "best checkpoint available at scoring time".

**Neither affects a headline.** Both are D1/LORO arms scoring at or below zero, and both were scored
*early*, i.e. the error runs **against** the arm, not for it. `p3_a1_d1_s0_D1_uemb_seed42` appears in
`MASTER.md:61` only as a ✗ rejection at -0.0741; scoring it at its (better) val-selected e14 would
weaken that rejection slightly, not overturn it. Flagged for correctness, not urgency.

This check is now a script, `scripts/audit_epoch_provenance.py`, so it can be re-run as new evals
land. It reproduces `run_calib_eval.sh`'s selection exactly (same regex, same first-max tie-break).
Final coverage over all 34 run tags:

    MATCH      30
    MISMATCH    2     loroW_onehot_s42, p3_a1_d1_s0_D1_uemb_seed42
    NO_LOG      0
    AMBIGUOUS   0
    SKIP        2     the two allowlisted trajectory runs

**A warning for whoever re-runs this.** Matching a run tag to its training log by substring
produces FALSE mismatches, and it produced one for me before I tightened it:

- `sigma_seed4` is a substring of `sigma_seed42`, so it picked up seed42's log and *looked*
  mis-selected. Against its own log `logs/sigma_seed4_21016225.out` the val argmax is e14 and the
  scored CSV is e14. ✅ **Correct, not a defect.**
- The log naming is also inconsistent in two ways the script now handles explicitly: several `gld_*`
  training logs drop the seed suffix (`gld_w5_dg_21084598.out` for tag `gld_w5_dg_s42`), and the
  LORO log carries an extra prefix (`gld_loroW_onehot_...` for tag `loroW_onehot_s42`). Matching on
  whole jobid-stripped stems plus an explicit variant list takes NO_LOG from 5 to 0 without
  reintroducing the substring trap.

---

## 6b. WHAT THE AUDIT CLEARS

Three load-bearing tables were checked and are **not** affected.

**(a) The scoreboard values themselves reproduce exactly** (27-protein basis, pooled ddG PCC):

| row | claimed | measured |
|---|---|---|
| `sigma_seed2_e10` (headline) | 0.6382 | **0.6382** |
| `calib_ctrl_repro2_e14` (control) | 0.5635 | **0.5635** |
| `gld_slope1.0_s42_e13` | 0.5810 | **0.5810** |
| `gld_dg_coil_s42_e14` | 0.2212 | **0.2212** |

The numbers are right. Only the *epoch choice* for `gld_slope1.0_s42` is wrong.

**(b) The `--slope_weight` sweep table (`MASTER.md:153`) is clean.** All four cells are `p3_*` runs
that are single-epoch and val-selected:

| weight | run | s | a_p | r | doc claims |
|---|---|---|---|---|---|
| control | `calib_ctrl_repro2_e14` | 0.635 | 0.496 | 0.793 | 0.635/0.499/0.798 |
| 0.3 | `p3_slope0.3_s42_e9` | 0.573 | 0.398 | 0.798 | 0.573/0.415/0.801 |
| **1.0** | `p3_slope1.0_s42_e13` | 0.756 | 0.507 | 0.803 | 0.756/0.507/0.803 |
| 3.0 | `p3_slope3.0_s42_e8` | 0.472 | 0.326 | 0.793 | 0.472/0.326/0.795 |

Small drifts (a_p 0.398 vs 0.415 at w=0.3) are basis/rounding, not epoch selection.
**The conclusion "weight 1.0 best, weight 3.0 overshoots" survives the audit intact** — it never
depended on the trajectory run.

**(c) Scope.** The only scored-epoch artifacts are `eval_results/abl_*.csv`. The 36 entries under
`Megascale-fineTuning/models/eval_results/` are **wandb run directories, not CSVs**, and the
`results/08_data/seqabl/abl_*.csv` files carry no `_e<N>` suffix. The gate's glob is complete.

---

## 6c. THE INHERITED (shaharax) BASELINES — a gate blind spot, but the quotes are clean

Seven epoch-references in `results/01_start_here/` have **no backing CSV in our `eval_results/`**:

    calib_combo_e13  calib_ctrl_e12  ddg_core_e9  ddg_nopretrain_e11
    designed_w3_e8   randscratch_e12  readout_attn_e12

They are not fabricated. All seven live in the **read-only** prior-work tree
`$SHAHAR = /groups/keasar_group/casp15/Shahar/DeepPEF/eval_results` (79 CSVs, 14 run tags), and
`CONTEXT.md:52-59` says so explicitly. They are the inherited baselines, on the 28-protein basis.

**But five of them have MANY scored test epochs** — the exact condition the gate exists to catch:

| run | scored epochs | our docs quote | pooled at quoted | max over all scored | is the quote the max? |
|---|---|---|---|---|---|
| `readout_attn` | **15** (e0-e14) | e12 | 0.6055 | e5 = 0.6496 | **no** |
| `randscratch` | **14** (e0-e13) | e12 | 0.6237 | e2 = 0.6393 | **no** |
| `ddg_nopretrain` | **9** | e11 | 0.6471 | e2 = 0.6546 | **no** |
| `ddg_core` | 5 | e9 | 0.5992 | e5 = 0.6044 | **no** |
| `calib_combo` | 4 | e13 | 0.6313 | e2 = 0.6459 | **no** |
| `calib_ctrl`, `designed_w3` | 1 each | e12, e8 | — | — | n/a |

**Every quoted epoch is BELOW its run's best scored epoch.** We are not cherry-picking the prior
work's numbers — if anything we quote them conservatively. The quotes are clean.

**The blind spot is real, though.** `scripts/gate_epoch_selection.py` globs only our own
`eval_results/abl_*.csv`, so a future number pulled from `$SHAHAR` at a hand-picked epoch would
pass the gate silently. Two mitigations, neither urgent:

1. The tree is read-only and frozen, so no *new* multi-epoch runs can appear there.
2. Anyone quoting a `$SHAHAR` number should state the epoch and confirm it is not that run's test
   argmax. The table above does this once for all seven currently-quoted references.

Also note these are **28-protein** numbers, so they must not be compared directly against our
27-protein scoreboard — the same basis hazard as DEFECT 3.

---

## 7. DEFECT 5 — `score_runs.py` PRINTS A PEEKING AFFORDANCE (hazard)

`validation/score_runs.py:64` prints, for every run tag:

    BEST PCC ep{bp[0]}={bp[1]:.3f} | BEST PCC-PP ep{bpp[0]}={bpp[4]:.3f}

That is an argmax over **scored TEST epochs**, printed next to the per-epoch test curve. For the 31
single-epoch runs it is harmless (the max of one). For the two trajectory runs it prints exactly the
number a person must not quote — and `MASTER.md:59`'s `e13` **is** what that line emits for
`gld_slope1.0_s42`. This is the most probable proximate cause of DEFECT 1. Recommend relabelling it
`BEST-OF-SCORED (NOT a selection criterion — see results/trajectory_runs.txt)`.

---

## 8. THE GATE

`scripts/gate_epoch_selection.py`. Groups `eval_results/abl_*.csv` by run tag via
`^abl_(.+)_e(\d+)\.csv$` and FAILS on any of:

1. a tag with >1 scored epoch that is not in `results/trajectory_runs.txt`;
2. an allowlisted tag whose **canonical epoch has no scored CSV**;
3. a stale allowlist entry (tag has no CSVs);
4. an allowlist entry for a tag that has only one scored epoch (exemption no longer needed).

The allowlist format binds the canonical epoch into the record:

    <run_tag> | canonical=e<N> | <one-line reason>

**It was verified against ten synthetic cases, including the one that matters** — a run with two
scored epochs and no allowlist entry, which must FAIL:

    PASS  | clean: all single-epoch                 -> rc=0
    PASS  | SNEAK 2 epochs unallowlisted            -> rc=1
    PASS  | allowlisted+canonical scored            -> rc=0
    PASS  | allowlisted+canonical NOT scored        -> rc=1
    PASS  | stale allowlist entry                   -> rc=1
    PASS  | unneeded allowlist (1 epoch)            -> rc=1
    PASS  | 3 epochs unallowlisted                  -> rc=1
    PASS  | malformed allowlist line                -> rc=ValueError
    PASS  | dotted tag gld_slope1.0_s42             -> rc=0
    PASS  | dup epoch same tag is ONE epoch         -> rc=0

Wired into `scripts/loop_check.sh` as section 5, so it runs every round.

**First run against the real tree FAILED, on exactly the real defect:**

    GATE epoch_selection: 33 run tags, 31 single-epoch, 2 multi-epoch, 2 allowlisted
      allowlisted gld_dg_coil_s42: scored [e0,e4,e8,e12,e13,e14], CANONICAL=e14
    GATE epoch_selection: FAIL
      - CANONICAL EPOCH NOT SCORED: gld_slope1.0_s42 canonical=e10 (val-selected) but the
        scored epochs are [e0,e4,e8,e12,e13,e14].

`scripts/gate_g4_cpu.py` re-run after every change above: **baseline dG=-0.0030 width=1092, ALL PASS.**
Unmoved.

---

## 8b. A NOTE ON THE P8 GATE (not caused by this work)

`scripts/gate_headline.py` was **PASSING** at the start of this audit and is **FAILING** at the end:

    FAIL - 18 unmarked retired figure(s) in results/01_start_here

**This is not from the epoch audit.** The failing figures are all in
`results/01_start_here/AGENT_HARVEST.md`, a 107KB file written at **21:26** by another agent while
this audit was running; it quotes the retired 22-CSV headline `0.4899/0.6443/+0.1544` and the
20-checkpoint `0.5646/0.6347/+0.0702` without marking them retired.

The canonical basis itself is untouched and still reads exactly:

    n=9  pooled 0.5772 +/- 0.0316   oracle 0.7156 +/- 0.0474   gain +0.1384
    best single run: pooled 0.6382 (abl_sigma_seed2_e10.csv)

The nine-CSV membership in `gate_headline.py` is **hardcoded**, not globbed, so the
`abl_gld_slope1.0_s42_e10.csv` this audit added cannot have entered that population. Someone should
mark the AGENT_HARVEST figures as retired, but it is a separate task from P0.

---

## 9. WHAT MUST CHANGE IN THE DOCS

**The replacement row is measured and ready to paste** (27-protein basis):

| rank | run | pooled | a_p | r | s |
|---|---|---|---|---|---|
| … | `gld_slope1.0_s42_e10` | 0.5798 | **0.721** | 0.788 | 0.976 |

1. **`MASTER.md:59`** — `gld_slope1.0_s42_e13` (pooled 0.5810, a_p 0.740, s 1.042) is a best-of-six
   test pick. Replace with the `_e10` row above, footnoting that e0/4/8/12/13/14 exist only as a
   trajectory (`results/trajectory_runs.txt`).
2. **`MASTER.md:141`, `CONTEXT.md:2696`** — the "epoch 14" numbers (a_p 0.7396 / s 0.9901 / r 0.7925)
   are both non-canonical **and** on the 28-protein basis. Requote at e10 on 27 proteins.
3. **`MASTER.md:685`, `STATUS_FULL.md:23,40,97`** — "best a_p = 0.740 / the only proven lever"
   inherits the e13 pick. Restate at the canonical value.
4. **`OPEN_PROBLEMS.md` P0 step 3** — correct "for both runs that is e14" to
   `gld_slope1.0_s42 -> e10`, `gld_dg_coil_s42 -> e14`.
5. **`validation/score_runs.py:64`** — relabel the `BEST PCC` line (DEFECT 5).
6. **`loroW_onehot_s42` and `p3_a1_d1_s0_D1_uemb_seed42`** — rescore at e14, or footnote the
   deviation. Low priority: neither carries a claim.

**Why the mechanism claim survives the requote.** The identity `a_p = r·s` was re-verified at
**every** scored epoch of the slope arm on the 27-protein basis — `max|a_OLS − r·s|` is
1.1e-15, 6.7e-16, 6.7e-16, 1.1e-15, 8.9e-16, 8.9e-16 at e0/4/8/12/13/14, matching the documented
2e-15. And `r` is pinned at 0.7887–0.7904 across e4–e14 while `s` climbs 0.805 → 1.042. So
"the gain comes through `s`, not `r`" is true at whichever epoch you quote. **Only the magnitude
of a_p moves, not the mechanism.**

**The slope lever's QUALITATIVE claim is not at risk.** `r` is flat at 0.789-0.790 across e8-e14
while `s` climbs, so `a_p = r·s` moving via `s` holds at every scored epoch including the canonical
one. What changes is the *magnitude* quoted for a_p, and the honesty of how it was chosen. The arm
is also still **n=1**, and its pooled gain (+0.0116) is far inside the ±0.060 seed-noise band.
