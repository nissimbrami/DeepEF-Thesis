# INTEGRITY AUDIT — the 28 test proteins, their labels, their tensors, and the split
### DeepEF, M.Sc. thesis, Nissim Brami. Audit run 2026-09-08.
### Gate status after this work: `gate_g4_cpu.py` prints `baseline ... dG=-0.0030 width=1092` — UNCHANGED.

**Mandate.** The 2K5H defect (§12.1 of FINDINGS) inflated the headline gain by 35% and manufactured
a phantom "26,315 double mutants" from one malformed file. This audit assumed there were others.

**Headline: the data are sound.** The labels are bit-identical to the benchmark source, the tensors
align perfectly with the labels on all 28 proteins, and the train/test split has no leakage. Two new
defects were found; **neither affects any number currently in FINDINGS.md**, and one is a live trap
that would have corrupted the next person to touch it. The scale of the search is stated below so
that "we found nothing" is a measurement rather than an absence of effort.

---

## 0. Verdict table

| # | Check | Scope | Result |
|---|---|---|---|
| 1 | `gate_refrow.py` over every eval CSV | 24 CSVs | 24/24 FAIL — **all on 2K5H alone**, the known defect |
| 2 | ddG internally consistent with dG − dG(row 0) | 28 proteins, 28,314 rows | **0 inconsistent rows** |
| 2 | Labels vs the independent ThermoMPNN source | 28 proteins, 28,312 rows | **max abs diff exactly 0.0** |
| 2 | Duplicate mutation names | 28 proteins | 48 raw → **2 survive into eval**, both 2K5H |
| 2 | NaN / non-finite labels | 28 proteins | **none** |
| 2 | Impossible dG | 28 proteins | **none** — source is clipped to [−1, 5] |
| 3 | coords/mask/one_hot/emb/dG shapes mutually agree | 28 proteins | **28/28 agree** |
| 3 | Tensor rows == mutation-file rows (post-filter) | 28 proteins | **28/28 exact, delta 0** |
| 3 | dG tensor == CSV dG column, elementwise | 28 proteins | **max 3.9e-07** (float32 round-trip) |
| 3 | **one_hot encodes the mutation `mut_type` names** | 29,517 rows | **29,517 OK, 0 anomalies** (excl. 2K5H) |
| 3 | Non-finite / all-zero embedding rows | 28 proteins | **0** |
| 4 | Train/test name overlap | 239 train × 28 test | **NONE** |
| 4 | Exact WT sequence match train↔test | 239 × 28 | **NONE** |
| 4 | Test *variant* sequence equals a training WT | 28,314 × 239 | **NONE** |
| 4 | Max sequence identity to any training protein | 28 test | **0.349 max, 0.159 median** |
| NEW | `data_fixed/2K5H.csv` vs its tensors | positional | **DESYNCHRONISED — 2,367 of 3,549 rows** |
| NEW | 2K5H-style multi-background files | all 368 dirs | **36 files**; 1 test, 4 train, 31 unused |

---

## 1. `gate_refrow.py` — what it finds now

**24 of 24 eval CSVs FAIL, and every failure is 2K5H.** No other protein trips either the
background-multiplicity test or the percentile test. Each CSV reports two hard findings:

```
[HARD] 2K5H  has 2 rows with ddG==0 (expected exactly 1)
[HARD] 2K5H  row0 dG=1.723 at the 14.6-th percentile AND the mutation file has 3 distinct
             backgrounds - row 0 is probably a MUTANT background, not the true WT
```

The first line is worth recording because FINDINGS does not mention it: because the true WT
(dG 4.805) and the G11S background (dG 1.723) both appear in the evaluated subset, and ddG is
computed relative to row 0, **two different rows carry ddG == 0**. That is an independent
signature of the same defect, visible without any percentile reasoning.

**These eval CSVs were produced before the fix and read the original tree**, so this is the
expected state, not new damage. It does mean the corrected 2K5H has **not** propagated into any
scored artifact: every number derived from `eval_results/abl_*.csv` still carries the defect, which
is exactly why the corrected headline (§1.3) was computed by dropping/correcting 2K5H downstream
rather than by re-running evaluation.

---

## 2. Label audit — per protein

Two independent checks. Neither found a defect outside 2K5H.

**(a) Internal consistency.** In the eval CSVs, `ddG` is generated as `deltaG − deltaG.iloc[0]`
(`evaluate.py:431`), so consistency is true by construction; the check confirms no post-hoc edit has
broken it. **0 inconsistent rows in all 28 proteins.**

**(b) Against an independent source — the decisive test.** `data/ThermoMPNN/mega_test.csv` is the
published benchmark and was produced by a different pipeline. Joining on `name`:

> **All 28 proteins, 28,312 joined rows: maximum absolute difference exactly 0.000e+00.**

The labels are not merely self-consistent, they are byte-identical to the benchmark. This retires
label corruption as a hypothesis for the calibration failure.

*A trap avoided.* Joining on `deltaG_t` instead gives max differences of ~15 kcal/mol and correlations
collapsing to 0.61–0.79 — which looks exactly like a catastrophic label defect. It is not:
`deltaG_t` is the **trypsin** fit and uses **−15.0 and +15.0 as sentinels** for unmeasurable values
(120 and 69 occurrences). The column to use is `deltaG`. Reporting the `deltaG_t` comparison would
have manufactured a spectacular false defect in 17 of 28 proteins.

**Per-protein results** (raw rows / rows after the ins-del filter / rows reaching eval):

| protein | raw | filt | eval | dup names | dup surviving | NaN | impossible dG | backgrounds |
|---|---|---|---|---|---|---|---|---|
| 1GYZ | 1703 | 1526 | 1126 | 0 | 0 | 0 | 0 | 1 |
| 1PSE | 1901 | 1697 | 1218 | 0 | 0 | 0 | 0 | 1 |
| 1QKH | 2984 | 2780 | 1272 | 0 | 0 | 0 | 0 | 1 |
| 1QP2 | 4991 | 4802 | 1083 | 0 | 0 | 0 | 0 | 1 |
| 1TUC | 1702 | 1525 | 1044 | 0 | 0 | 0 | 0 | 1 |
| 1W4H | 2209 | 2083 | 802 | 0 | 0 | 0 | 0 | 1 |
| 2BTH | 1085 | 955 | 801 | 0 | 0 | 0 | 0 | 1 |
| 2K1B | 2278 | 2132 | 934 | 0 | 0 | 0 | 0 | 1 |
| 2K28 | 2988 | 2838 | 920 | 7 | 0 | 0 | 0 | 1 |
| **2K5H** | 4107 | 3549 | 1125 | 2 | **2** | 0 | 0 | **3** |
| 2KVS | 3080 | 2879 | 1207 | 1 | 0 | 0 | 0 | 1 |
| 2KWH | 1544 | 1389 | 980 | 0 | 0 | 0 | 0 | 1 |
| 2KXD | 2360 | 2147 | 1173 | 2 | 0 | 0 | 0 | 1 |
| 2L33 | 1564 | 1352 | 1334 | 0 | 0 | 0 | 0 | 1 |
| 2LQK | 1769 | 1583 | 990 | 0 | 0 | 0 | 0 | 1 |
| 2WXC | 1131 | 1004 | 787 | 0 | 0 | 0 | 0 | 1 |
| 3DKM | 9037 | 8821 | 1239 | 34 | 0 | 0 | 0 | 1 |
| 4C26 | 3654 | 3468 | 1160 | 2 | 0 | 0 | 0 | 1 |
| 6EWS | 1637 | 1469 | 1036 | 0 | 0 | 0 | 0 | 1 |
| 6EWT | 1659 | 1488 | 1025 | 0 | 0 | 0 | 0 | 1 |
| 6EWU | 2436 | 2268 | 1010 | 0 | 0 | 0 | 0 | 1 |
| HEEH_KT_rd6_0746 | 941 | 813 | 807 | 0 | 0 | 0 | 0 | 1 |
| HEEH_KT_rd6_0793 | 940 | 811 | 796 | 0 | 0 | 0 | 0 | 1 |
| HHH_rd1_0142 | 936 | 807 | 804 | 0 | 0 | 0 | 0 | 1 |
| HHH_rd1_0244 | 1323 | 1195 | 771 | 0 | 0 | 0 | 0 | 1 |
| r11_1081_TrROS_Hall | 1082 | 935 | 898 | 0 | 0 | 0 | 0 | 1 |
| r12_757_TrROS_Hall | 1193 | 1031 | 982 | 0 | 0 | 0 | 0 | 1 |
| r18_3_TrROS_Hall | 1146 | 991 | 990 | 0 | 0 | 0 | 0 | 1 |

**Duplicate names — checked, and benign.** 48 rows across 6 proteins share a `name` with a
*different* `deltaG` (4C26 has a pair 2.6 kcal/mol apart). That looks alarming. But **46 of the 48
are double mutants** (`mut_type` contains `:`) and are removed by the `one_mut` filter before
evaluation. Only 2 survive, both in 2K5H, both already attributable to the known defect.
**Net effect on every reported number: zero.**

Likewise the "365 duplicate `mut_type`" figure for 1QKH (and 759 for 3DKM) is not a defect: those are
double mutants sharing a code, plus the five-row WT-replicate convention
(`.pdb`, `_wtm`, `_wte`, `_wty`, …) that every file carries.

**Impossible dG — none.** dG spans −12.23 to +7.85 across the source files. The evaluated subset is
narrower still: the benchmark `deltaG` column is **clipped to [−1, 5]** (observed min −0.998, max
4.999). Nothing is non-physical, and there are no NaN or infinite labels anywhere.

---

## 3. Tensor audit — the misalignment test

This was the check with the most at stake: *"a silent mismatch would misalign labels to structures,
which is the worst possible bug and would be invisible in every metric."*

**First, the right root.** Three tensor roots exist. `scripts/tensor_paths.py` documents the two
`meytav` roots, and auditing those alone reports 9 of 28 proteins missing — a false alarm.
**Production training and evaluation read a third root**,
`data/Processed_K50_dG_datasets/training_data` (`evaluate.py:550`), which holds all 368 proteins.
That is the root audited here. *Anything auditing only the meytav roots is auditing a different
dataset than the model trained on.*

**Shape agreement: 28/28.** For every protein, `coords (L,4,3)`, `mask (L,)`, `one_hot (N,L,21)`,
`prott5 (N,L,1024)` and `deltaG (N,)` agree on both N and L. Every N equals the mutation-file row
count after the ins/del filter **exactly** — delta 0 for all 28. Every dG tensor matches the CSV
column elementwise to ≤3.9e-07 (float32 round-trip). Zero non-finite values in any coords, mask,
one-hot, dG or embedding tensor; zero all-zero embedding rows.

**The strong test.** Shape agreement does not prove alignment — two tensors can be the right size
and still be permuted. So each row's one-hot was *decoded* and compared to the mutation its own
`mut_type` names: for row `A12G`, the decoded sequence must differ from the WT at exactly one
position, that position must be 12, and the change must be A→G.

> **29,517 single-point rows verified across 27 proteins. 0 position errors, 0 residue errors,
> 0 rows differing at more than one site.**

The one-hot alphabet was confirmed empirically as `ACDEFGHIKLMNPQRSTVWY` (index 0=A, 18=W, 19=Y),
rows sum to 1, and the 21st column is never used — consistent with §6.3's note that it is dropped at
`train.py:317`.

The **only** protein with anomalies is 2K5H: 2,336 rows differ from the WT at more than one position.
That number is not a new defect — it is the same 2,356 two-point rows §12.2 identified as
single mutations on the G11S background. **This is a positive result about the tensors:** they encode
the true multi-mutation background *correctly*. It is the CSV's `mut_type` column that is written
relative to the wrong reference. The structures were never wrong.

**Conclusion: there is no label/structure misalignment. The worst-case bug does not exist here.**

---

## 4. Train/test leakage — none

The split was reconstructed exactly as `evaluate.py` performs it: of 368 tensor directories, the 28
appearing in `mega_test.csv` are test; the remainder intersected with the PNAS `train_proteins.csv`
list gives **239 training proteins** (101 dirs are used by neither).

| test | result |
|---|---|
| train ∩ test by name | **NONE** |
| exact WT sequence shared | **NONE** (239 × 28 compared) |
| a test *variant* sequence equal to a training WT | **NONE** (all 28,314 variants × 239) |
| containment (one sequence inside another) | **NONE** |

Because "no exact match" is weak on its own, the **maximum** ungapped identity of each test protein
to any training protein was measured:

| test protein | closest training protein | identity |
|---|---|---|
| HHH_rd1_0244 | HHH_rd1_0473 | 0.349 |
| HHH_rd1_0142 | HHH_rd1_0473 | 0.349 |
| r12_757_TrROS_Hall | r11_951_TrROS_Hall | 0.296 |
| HEEH_KT_rd6_0793 | HEEH_KT_rd6_0872 | 0.279 |
| HEEH_KT_rd6_0746 | HHH_rd1_0516 | 0.256 |
| 2K1B | 2N88 | 0.204 |
| … | | |
| **median over 28** | | **0.159** |

The maximum anywhere is **0.349**, far below any homology threshold, and the highest values are
between *de novo designed* proteins from the same generator family (HHH_rd1, TrROS_Hall) — shared
design idiom, not shared ancestry. **No leakage. Nothing in the reported performance is inflated by
train/test contamination.**

*Caveat, stated honestly:* this is ungapped positional identity, not a gapped alignment, and it is
computed on WT sequences. It would miss a remote structural homolog with low sequence identity. What
it rules out is duplication and near-duplication, which is what a leak of consequence looks like.

---

## 5. NEW DEFECT 1 — `data_fixed/2K5H.csv` is positionally desynchronised from its tensors

**Severity: latent, currently harmless, would be severe on next use.**

The 2K5H fix moved the true WT (`2K5H.pdb`, dG 4.805) from index 2738 to row 0. Correct for the
label, but it **reordered 2,739 of 4,107 rows**, and the tensors were not regenerated:

```
data/       filt=3549  tensor=3549  max|dG_pt − csv| = 0.0000  rows mismatched =    0
data_fixed/ filt=3549  tensor=3549  max|dG_pt − csv| = 6.8914  rows mismatched = 2367
```

**Pointing the loader at `data_fixed/` would attach the wrong label to 2,367 of 3,549 structures
(67%)** — precisely the invisible misalignment this audit was asked to hunt for. It has simply not
happened yet, because the loader hard-codes `data/…/mutation_datasets` and nothing reads
`data_fixed/` positionally.

**Why nothing is corrupted today.** Seven scripts prefer `data_fixed/` first
(`gate_refrow.py`, `k13_muttype.py`, `loro_readout.py`, `sidechain_mech{,2,3,4}.py`). All of them
join on **`deltaG` value keyed to 6 decimals**, dropping ambiguous values — an order-invariant join.
Verified: both files yield 4,104 unique / 3 ambiguous values, identical. So §3.4's per-destination
slope table and the LORO readout are unaffected.

**Recommendation.** Do not silently re-sort a file whose row order is a foreign key into a tensor.
Either (a) add a header comment and a guard asserting `deltaG.pt` order matches, or (b) keep the
original row order and add an explicit `is_true_wt` column instead of moving the row. A one-line
assertion in `gate_refrow.py` comparing `data_fixed` row order against `deltaG.pt` would make this
un-missable.

---

## 6. NEW DEFECT 2 — 36 mutation files carry multiple WT backgrounds; 2K5H is the only test protein

Sweeping all 368 files for the 2K5H signature:

| role | count | proteins |
|---|---|---|
| **TEST** | **1** | **2K5H** (already known and fixed) |
| **TRAIN** | 4 | 1H8K, 1OPS, 2B88, 2HBB |
| unused by either split | 31 | 1A0N, 1BK2, 1BNZ, 1CSQ, 1EM7, 1GB4, 1MJC, 1PGA, 1PGX, 1S1N, 1SF0, 1SIF, 1UBQ, 1UCS, 2K52, 2KRS, 2KT8, 2KYB, 2LSS, 2LX2, 2LXK, 2LYP, 2M7O, 2MKX, 2PTL, 2ROT, 2ZW1, 5ECA, 5FWB, 5GU9, 5XR0 |

**The good news, and it is the important half: 2K5H is the only affected test protein.** The audit's
central worry — that another test protein carries a hidden constant label shift and is inflating the
offset-removal oracle a second time — is **ruled out**.

Of the 4 training proteins, 2 are false alarms (2B88 and 2HBB have multiple backgrounds but row 0 *is*
the true `.pdb` WT, shift +0.0000). Two have a mutant at row 0:

```
1H8K  row0 = 1H8K.pdb_R44S  dG 3.1155   true 1H8K.pdb dG 4.6122   shift −1.4968
1OPS  row0 = 1OPS.pdb_T53S  dG 3.5382   true 1OPS.pdb dG 4.5097   shift −0.9715
```

**This cannot affect training, and the reason is structural rather than lucky.** The training loader
`load_protein_data` returns `delta_g` and never constructs a `ddg` key — the row-0 subtraction exists
only in `load_test_protein_data` (`evaluate.py:198–200, 235`). Training regresses **absolute dG**, so
a wrong row-0 reference has nothing to propagate through. Verified by reading both loaders.

So: a real defect in 2 training files, with a measurable shift, that is **provably inert** under the
current objective. It becomes live the moment anyone trains on a ddG target. Worth a comment in the
loader so that change is not made unknowingly.

---

## 7. What was searched and NOT found

Stated explicitly, because a null result is only worth as much as the search behind it:

- No label corruption. 28,312 rows are **exactly** equal to the published benchmark.
- No misalignment between labels and structures. 29,517 rows verified at the level of *which residue
  changed where*, not merely tensor shape.
- No NaN, no infinities, no sentinel values, no all-zero embeddings, anywhere in 28 proteins.
- No impossible dG.
- No duplicate rows reaching any reported metric (46 of 48 filtered out; 2 are the known defect).
- No train/test leakage by name, by exact sequence, by containment, or by variant sequence; max
  identity 0.349, median 0.159.
- No second protein with a mutant reference row **in the test set**.

**One malformed file produced two false leads. It did not produce a third, and there is no second
malformed file in the test set.** The 2K5H correction in §12.1 remains the only label-level defect
affecting any reported number, and the corrected headline (pooled 0.4899, oracle 0.6443,
gain +0.1544) stands.

The two new findings are both **process** defects rather than data defects: a corrected file whose
row order silently broke a positional contract, and a class of multi-background files that happens
to be inert under the current training objective. Both are traps for future work rather than errors
in past work.

---

## 8. Artifacts

| path | what |
|---|---|
| `scripts/audit_labels.py` | label + tensor audit (meytav roots; kept for provenance) |
| `scripts/audit2.py` | production-root tensor audit + full split-leakage reconstruction |
| `scripts/t4.py` | the one-hot ↔ `mut_type` alignment test (the strong test) |
| `scripts/t5.py` | finiteness sweep + multi-background sweep over all 368 files |
| `/home/nissimb/audit_tensors_prod.csv` | per-protein tensor audit table |
| `/home/nissimb/audit_21109828.out` | leakage job log |

No production code, data, or configuration was modified. `gate_g4_cpu.py` re-run after all work:
**`baseline ... dG=-0.0030 width=1092`** — unchanged.
