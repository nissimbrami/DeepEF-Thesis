# Does `data/FINAL_DATASET_100k_030926.csv` have any legitimate use?

**Short answer: one, and it is not a join.** The catalogue cannot be joined to *anything* we
train or test on — not by PDB id, not by sequence, and the sequence route is impossible in
principle, not merely empty. What it *can* do, with no join at all, is describe the
pre-training distribution, and that comparison produces a large, clean, well-powered
distribution-shift result. That result does **not**, however, explain `b_p`.

Gate after all work: `scripts/gate_g4_cpu.py` -> `baseline ... PASS dG=-0.0030 width=1092`,
`G4-CPU: ALL PASS`. No repo file was modified; everything below lives in `scratch_cat/`.

---

## 0. Provenance of the protein lists (traced, not assumed)

`Megascale-fineTuning/train.py`:

- L1198 `tensor_root_dir = './data/Processed_K50_dG_datasets/training_data'`
- L245  `TM_PATH = './data/ThermoMPNN/mega_test.csv'`
- L388  `self.protein_dirs = os.listdir(self.tensor_root_dir)`
- L392-395 test = dirs whose name is in `mega_test.csv['name'].split('.')[0]`; **train = the rest**

Executed, not inferred: **368 directories -> 28 test, 340 train.** All 368 have a matching
`mutation_datasets/<P>.csv` (`n_with_mut_csv = 368`), so every number below is computed on
real rows, not on a name list.

Test set (28), for the record:
`1GYZ 1PSE 1QKH 1QP2 1TUC 1W4H 2BTH 2K1B 2K28 2K5H 2KVS 2KWH 2KXD 2L33 2LQK 2WXC 3DKM 4C26
6EWS 6EWT 6EWU HEEH_KT_rd6_0746 HEEH_KT_rd6_0793 HHH_rd1_0142 HHH_rd1_0244 r11_1081_TrROS_Hall
r12_757_TrROS_Hall r18_3_TrROS_Hall`

Of the 340 training proteins, **226 are real 4-character PDB ids** and 114 are Rocklin/Baker
designed names (`EEHEE_rd3_*`, `HHH_*`, `*_TrROS_*`) that cannot exist in the PDB.

---

## 1. Does the catalogue join the TRAINING proteins?  **No. 0 / 226.**

| join key | attempted against | hits |
|---|---|---|
| 4-char PDB id, uppercased | 226 PDB-like TRAIN proteins vs 66,945 unique catalogue PDB ids | **0** |
| 4-char PDB id, uppercased | 21 PDB-like TEST proteins | **0** (confirms the briefed 0/28) |
| full `protein_id` string | 340 TRAIN names | **0** |
| **sequence** | — | **impossible, see below** |

This is not a near miss. The catalogue holds 66,945 distinct PDB entries — roughly a third of
the whole PDB — and hits **zero** of 226 real PDB ids that we fine-tune on. Under any random
model that is a vanishing probability; the exclusion is structural, and section 2 shows the
mechanism (it is a *length* filter, not a blacklist).

### The sequence join is impossible, not merely empty

Verified against the actual header (`shift.json:join.columns_checked_for_sequence`). The
catalogue's 21 columns are:

```
rank, protein_id, split, loss, lossd, lossg, lossc, PDB_ID_and_Entity, Method,
Resolution (Å), AA Length, Oligomeric State, Is Complex?, Ligands_x, Global Symmetry,
Chain Composition, BSA_Percentage, BSA, BSA_Numeric_x, Ligands_y, BSA_Numeric_y
```

**There is no sequence column.** `catalogue_has_no_sequence_column = true`. A sequence-level
join was never on the table — it is not that we tried and got nothing, it is that the key does
not exist in the file. This closes question 1 permanently: no amount of fuzzy matching,
re-derivation, or effort will produce a join, short of re-downloading 66,945 PDB entries.

**Consequence: the "structural properties predict TRAINING error at large n" idea is dead.**
It required the join, and the join cannot exist.

---

## 2. Are our proteins typical of the pre-training distribution?  **No — extreme outliers.**

This needs no join, only a comparison, and it is by far the strongest thing in the file.

### 2a. Length

| | n | min | median | mean | max |
|---|---|---|---|---|---|
| catalogue (pre-training) | 99,708 | 20 | **477** | 1167.7 | 89,160 |
| our TRAIN (340) | 340 | 32 | 55 | 53.1 | 74 |
| our TEST (28) | 28 | 43 | 56 | 55.8 | 72 |

Every one of our 368 proteins is shorter than the catalogue's **1st percentile is wide**:
the catalogue's 25th percentile is 258 residues, ~4.6x our longest protein.

- Catalogue mass at or below our longest protein (74 aa): **3.05%**
- Catalogue mass inside our full range (32–74 aa): **2.53%** (2,526 of 99,708)
- Median test protein (56 aa) sits at catalogue percentile **1.77**

**The number that matters most — gradient share.** Pre-training loss is per-residue, so a
protein's influence scales with its length, not with its row count. Residues contributed by
the 32–74 aa regime:

> **140,383 of 116,431,172 residues = 0.1206%**

The model was pre-trained on a corpus in which our entire problem domain supplied roughly
**one residue in 830**. That is the distribution-shift finding, and it is not marginal: it is
three orders of magnitude.

### 2b. The short tail is a *different kind of structure*, not just a shorter one

Restricting the catalogue to our own length regime (32–74 aa, n=2,526) versus the whole file:

| property | whole catalogue | our regime (32–74 aa) | shift |
|---|---|---|---|
| Solution/solid-state NMR | 8.56% | **77.43%** | **9.05x enriched** |
| X-ray diffraction | 88.7% | 22.1% | 4x depleted |
| has a resolution value at all | 90.9% | **22.5%** | — |
| Is Complex? = Yes | 67.7% | 11.8% | 5.7x depleted |
| Oligomeric State = Monomer | 31.8% | **88.2%** | 2.8x enriched |
| has ligands | 75.3% | 43.3% | 1.7x depleted |
| BSA = 0 (no interface) | 31.2% | **87.8%** | 2.8x enriched |
| mean BSA | 13.02 | 2.64 | 4.9x lower |

So the sliver of pre-training data that resembles our domain at all is overwhelmingly **NMR,
monomeric, interface-free, ligand-poor** — while the 99.88% that dominated the gradient is
**X-ray, complexed, buried-interface, ligand-bearing**.

This is a coherent, mechanistically sensible story and it lines up with levers already found:
the model's folded-state reference was learned mostly from large X-ray assemblies with real
buried interfaces, and it is being applied to small ligand-free monomers with almost no buried
surface. That is exactly the regime where a whole-protein reference-state offset would appear
— i.e. `b_p`. It also explains *post hoc* why BSA and ligands registered as `a_p` levers.

### 2c. The MegaScale-like stratum barely exists

Filtering the catalogue to what our 28 actually are — `Monomer` **and** `Is Complex? = No`
**and** no ligand:

- 10,358 rows = 10.33% of the catalogue
- of those, also within 32–74 aa: **1,643 rows = 1.64%**

**Only 1.64% of the pre-training set is the kind of object we ask the model to score.**

### 2d. Composition: TRAIN vs TEST, n=340 vs n=28

Welch t-test per residue, Bonferroni alpha = 0.05/20 = 0.0025.

| aa | train freq | test freq | p |
|---|---|---|---|
| **T** | 0.0660 | **0.0395** | **2.73e-07** (survives) |
| M | 0.0191 | 0.0113 | 0.0113 (nominal only) |
| L | 0.0827 | 0.0963 | 0.072 |
| all other 17 | — | — | > 0.09 |

**Threonine is genuinely depleted in the test set** — test proteins carry 60% of the train-set
T content, and it survives correction comfortably. Every other residue is consistent with
chance. This is a real, reportable train/test composition asymmetry.

It is worth stating plainly what this is *not*: it did not translate into any predictive power
over `b_p` (section 4).

---

## 3. Ofir's train/test MEAN GAP, computed here.  **Honest negative — the gap is ~zero.**

Ofir found RMSE tracked the train/test mean gap. The analogous quantity here, computed on real
`deltaG` values from `mutation_datasets/`:

**Per-protein (n_train=340, n_test=28):**

| quantity | train mean ± sd | test mean ± sd | gap (test-train) | Cohen d | Welch p | MWU p |
|---|---|---|---|---|---|---|
| WT dG | 3.0930 ± 1.3362 | 2.9895 ± 0.9263 | **-0.1034** | -0.090 | 0.588 | 0.992 |
| mean dG over all variants | 1.7315 ± 1.1907 | 1.6518 ± 0.9635 | -0.0797 | -0.074 | 0.683 | 0.932 |
| within-protein sd of dG | 1.5014 ± 0.7613 | 1.3846 ± 0.5578 | -0.1168 | -0.175 | 0.309 | 0.708 |
| sequence length | 53.09 ± 10.82 | 55.82 ± 9.75 | +2.73 | +0.265 | 0.168 | 0.139 |

**Pooled over mutations (n_train = 670,565 rows; n_test = 63,380 rows):**

- train mean dG = 1.7387 (sd 2.2021), test mean dG = 1.3632 (sd 1.8915)
- **gap = -0.3756 kcal/mol**

### Why this kills the hypothesis rather than supporting it

The per-protein WT-dG gap is **-0.103 kcal/mol at p=0.588** — statistically indistinguishable
from zero at n=340 vs 28, which is a *well-powered* test, not an underpowered one. Even the
pooled-over-mutations gap, the largest version of the number, is **-0.376**.

Compare against what `b_p` actually is: **std(b_p) = 1.5741**.

> The train/test mean-dG gap is **-0.103**, i.e. **6.6%** of one standard deviation of `b_p`
> (or 24% using the most generous pooled figure). The offset it could mechanically produce is
> a *shared constant* across all 28 proteins in any case, whereas `b_p` is a **per-protein
> spread** — a common gap cannot generate dispersion at all.

So: **the train/test mean gap is not the mechanism behind `b_p`.** Ofir's finding does not
carry over. The distributions of dG that our model trains on and is tested on are, to within
noise, the same distribution. Whatever `b_p` is, it is not inherited from a label-scale shift.

This is a clean negative and it is worth having: it removes a plausible-sounding explanation
from the board and it means the offset has to come from the *input/representation* side
(section 2's structural shift), not from the label side.

---

## 4. Does any of it predict `b_p`?  **No. Nothing survives correction.**

The metric rule says whole-protein/reference-state levers must be scored on dG or `b_p`, so I
fitted, per protein and per checkpoint, `pred_deltaG ~ a_p * deltaG + b_p` across all **10**
eval CSVs and averaged over checkpoints. **n = 28. Threshold |r| > 0.374 at p = 0.05.**
(This convention gives std(b_p) = 0.7914, median a_p = 0.4148 — a different, checkpoint-averaged
convention from the briefed std(b_p)=1.5741 / median a_p=0.4990; only the correlations are used
below, and those are convention-free.)

Catalogue-derived / distribution-shift features:

| feature | r vs b_p | p | r vs a_p | p |
|---|---|---|---|---|
| length percentile within the 100k catalogue | **+0.0427** | 0.829 | -0.2081 | 0.288 |
| L2 distance from the 340-protein train composition centroid | +0.1904 | 0.332 | -0.1083 | 0.583 |
| Mahalanobis distance from that centroid | **+0.0277** | 0.889 | -0.0219 | 0.912 |
| sequence length | +0.0535 | 0.787 | -0.2174 | 0.267 |
| mean dG | +0.1156 | 0.558 | -0.2132 | 0.276 |
| WT dG | -0.1257 | 0.524 | -0.4205 | 0.026 |
| f_T (the Bonferroni-surviving composition hit) | -0.1189 | 0.547 | -0.2877 | 0.138 |

**The atypicality features are flatly null against `b_p`** — `r = 0.043` and `r = 0.028` are
about as close to zero as n=28 permits. The residue-composition sweep produced three nominal
`b_p` hits (f_H -0.522 p=0.0044; f_R -0.448 p=0.0169; f_K +0.390 p=0.0400) and two nominal
`a_p` hits (f_F -0.510 p=0.0056; wt_dG -0.421 p=0.0259), but across **m = 25 tests**:

- Bonferroni alpha = 0.0020 -> **NONE survive**, for either `b_p` or `a_p`
- Benjamini-Hochberg FDR 5% -> **NONE survive**, for either

At n=28 with 25 tests, expecting ~1.25 nominal hits by chance and observing 3 is unremarkable.
I am reporting these as **not findings**. The charged-residue pattern (H, R negative; K
positive) is the sort of thing that would be tempting to write up; it does not clear the bar
and should not be.

---

## 5. Verdict on the file

| use | status |
|---|---|
| join to the 28 test proteins | dead (0/28, by construction) |
| join to the 340 training proteins | **dead — 0/226, verified** |
| join by sequence | **impossible — no sequence column exists in the file** |
| structural properties -> training error at large n | **dead** (requires the join) |
| describe the pre-training distribution / quantify shift | **ALIVE — the one legitimate use** |
| explain `b_p` via train/test dG gap | **negative** — gap is -0.103 (p=0.59) vs std(b_p)=1.57 |
| explain `b_p` via catalogue atypicality | **negative** — r=+0.043, n=28 |
| `loss`/`lossd`/`lossc` correlations | uninformative by prior agreement; not attempted |

**The file is not worthless, but it is worth exactly one thing:** it is the only artifact that
documents what the model saw during pre-training, and it says our entire problem domain was
**0.12% of pre-training residues**, drawn from a structural regime (X-ray, complexed,
interface-rich) that is the near-opposite of our test regime (NMR-like, monomeric,
interface-free). That is a legitimate, quotable, thesis-grade distribution-shift statement
that needs no join.

**But it must be stated as a shift, not as an explanation of `b_p`.** Both direct tests of
"shift causes the offset" came back null (r = +0.043 against `b_p`), and the mean-gap
mechanism came back at 6.6% of the effect it would need to explain. The honest write-up is:
*the pre-training distribution is severely mismatched to the fine-tuning/test domain, and we
could not demonstrate that this mismatch is what produces the per-protein offset.*

### What would actually be needed to revive the join

Re-derive sequences for the 66,945 catalogue PDB entries (they are not in the file) and match
by sequence identity to the 226 PDB-like training proteins. Given that these are 32–74 aa
domains often excised from larger chains, even that would likely need substring/HMM matching
rather than exact equality. Cost is high, and section 2a suggests the yield would be near zero
anyway: the catalogue has only 2,526 rows in our length regime at all.

---

## Artifacts (all on the cluster, nothing committed, no repo file modified)

| path | contents |
|---|---|
| `/home/nissimb/DeepPEF/scratch_cat/cat_analysis.py` | join + gap + composition |
| `/home/nissimb/DeepPEF/scratch_cat/cat_out.json` | all section 1/3 numbers |
| `/home/nissimb/DeepPEF/scratch_cat/cat_perprotein.csv` | 368 proteins x (L, n_mut, mean_dG, wt_dG, std_dG, split) |
| `/home/nissimb/DeepPEF/scratch_cat/cat_comp.csv` | 368 x 20 residue frequencies + split |
| `/home/nissimb/DeepPEF/scratch_cat/shift.py`, `shift.json` | section 2, incl. the column-list proof |
| `/home/nissimb/DeepPEF/scratch_cat/bp_link.py`, `bp_link.json` | section 4 correlations |
| `/home/nissimb/DeepPEF/scratch_cat/bp_features.csv` | 28 x (features, a_p, b_p) |
| `/home/nissimb/DeepPEF/scratch_cat/gate.txt` | gate output, `dG=-0.0030 width=1092` |

SLURM CPU jobs: 21085626, 21085645, 21085673 (all COMPLETED, no GPU).
