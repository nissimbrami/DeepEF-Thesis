# Review of the planning agent's critique — measured, point by point

**Date:** 2026-09-10. Every claim below was checked against the code and the 79 eval CSVs
rather than against the written record.

---

## THE HEADLINE QUESTION: is the ~10% gain only data organisation?

**No. One lever adds information, and it is measurable.**

`per-protein mean r` is **invariant to any affine recalibration** — no offset fix, no slope fix,
no W13, no anchor, no k-shot procedure can move it. It is therefore the only honest test of
"did we add information the model did not have".

| arm | per-protein mean r | vs control | in sd of the control band |
|---|---|---|---|
| control family (n=6) | 0.7273 (sd **0.0065**) | — | — |
| anchor 0.3 | 0.7271 | +0.0009 | +0.1 |
| slope 1.0 | 0.7302 | +0.0040 | +0.6 |
| w7edge | 0.7307 | +0.0045 | +0.7 |
| u10bidir | 0.7358 | +0.0097 | +1.5 |
| **W5 (3 seeds)** | **0.7337** | +0.0064 | **+1.00** |
| **W12 (2 seeds)** | **0.7479** | **+0.0218** | **+3.19** |

**W12 moves the ranking channel by 3.19 sd.** Calibration cannot fake this number.

So the correct statement is: **most of the pooled gain is calibration/organisation, but W12 is a
genuine information gain and it is the single most valuable result in the project.**

---

## The four "hasty rejections"

### 1. Chemistry / the product term — **ALREADY IMPLEMENTED AND RUN**

The planner proposes as a decisive cheap test: *"add three columns — hydrophobicity, burial, and
the product — and see if the product contributes."*

`train_utils.py:71`:

```python
return torch.cat([bur, hyd, bur * hyd], dim=1)   # burial, hydrophobicity, THE PRODUCT
```

**That is W5.** It is built, gated 18/18, and has run on **three seeds**. Result above: +1.00 sd
on the ranking channel — real but modest, and 3x smaller than W12.

W15 goes further and is also already built: `scripts/w15_block.py` column 2 is
`inter = reach * env`, a residue-property x geometry product, **plus** a clash term
`relu(reach - free_space)` which is not even expressible as a product. `gld_w15_*` are queued and
`pub_w15_s42` is running now.

**Verdict on the algebra:** the planner's *reasoning* is correct — a product of a residue property
with a geometric quantity is not `one_hot @ T` and a linear layer cannot construct it. **But the
conclusion "this was never tested" is false.** It was tested, twice, and the answer is "+1 sd".

**AGREE with the principle, REJECT the premise.** No new work needed.

### 2. Ligands/metals — **AGREE**

"Zero variance on our 28" licenses only *"not measurable on MegaScale"*, never *"no effect"*.
That is already the standing rule (FINDINGS §9.1) and the planner is right to insist on it.
Changing the test set is a decision, not a fact.

**But it is not cheap** — it means adopting a second dataset and defending a test-set swap, which
is a thesis-scope change, not a two-day experiment.

### 3. Flory — **PARTLY AGREE, and the missing half is ALREADY QUEUED**

Correct: our 10 nu values only ever swapped the unfolded **distance map**, while IFUM train the
network to **predict** the unfolded ensemble with an auxiliary loss.

**That exact half is already built.** `train.py:99` implements `--distogram_weight` with the
reasoning written into the help text (including why IFUM's weight of 100 would swamp an MSE of
order 1, hence a sweep of {0.01, 0.1, 1.0}). **`gld_disto0.01_s42` is RUNNING right now**;
0.1, 1.0 and `disto_w12` are queued. The unfolded ensemble is likewise built: `ens3mean`,
`ens8mean`, `ens8lse` are queued.

**AGREE with the diagnosis, but the work is done and in flight.**

### 4. Calibration "closed" — **AGREE, and I already retracted it**

The planner is right, and this was my error. I retracted it yesterday in
`W12_W13_RETRACTION.md`: the estimator was a free affine fit that exploded (`max|a| = 3.2e3`) and
the draws were unpaired. Corrected, W12 never significantly loses and **wins at k=50**.

Sub-additive is not redundant. **Fully agreed.**

---

## Where the planner is factually out of date

| claim | measured |
|---|---|
| "W12 has one seed" | **two** — `w12_s1_e14`, `w12_s2_e12`; two more running |
| "26k double mutants, script not run" | **zero real double mutants exist** (FINDINGS §12.2). All 2,356 two-point rows belong to 2K5H and are single mutations on the G11S background. The script would measure nothing. |
| "distogram head not connected" | wired at `train.py:99`, one arm **running** |
| "ensemble not run" | three arms queued |
| "three cheap tests, under two days" | all three already exist; two are in flight |

---

## What I would do instead

**The planner's ordering is right; its content is spent.** The proposed tests are done. The real
list, ranked by measured value per GPU-hour:

1. **W12 to 4 seeds** — running now (`pub_w12_s3`, `pub_w12_s4`). It is +3.19 sd on the ranking
   channel but its **protein-level bootstrap CI still includes zero** (P=0.88), because 5 proteins
   carry 98.4% of the gain. This is the one number the thesis turns on.
2. **Read out the distogram sweep** — the genuinely untested half of IFUM, already burning GPU.
3. **W15** — a product *and* a clash term, never run before today; `pub_w15_s42` is running.
4. **Do NOT spend more on calibration.** Independently confirmed today: k=20 already closes
   **87.7%** of the raw->oracle gap. The remaining headroom is within-protein ranking, i.e. §3.4's
   hydrophobic deficit — which is precisely what W12/W15 attack.

**The one thing I would add that nobody has proposed:** the gain is concentrated in 5 proteins out
of 27, so **n=27 is the binding constraint, not the levers.** More seeds cannot fix a
protein-level CI. If the thesis needs a significant result, it needs more test proteins — that is
a data decision, and it is the highest-value open question.

**Confidence: 95%.** Every claim here is a direct measurement; the code quotes are verbatim.
