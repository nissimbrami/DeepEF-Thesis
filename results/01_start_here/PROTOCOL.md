# THE PROTOCOL — one task at a time, and planning has equal standing

**Adopted 2026-09-10** after a wave of twelve arms produced zero surviving positive results.
**That outcome was mostly a fact about the design, not about the science.**

This file is written to survive context compaction. Read it before submitting anything.

---

## 0. Why this exists

Eleven identified errors. **Five were sealed before a line of code was written.**

| stage | errors | examples |
|---|---|---|
| **PLANNING** | **5** | Flory implemented as an input swap, not a learned reference; burial and descriptors designed as per-residue blocks identical in both states; the RBF bank summed back to flat; `span4` changing range AND directionality together |
| EXECUTION | 3 | patch applied to one of two call sites; W5 arms run on `dg` instead of `ddg`; single-seed arms judged as confirmation |
| VERIFICATION | 3 | the coil scored on ddG which cancels it; W12 judged with an estimator whose slope blew to 3,200; Flory and descriptors written up without checking the model trained |

**No amount of verification rescues a planning error.** Planning is first in order and first in
severity.

---

## 1. One active task. Three stages, in sequence, uninterrupted.

Background GPU jobs are **not** active tasks — they run while CPU analysis proceeds. What must
never be interleaved is the **planning, execution and verification of the same task.**

### Stage 1 — PLAN (no time budget; take what it needs)

Write, before any code:

- the question, the hypothesis, and **what result would falsify it**
- the metric, and the exact configuration
- the sentence *"if the result is X, the conclusion is Y"* — completed in advance

**The plan is not finished until that sentence can be written.**

### Stage 2 — EXECUTE

End to end, to the plan. **A deviation sends you back to Stage 1 with the reason recorded.**

### Stage 3 — VERIFY

- did the result meet the pre-registered criterion?
- **did the model actually train?** (loss moved, RMSE moved, gradients finite)
- did the predicted *mechanism* occur, not just the number?

**The task is not done until this stage passes.**

### Definition of DONE (all four, or the task stays open)

1. plan written · 2. execution complete · 3. verification passed ·
4. result reported with **mean AND median**, plus a training-health line

---

## 2. The pre-flight test — eight questions, ten minutes, before any code

The first four catch the five planning errors above. The last four catch what the wave exposed
after that.

| # | question | would have caught |
|---|---|---|
| 1 | **What exactly is the idea in the source, and what exactly am I implementing?** If they differ, record the gap and decide explicitly whether it is acceptable. | **Flory** |
| 2 | **Does the feature differ between the folded and unfolded graphs?** If not, it cancels in dG **by construction** — do not run it. Enforced by `scripts/gate_state_dependence.py`. | **W6 descriptors, the original W5 burial bug** |
| 3 | **Does the change alter more than one thing?** If yes, the result is unattributable. | **`span4`** (range + directionality), **W12+W5** (block + objective) |
| 4 | **Which metric does the lever act on, and which am I measuring?** If they differ, the measurement is void. | **the coil on ddG, BSA vs the pre-training loss** |
| 5 | **Is this a screen or a confirmation?** A screen is one seed, no multiplicity correction, and yields a *ranking* — never a claim. | `slope 0.7` judged as if confirmed |
| 6 | **How many seeds does the effect size require?** Control SE 0.0027, single-arm SE 0.0065, difference SE ~0.0070. A 0.011 effect is 1.6σ at n=1. **Three seeds minimum for any claim, five for a headline.** | the whole wave |
| 7 | **Will I report mean AND median?** They can disagree by 5σ. | **W5** (+1.00σ mean, −4.15σ median) |
| 8 | **After editing shared code, which call site does `__main__` actually reach?** Grep every one. | the loader patch applied to dead code |

---

## 3. Rejections are written as "rejected in configuration X"

Never "dead". Four of six rejected levers were rejected in a configuration in which the idea
could not express itself. State the configuration, and state what would change the verdict.

---

## 4. Standing measurement rules

- **Screen vs confirm are never mixed.** Multiplicity correction applies only within a screen.
- **Every per-protein number states mean or median, and reports both.**
- **Every arm reports a training-health line before its score.** A frozen loss is a build failure,
  not a result, and is marked as such in the table.
- **A DONE message is not evidence.** Only `ls` on the artifact.
- **State-dependence gate is mandatory** for any new feature block.

**Confidence in this protocol: it is a process document, not a measurement. Its warrant is the
eleven errors it is derived from, each of which is individually documented.**
