# IFUM (Lee et al., Nat Commun, 21 Jan 2026) — read in full. What it actually does.

**DOI 10.1038/s41467-026-68637-4 · PMID 41565673**
Read from the PMC full text. This corrects several assumptions I had been working from.

## The headline number, and what it is

**PCC 0.78 / RMSE 1.16 kcal/mol on ABSOLUTE dG**, Mega-scale test set of 86 domains, with labels
and predictions clamped to [-1, 5] kcal/mol.

**Nissim was right and I was wrong:** the headline is **dG, not ddG**, and it is far above
anything in this project. Their ddG numbers are also strong — point mutants PCC **0.81**, indels
0.80, double mutants 0.63.

## The unfolded state: ANALYTIC, not sampled and not learned

**Equation 2:**

    r(i,j)_U^2 = 6 x 1.927 x |i-j|^0.598

A Flory random coil, a function of **sequence separation only**. No sampling. No learned unfolded
representation.

**This is the single most important correction to my plan.** I built a sampler
(`sample_coil_coords`, 8 conformations) because I assumed IFUM's "ensemble" meant sampled
conformations. **It does not.** Their unfolded state is the same closed-form object we already
had — the analytic coil.

**So the sentence I wrote in `FLORY_QUARTER_TESTED.md` — "IFUM predict a sampled ensemble, we
substituted a map" — is wrong on the first half.** The difference is not sampled-vs-analytic.

## What IFUM actually adds: the distogram is BOTH input and label

- **INPUT:** a **folded-state** distogram of binned Ca distances from **ESMFold** ([N,N,21]),
  alongside ProtT5 sequence embeddings and **ESM-IF1 structure embeddings**.
- **LABEL:** an **equilibrium ensemble distogram** — the folded and unfolded distance maps mixed
  by Boltzmann weights.

**Equation 4:** `[F]:[U] = exp(dG/RT) : 1`

**Equation 5:**

    p(r_F) = exp(dG/RT) / (1 + exp(dG/RT))
    p(r_U) = 1          / (1 + exp(dG/RT))

**The dG in that mixing is the EXPERIMENTAL LABEL, and the mixing happens only at training time.**
So the auxiliary target is: *"given this sequence and its folded structure, predict the distance
distribution the protein actually occupies at equilibrium, where the folded/unfolded mixture is
set by its measured stability."*

**That is the real idea, and it is not what I implemented.** It is a way of injecting the
experimental dG into a dense per-residue-pair signal, rather than only into a single scalar loss.

## Loss

**Equation 6:** `L = L_dG + 100 * L_ensemble + L_seq`

- `L_dG` — Gaussian negative log-likelihood on dG
- `L_ensemble` — cross-entropy on the equilibrium distogram, **weight 100**
- `L_seq` — sequence-reconstruction cross-entropy

**The weight 100 is defensible in their setup** because the auxiliary target carries the label
information; it is not an arbitrary regulariser. My "start at 0.1" reasoning was based on treating
it as a regulariser, which it is not.

## Per-residue dG

> *"IFUM predicts a protein's dG by predicting per-residue dG contributions and then through an
> unweighted summation of these contributions."*

We already do a sum over residues, so this part matches.

## The ablation we have been quoting

Training without equilibrium-ensemble prediction: **PCC 0.78 -> 0.70, RMSE 1.16 -> 1.39.**

**It removes only the ensemble prediction**, not the analytic coil as well. So the ablation
**does** isolate the learned half — my earlier claim that "their ablation removes both together
and isolates neither" is **wrong**, and `FLORY_QUARTER_TESTED.md` must be corrected.

## What this means for our arms

| our implementation | IFUM | status |
|---|---|---|
| analytic coil replaces the unfolded **input** map | analytic coil used to build a **training label** | **different use of the same object** |
| distogram head predicts sampled coil distances | predicts the **equilibrium mixture** weighted by experimental dG | **wrong target** |
| weight 0.1 | weight 100 | ours is a regulariser; theirs carries label information |
| no folded distogram as input | **ESMFold distogram is an input** | **missing entirely** |
| no ESM-IF1 structure embedding | ESM-IF1 [N,512] | missing |

**The running `ensdisto` arms predict sampled coil geometry. IFUM predicts an experimentally
weighted folded/unfolded mixture. Those are different objectives, and only the second injects
stability information.**

## What I would build instead

The minimal faithful version, using only what we already have:

1. Build the unfolded map analytically with **Eq. 2** — we already have `_flory_unfolded_graph`.
2. Build the folded map from the **real coordinates** we already load.
3. Mix them per residue pair with **Eq. 5**, using the **experimental dG of that variant** (we
   have it: `batch['delta_g']`).
4. Train the existing `DistogramHead` on that mixed label with cross-entropy.
5. Train on **dG** (their headline channel) and on ddG, per the four-combinations rule.

**Steps 1-3 need no new machinery.** Step 4 is the head that is already wired.

**Confidence: 95%** on the reading (two independent fetches of the PMC full text agree on the
equations); the residual is that I have not seen the supplementary methods.
