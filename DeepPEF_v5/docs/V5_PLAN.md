# DeepPEF v5 — The Ultimate Version: Build Plan & Rationale

**Baseline to beat:** PCC 0.5259 (dG). **Target:** ≥0.70.
**Design principle:** every change is grounded in the REAL verified code (read 3x), touches ONE
concern at a time, and is toggled by a flag so we can ablate. We KEEP the proven working flow
(energy-difference dG, PNAS filtering, Huber+ranking, k-NN GAT) and ADD verified improvements.

---

## What we keep (proven, DO NOT break)
- Energy-difference framing: `dG = E_unfolded − E_folded` (pnas_train.get_deltaG).
- PNAS filtering (`--one_mut --dg_ml`, dG clamp [-1,5]) — proven essential (unfiltered hurt 0.41).
- Huber + pairwise ranking loss on dG.
- k-NN GAT (k=30, 12 Å) — our own +0.043 result.
- `--mini_batch_size 16` (OOM-safe), `WANDB_MODE=disabled`, cache cleanup.

## What we ADD in v5 (each flag-gated, dG-focused)

### 1. Length / mass-balance normalization  ★ (the cleanest dG fix)
- **Where:** `model/hydro_net_v5.py` `PEM.get_energy` — currently `E = torch.sum(Fh, dim=(1,2))`.
- **Change:** divide by residue count → intensive energy. Flag `--length_norm`.
- **Why:** our sum-energy scales with protein length; a 200aa protein looks 4x more stable than a
  50aa one just from size. Per-residue removes that bias so the loss learns real stability.
- **Verified:** confirmed `get_energy` is a plain sum at hydro_net.py:525.

### 2. Serial fusion ON by default (already coded, verified)
- **Where:** `PEM.__init__` / forward already implement `serial_fusion` (hydro_net.py:319-322, 443-446).
- **Change:** default it ON in the v5 config + training. Flag `--serial_fusion`.
- **Why:** injects ProtT5 into GNN message-passing so neighbors share PLM signal.

### 3. Correlation loss (align training with the PCC metric)
- **Where:** `training/losses_v5.py` new `pearson_loss` / `ccc_loss`; added in `pnas_train_v5.py`.
- **Change:** `loss += corr_weight * (1 - pearson(output, delta_g))`. Flag `--corr_weight`.
- **Why:** we train Huber (point error) but are graded on PCC (trend). Optimizing correlation
  directly rewards correct ordering — exactly the metric.

### 4. dual_esmif embeddings route (inverse-folding structural prior)
- **Where:** `new_dataset` already supports `EMB_TYPE='dual_esmif'` (1024+512=1536). Verified.
- **Change:** v5 config sets `--emb_type dual_esmif` as a tier; needs `esmif_enc.pt` per protein
  (generation script referenced: generate_proteinmpnn_features.py exists).
- **Why:** ESM-IF1 is an inverse-folding model; its features carry structural priors — the biggest
  documented gap to ThermoMPNN.

### 5. Multi-seed ensembling scaffolding (do last, reliable)
- **Where:** `training/ensemble_v5.py` — averages best_model.pt across seeds at eval.
- **Why:** cancels per-seed noise, reliable +0.02-0.04.

## The evolution ladder (one change at a time — measure each)
```
Rung 0  baseline (reproduce 0.5259): huber_rank + knn_gat + one_mut + dg_ml, prott5
Rung 1  + length_norm
Rung 2  + serial_fusion
Rung 3  + corr_loss
Rung 4  switch emb_type -> dual_esmif  (needs esmif_enc.pt)
Rung 5  5-seed ensemble of the best rung
```
Each rung inherits the winner of the previous. If a rung REGRESSES, revert it (memory lesson:
don't stack losers). Keep the exact baseline command fixed except the one new flag.

## Files in DeepPEF_v5/
```
model/hydro_net_v5.py   — PEM with length_norm added to get_energy (rest identical)
model/model_cfg_v5.py   — v5 defaults (serial_fusion on, length_norm flag)
training/losses_v5.py    — pearson_loss, ccc_loss (new)
training/pnas_train_v5.py — training with --length_norm and --corr_weight wired in
training/ensemble_v5.py   — seed-ensemble eval
run_v5_evolution.sh      — the rung ladder
docs/V5_PLAN.md          — this file
docs/V5_README.md        — how to run
```

## Verification protocol (3x, every angle) — DONE for the read phase
1. Read hydro_net.py fully → confirmed energy sum, feature layout, serial_fusion/learned_aa coded.
2. Read pnas_train.py fully → confirmed dG target, loss composition, get_deltaG, validation PCC.
3. Read new_dataset.py + train_utils.py + model_cfg.py → confirmed data flow, graph construction,
   emb_type routing, dG clamp.
All v5 changes are diffs against these verified facts, not guesses.
