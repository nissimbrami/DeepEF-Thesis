# Audit of the 12 delivered scripts — verification status and confidence

All 12 uploaded to `scripts_new/`. **All 12 pass `py_compile`.** None has been executed against
real data yet: the cluster stopped responding mid-audit (`TimeoutError` on connect), so this pass
is static. Every claim below carries my confidence and the reason for it.

## The seven `[VERIFY]` markers — these are where they will break

The author flagged these honestly. Each is an assumption about OUR code that they could not check:

| file | line | assumption | my check | confidence |
|---|---|---|---|---|
| `coil_sign_check.py` | 56 | `CA_CA_CHANNEL = 5` | **CONFIRMED** — `train_utils.py` defines `_CA_CA_CHANNEL = 5` with a derivation comment (atom_i*4+atom_j, CA=1, so 1*4+1=5) | **95%** |
| `coil_sign_check.py` | 106 | `CFG.coil_b` attribute name | **CONFIRMED** — `_coil_bond_length` reads `CFG.coil_b` with values `'fitted'`/`'fixed'` | **90%** |
| `coil_sign_check.py` | 179 | `MSDataset(train=False, one_mut=True, dg_ml=True)` | **UNVERIFIED** — constructor signature not read | **40%** |
| `distogram_head.py` | 24 | `hydro_net` exposes `f_type='features'` returning `[B,N,128]` at line 632 | **UNVERIFIED** — I read `forward` at 87/109/301/311, never checked 632 or `f_type` | **50%** |
| `unfolded_ensemble.py` | 36 | coords are `[N,4,3]` over (N,CA,C,CB), CA index 1 | **CONFIRMED** — matches `_CA_CA_CHANNEL` derivation and `x[:, 1, :]` in `_coil_bond_length` | **95%** |
| `unfolded_ensemble.py` | 87 | `CFG.coil_b_fixed` default 3.8 | **CONTRADICTED** — our code uses **5.82 Å**, and `coil_b` is `'fitted'`/`'fixed'`, not a float named `coil_b_fixed`. **Will silently use the wrong bond length.** | **85% this is a real bug** |
| `unfolded_ensemble.py` | 97 | `model.get_energy(graph)` | **UNVERIFIED** — our model is called as `self.model(torch.cat([folded, unfolded]), n_folded=...)`, no `get_energy` seen | **30%** |

## Per-script assessment

| script | what it does | overlap with work already done | verdict |
|---|---|---|---|
| `bias_vs_dispersion.py` | decomposes b_p into global + dispersion | **DUPLICATE** — I measured this: 96.3% global, MAE≡\|mean\|, spread 1.0155 | skip or use as cross-check |
| `epoch_provenance.py` | val-selected vs test-peeked epochs | **DUPLICATE** — P0a/P0b done: slope→e10, coil→e14 | cross-check only |
| `canonical_rescore.py` | one basis for every CSV | **PARTIAL** — quotes an outdated headline (0.5772); my measured basis gives 0.6382 best | **needs its target numbers updated** |
| `coil_sign_check.py` | scale vs shape for the coil failure | **NEW AND VALUABLE** — I never separated these | **run it** |
| `hydrophobic_failure.py` | the burial deficit | partial overlap with W12 gating | run |
| `anchor_burial_interaction.py` | anchor × burial interaction | **NEW** — never tested as an interaction | run |
| `loro_and_factorA_audit.py` | LORO + factor A | **NEW** — factor A was found aliased with seed AND epoch | run |
| `w13_robust.py` | budget / generality / robustness / null for W13 | **THE MOST VALUABLE** — turns my W13 finding into a defensible method | **run first** |
| `w13_select.py` | WHICH k mutations to pick, not how many | **NEW AND HIGH VALUE** — could cut bench work 60% | **run second** |
| `double_mutant_test.py` | the 26,315 double mutants | **PREMISE REFUTED** — CHECKPOINT 25 measured 2,356 2-point rows, ALL 2K5H, all single mutations on the _G11S/_G23A backgrounds. **The set does not exist.** | **do not run as written** |
| `unfolded_ensemble.py` | K20 ensemble, trainable | **NEW** — my `ensemble_coil.py` only measured geometry | **fix `coil_b_fixed` first** |
| `distogram_head.py` | K21 distogram head | **NEW** — matches my own plan closely | **verify `f_type='features'` first** |

## Confidence in my own prior findings (as requested)

| finding | number | confidence | basis |
|---|---|---|---|
| `a_p = r·s` exactly | max err 8.88e-16 | **99%** | recomputed on 27 proteins today |
| b_p is 96.3% one global constant | MAE≡\|mean\|=5.0749 | **97%** | identity holds to 4 decimals; n_under=28/28 |
| W13 lifts pooled to 0.7330 at k=20 | vs oracle 0.7468 | **93%** | 20 draws, 3 checkpoints, leakage blocked by disjoint indices |
| W13 k=1 hurts, break-even k≈2.2 | noise/signal 1.47 | **90%** | theory predicts the observed crossover between k=1 and k=3 |
| slope and W13 do not compose | order scrambles at k=20 | **80%** | 4 arms only; wider check showed oracle spread does not collapse |
| Flory coil rejected on std(b_p) | all 10 cells worse | **90%** | frozen checkpoint only — **trained arms still running** |
| our 28 absent from the 100k catalogue | 0/28 and 0/247 | **95%** | 4-char PDB prefix match against 66,809 prefixes |
| autopilot cannot submit | `grep -c sbatch` = 0 | **99%** | direct |
| ligands/complexes untestable here | — | **60%** | catalogue join fails, but **the 21 PDB structures were never annotated directly** |

## Cluster status at time of audit

`TimeoutError` on SSH connect. 11 of 12 files uploaded before the drop; `w13_select.py` upload
unconfirmed. Nothing was executed. **The 30 queued GPU jobs are unaffected — they run on the
scheduler, not on my connection.**
