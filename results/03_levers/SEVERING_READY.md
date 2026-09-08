# SEVERING EXPERIMENT — READY TO RUN (NOT SUBMITTED)

Status: **built, dry-run green, NOT SUBMITTED.** No `sbatch` was issued. The command below is
written out for a human to run.

Script: `scripts/severing.py`
Cost: 28 proteins x 4 conditions x 2 rows = **224 forward passes**, no training, well under
1 GPU-hour (the ~3 GPU-hour budget is generous; the job asks for 1 hour of walltime).

---

## 1. What this experiment is for

The project's headline is that **77–88% of across-protein dG variance sits in `E_u`**, with
`corr(E_u, wt_err) = 0.865`, and it reads that as *the reference state is wrong*. That
inference does not follow, because `wt_err` is built from a **difference**:

```
wt_err = (E_u - E_f) - dG_true
var(E_u - E_f) = var(E_u) + var(E_f) - 2*cov(E_u, E_f)
```

`cov(E_u, E_f)` **has never been reported.** I grepped the whole tree for `cov_Eu_Ef`,
`corr(E_u,E_f)` and every spelling variant across `*.py`, `*.md`, `*.json` — zero hits. The
CSV the 0.865 came from (`analysis/wt_dg_error/…`) is no longer on disk either.

`E_u` and `E_f` come from the **same frozen network, on the same coordinates, on the same
sequence**. A shared per-protein bias `c_p` — the network recognising the protein and adding
an offset to *both* energies — would inflate `var(E_u)`, inflate `corr(E_u, wt_err)`, and
contribute **exactly zero** to `var(E_u - E_f)`. It reproduces the published triple without
the reference state being at fault at all.

Those two diagnoses call for completely different fixes, and ~150 GPU-hours of reference-state
work is queued behind the first one. This experiment separates them for under an hour of GPU.

## 2. Design

One frozen checkpoint, 28 test proteins, **wild type only**, one batched forward per protein
per condition, logging `E_u` and `E_f` **separately** — the thing the original decomposition
never did.

| id | condition | flags |
|----|-----------|-------|
| C0 | base (in-distribution reference) | — |
| C1 | fold-blind reference state | `unfolded_emb=zero` |
| C2 | analytic ideal-coil geometry | `flory_unfolded`, `coil_b=fixed` |
| C3 | full severing | C1 + C2 + `coil_edges` |

**`f_resid = var(wt_err | C3) / var(wt_err | C0)`**

Also reported for **every** condition: `cov(E_u,E_f)`, `corr(E_u,E_f)`, `var(E_u)`, `var(E_f)`,
`var(dG)`, `corr(E_u,wt_err)`, plus the C0 reproductions of the project's own headline numbers
(`var_Eu/(var_Eu+var_Ef)` and `corr(E_u,wt_err)`) so this run can be checked against the
77–88% / 0.865 claims directly.

### Two design decisions that were not optional

**One batched forward, not two.** `--coil_edges` (U5) rebuilds only the *unfolded* rows' GAT
topology, and `coil_topology.apply_coil_edges` finds those rows via the `n_folded` split of the
batch. Scoring folded and unfolded in two separate single-row passes makes `n_folded == B` in
each, `apply_coil_edges` returns its input unchanged, and **`--coil_edges` silently does
nothing** — C3 would be a mislabelled C1+C2 and nothing would crash. That is this site's
signature failure mode exactly. So the script uses the training code's own convention
(`Megascale-fineTuning/train.py:718`):

```python
model(torch.cat([folded, unfolded], dim=0), n_folded=1)
```

and then **proves at runtime** that the unfolded edge set really changed. Dry run measured
`C0 = 2970 edges -> C3 = 108 = 2(N-1)`: the chain, as U5 specifies. If that check ever fails
the run **aborts** rather than reporting.

**`E_f` is recomputed in every condition.** `scripts/w0_dg.py` hoists `E_f` out of the loop,
assuming unfolded levers cannot touch the folded pass. True for C0/C1/C2 — **false in
principle for C3**, since `--coil_edges` acts inside `PEM.get_edge_index`, which also owns the
edge cache. Recomputing costs 28 extra forwards and removes the assumption. The script asserts
`E_f` is in fact untouched in C1/C2; the dry run confirms `max|dE_f| = 0.000e+00`.

## 3. The OOD objection, and the guard built for it

C1/C2/C3 push a frozen checkpoint out of distribution **by construction** — the chosen
checkpoint was trained with all four levers off. A network fed a zeroed unfolded embedding does
not hand you "the reference state, severed"; it hands you an OOD `E_u` whose variance can
collapse for reasons unrelated to physics. Degenerate outputs have low variance. So
`var(wt_err)` shrinks, `f_resid` looks wonderful, and the number is measuring OOD collapse.

**`f_resid` is therefore not interpretable on its own, and the script does not let it be.**
Per condition it prints mean, std, min, max, p5, p95 and range of `E_u` and `E_f`, plus dG MAE,
RMSE and bias. It raises an explicit `*** OOD WARNING — f_resid IS NOT INTERPRETABLE ***` when
any of these fire:

| guard | default | meaning |
|---|---|---|
| `--mae_blowup` | C3 MAE > 2.0x C0 MAE | checkpoint has left its training distribution |
| `--range_collapse` | C3 range(E_u) or std(E_u) < 0.25x C0 | variance from a **degenerate** output, not physics |
| `--bias_blowup` | \|C3 bias\| > 3x C0 and > 1.0 | no longer on the same energy scale |
| `--fresid_growth` | `f_resid` > 1.25 | severing **added** variance — it cannot legitimately do that, so this is OOD damage |

When the warning fires the verdict is `UNINTERPRETABLE_OOD`, the band interpretation below is
**suppressed**, and the script states in its own output that the result must not be used to
cancel the reference-state work — the honest next step being a *retrained* severed model, not a
frozen-checkpoint ablation. All four guard branches were exercised on synthetic inputs and all
four fire correctly.

## 4. What each outcome would mean

Only valid when the OOD guard **passes**.

- **`f_resid < 0.3`** — severing removed most of the across-protein error variance. The offset
  really is manufactured in the unfolded state; the shared per-protein bias is not the main
  story. **The ~150 GPU-hours is justified: proceed.**
- **`0.3 ≤ f_resid ≤ 0.7`** — both mechanisms are live. The reference state carries a real part
  of the offset, but a substantial remainder survives total severing and must live in the
  shared network bias (`corr(E_u,E_f)` should be materially > 0). Reference-state work will
  help and will **not** be sufficient. **Fund at reduced scope**, budget separately for the
  per-protein bias, and do not promise the offset will close.
- **`f_resid > 0.7`** — the error survived being severed three ways, so it cannot be coming
  from the reference state. The 77–88% / 0.865 headline was reading a **shared per-protein
  network bias that cancels in `E_u - E_f`**. The ~150 GPU-hours would not have fixed the
  offset. **Stop and redirect.**

`corr(E_u, E_f)` at C0 is the direct corroborating read: a large positive value is the
signature of the shared-bias explanation, and it should move consistently with `f_resid`.

## 5. Checkpoint choice

`…_p3_a0_d0_s0_D0_coil_seed42/kf_all_epoch_9.pt` — the all-levers-off cell of the calibration
factorial (`A=0 B=0 C=0 D=0`; in `submit_factorial.sh`, `D=0` sets `DD=""`, so no coil flag).
It is the correct in-distribution baseline for C0: any lever active at training time would
make C0 itself a severed condition. Epoch 9 is the last checkpoint that run wrote (it stopped
short of the 15 epochs; note there is **no `affine.json`** in that directory, so no affine
calibration is applied and the MAE below is raw — fine here, since `f_resid` is a *ratio* of
variances and an affine map would cancel from it).

## 6. THE COMMAND — run from `/home/nissimb/DeepPEF`. **DO NOT let an agent submit this.**

Site rules applied: `WANDB_MODE=disabled`; `--qos normal`; GPU requested by
`--gres=gpu:rtx_6000:1` **never** by partition name (the submit filter reroutes onto a 1080 →
OOM); the `ise-pheno-*` exclude list.

```bash
cd /home/nissimb/DeepPEF && mkdir -p logs results && sbatch --parsable \
  --job-name DeepEF_severing \
  --qos normal \
  --gres=gpu:rtx_6000:1 \
  --cpus-per-task=8 \
  --time 01:00:00 \
  --exclude=ise-pheno-01,ise-pheno-02,ise-pheno-03,ise-pheno-04,ise-pheno-05,ise-pheno-06,ise-pheno-07,ise-pheno-08,ise-pheno-09,ise-pheno-10,ise-pheno-11,ise-pheno-12 \
  --output "logs/severing_%j.out" --error "logs/severing_%j.out" \
  --wrap "module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled; \
python scripts/severing.py \
  --ckpt Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_p3_a0_d0_s0_D0_coil_seed42/kf_all_epoch_9.pt \
  --device cuda \
  --out results/severing.json"
```

Outputs: `results/severing.json` (all conditions, all statistics, thresholds, verdict,
warnings) and `results/severing_per_protein.csv` (per-protein `E_f`, `E_u`, `dG_pred`,
`wt_err` for all four conditions — so `cov(E_u,E_f)` is independently recomputable).

## 7. Pre-submission validation already done (CPU, no GPU, no queue)

```bash
python scripts/severing.py --dry_run      # 8 random proteins, untrained weights, CPU
```

All checks PASS:

- every condition runs; every statistic finite; no NaN
- `var(dG) = var(Eu)+var(Ef)-2cov` identity holds to ~1e-18 in all four conditions
- **every lever provably read**: C1/C2/C3 each move `E_u` off C0 (max|dE_u| 1.9e-3 / 3.4e-3 / 3.0e-3)
- C1 and C2 leave `E_f` bit-identical (`max|dE_f| = 0.000e+00`) — unfolded-only, as designed
- **U5 active in C3**: unfolded GAT edges `2970 -> 108`, and `108 == 2(N-1)` exactly
- `f_resid` computable

`scripts/gate_g4_cpu.py` re-run after all edits: **`baseline … dG=-0.0030 width=1092`**,
unchanged, ALL PASS. `severing.py` is a new read-only script and touches no feature-vector code.
