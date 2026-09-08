
---

# CHECKPOINT 9 — 2026-09-07 — BIOLOGY/CHEMISTRY AUDIT + WAKE-UP

## READ THESE FIRST, IN THIS ORDER, AFTER ANY COMPACT

1. `results/CONTEXT.md` — ALL checkpoint sections. Later ones CORRECT earlier ones.
2. `results/w0.json` + `results/w0_dg.json` — they disagree, and the disagreement IS the finding.
3. `results/calib_per_protein.json` — b_p / a_p per protein.
4. `results/catalogue_vs_bp.json` — 90 correlations; only ONE clears Bonferroni on both tests.
5. `results/FINDINGS.md`, `results/VERIFY_9_ITEMS.md`.
6. `~/auto/autopilot.log` (tail) and `results/state.json`.

## RECONNECT (this cost a whole cycle once)

The password lives ONLY in the shell env as `SLURM_PW`; a compact kills it. Recover it from
`C:/Users/User/Downloads/for claufe.txt`. Then:
```
cd <scratchpad>
PYTHONIOENCODING=utf-8 SLURM_PW='<pw>' python cluster.py exec <<'EOF'
cd /home/nissimb/DeepPEF
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate esm2_env_py38
export WANDB_MODE=disabled
EOF
```
Upload: `MSYS_NO_PATHCONV=1 SLURM_PW=... python put.py <REMOTE_DIR_FIRST> <local_file>`
Download: `MSYS_NO_PATHCONV=1 SLURM_PW=... python get.py <local_dir> <remote_file>`
`/tmp` is PER-LOGIN-NODE — write shared files to `/home/nissimb/`, not `/tmp`.
Connections sometimes time out; just retry.

## THE BIOLOGY/CHEMISTRY AUDIT — measured on the raw catalogue, 100,246 rows

### Dead columns that look alive (this is where the "-0.001 BSA null" came from)

| column | non-null | live sibling |
|---|---|---|
| `Ligands_x` | **4** of 100,246 | `Ligands_y` (75,464) |
| `BSA_Numeric_x` | **4 unique** | `BSA_Numeric_y` (4,867) |
| `BSA_Percentage` | 5 | — |
| `lossg` | constant 0 | — |
| `Global Symmetry` | constant | — |

The old BSA null was VOID TWICE: a dead column scored against the pre-training decoy loss, a
metric already known not to predict ddG. Neither the feature nor the metric could have shown
anything.

### Internal inconsistency
48 rows where the `BSA` string parses nonzero (e.g. "18.76%") while `BSA_Numeric_y` is 0.
`BSA_Numeric_y` is a PERCENTAGE in [0,70], NOT Angstrom^2. 31.3% are exactly 0.

### THE BIOLOGICAL DEFECT — 205,648 het-code occurrences, 8,187 distinct codes

| class | share |
|---|---|
| crystallization additives (SO4, GOL, EDO, NA, CL, PEG) | **27.4%** |
| metal ions | 18.9% |
| **real cofactors/substrates (NAD, ATP, HEM, FAD)** | **6.6%** |
| **MODIFIED RESIDUES (MSE, SEP, TPO)** | **6.3%** |
| unclassified | 43.5% |

12.8% of "ligand-bearing" rows carry ONLY crystallization additives.

**Only 6.6% of het codes are real biological ligands.** SO4 and glycerol come from the
crystallization buffer and do not stabilise the protein in vivo. **MSE is selenomethionine — an
amino acid IN THE CHAIN, used for phasing — so counting it as a bound ligand is both biologically
wrong and a double-count of a residue.**

**A ligand feature built on the raw column would learn crystallography, not biochemistry.**
Any W11 result computed before a het-code classification exists is uninterpretable.

## THE FOURTH MIS-SCORED LEVER: W5 BURIAL

`gate_w5.py` passes 18/18 and every assertion is MECHANICAL — shapes, byte-identity, hydropathy
ordering. **Not one scores performance.** W5's own help text says burial is ZERO in the unfolded
state and the folded-minus-unfolded delta IS the hydrophobic driving force — so it acts on dG/b_p.
But its only scheduled scoring is the S7 factorial whose EFFECT_MIN is defined on POOLED ddG.
**This is the coil case, one lever later.** AUTOPILOT.md has no dG-side acceptance criterion at
all, so a lever that shrinks std(b_p) without moving pooled ddG is recorded as noise.

Evidence the effect is real: `frac_buried_rel_lt_0.25` vs the WT error, mean r = **-0.475**,
sign-consistent **10/10**, significant in 8/10.

## CORRECTION TO CHECKPOINT 8

CHECKPOINT 8 said "the signal is on a_p, NOT b_p". **That was premature.** The b_p side was tested
ONCE, failed a uniform Bonferroni bar over 90 tests, and was then dropped from the replication
sweep — while its sibling WAS replicated and promoted. Replication is the stronger evidence.

**The honest picture: ONE structural axis acts on BOTH calibration channels.**
- exposure -> a_p (slope): +0.624 mean over 10 checkpoints
- burial -> b_p (offset): -0.475 mean over 10 checkpoints
Both show the SAME anchor-weight suppression gradient — independent corroboration on two channels.
**W5 burial is the lever that attacks both.**

## CODE STATUS — all 41 flags audited against live source

15/15 levers are wired end to end (flag -> CFG -> read in train_utils/hydro_net -> gate).
`--slope_weight` genuinely computes (`loss = loss + SLOPE_WEIGHT * slope_loss`) and now RAISES
under `--loss_mode dg` (the slope is defined on ddG spread).

**The one code gap: W9 has NO `--metal_features` flag in train.py**, though
`scripts/metal_features.py` reads `getattr(cfg,'metal_features')`. It can never fire. Decide:
wire it, or formally retire it as superseded by W11 (a metal is one ligand class).

## READY means FOUR things, not one

A lever is READY only if: internally correct AND biologically defensible AND actually read by the
model at training time AND scheduled to be scored on the metric it acts on.
**Correct code judged on the wrong metric is NOT ready.** Three levers have already been caught by
that rule: the coil, BSA, and now W5.

## STILL NEVER RUN (all need GPU from the cap of 5)

- the dG training arm (`--loss_mode dg --flory_unfolded --coil_b fixed`)
- `--slope_weight` (factor C) — now the best-motivated lever we have
- the severing experiment (built, dry-run green, `results/SEVERING_READY.md`)
