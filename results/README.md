# DeepEF — Results Index

M.Sc. thesis · Nissim Brami · supervised by Prof. Chen Keasar · Ben-Gurion University

**Problem:** DeepEF predicts protein stability as `dG = E_unfolded − E_folded`, and `ddG` as the
difference between mutant and wild type. It **ranks mutations well within a protein
(ddG PCC 0.798) and is miscalibrated between proteins (pooled 0.59)**. Closing that gap is the
thesis.

---

## Read these first

| file | what it is |
|---|---|
| **[`01_start_here/MASTER.md`](01_start_here/MASTER.md)** | **The complete record.** 13 parts, written to be self-contained. Start here. |
| [`01_start_here/FINDINGS.md`](01_start_here/FINDINGS.md) | The findings of record, 15 parts |
| [`01_start_here/STATUS_FULL.md`](01_start_here/STATUS_FULL.md) | The scoreboard and the task lists |
| [`01_start_here/TASKS.md`](01_start_here/TASKS.md) | Open work, one line each |
| [`01_start_here/CONTEXT.md`](01_start_here/CONTEXT.md) | 28 dated checkpoints, chronological |

---

## The headline numbers

**Scoring basis: 27 test proteins (2K5H excluded — see §11.1 of MASTER), ddG metric, per-protein fits.**

| | value |
|---|---|
| per-protein ddG PCC | **0.798** |
| pooled ddG PCC (control) | **0.5635** |
| **best pooled achieved** | **0.6382** (+0.075) |
| offset-removal oracle | **0.7534** — the ceiling if `b_p` were solved |
| best `a_p` achieved | **0.740** (from 0.496) |
| **seed noise band** | **±0.060** — the bar every lever must clear |

**Only two things raised the score:** `--slope_weight 1.0` and requiring factor D0.
**Eleven hypotheses were tested and rejected.**

---

## Directory map

### `01_start_here/` — orientation
The five documents above. Everything else supports them.

### `02_findings/` — the science, one file per question
| file | question it answers |
|---|---|
| `THESIS_RESULTS.md` | the results chapter as it would be written, with figures |
| `SLOPE_ARM.md`, `SLOPE_OBJECTIVE.md`, `SLOPE_ORIGIN.md` | the only proven lever: `a_p = r·s`, why it works, and its ceiling |
| `DG_ARM.md` | the dG/coil arm — **rejected**, and why the motivating result was misread |
| `W5_ON_DG.md`, `INFO_LEVERS.md` | burial on dG; the four information levers (all inside the noise band) |
| `LORO_RESULT.md` | the descriptor test — **the arm collapsed technically and never ran** |
| `SEVERING_RESULT.md` | `corr(E_u, E_f) = 0.543` — 45% of the two energies' variance is shared |
| `PER_MUTATION.md`, `SIDECHAIN_DESIGN.md` | where the ranking error lives: burying hydrophobics |
| `TWO_PROTEINS.md` | two proteins carry 78% of the oracle gain |
| `IDENTIFIABILITY.md`, `OFFSET_CORRECTOR.md` | `b_p` is learnable (ICC 0.898) but not predictable from features |
| `LENGTH_CONFOUND.md` | the exposure→`a_p` finding survives partialling out length |
| `FACTORIAL_ANALYSIS.md` | factor D separates at 14.4σ; A is not estimable |

### `03_levers/` — design specs
How each lever enters the 1092-dim feature vector, which layers grow, what its gate asserts.
Includes the ready-to-run specs for arms not yet launched.

### `04_data_integrity/` — what the data actually contains
`INTEGRITY_AUDIT.md` (the 2K5H bug and the audit that found nothing else),
the 100k catalogue analysis, the MegaScale raw-data inventory,
and the reconciliation with Ofir Ezrielev's thesis.

### `05_infrastructure/` — cluster and tooling
GPU lanes and quotas, the gate suite verification, compute optimisation.

### `06_planning/` — historical
Earlier plans and the per-checkpoint fragments that were concatenated into `CONTEXT.md`.

### `07_agent_reports/` — raw
`AGENT_REPORT.md` (286 KB): all 28 background-agent results, unedited, kept as provenance.

### `figures/` — 9 figures with `MANIFEST.json`

---

## The two rules that organise the work

**1. Score a lever on the metric it acts on.** `ddG` cancels anything identical between wild type
and mutant, so a reference-state or whole-protein lever must be judged on `dG` or `b_p`. `a_p` is a
within-protein slope and does not cancel. **This rule caught five levers**, including our own
model-selection metric, which returns pooled `dG` PCC — scale-invariant, and therefore blind to
exactly what the slope lever changes.

**2. Always compute the degenerate baseline.** `--dg_length_norm` looked like a 31% improvement and
was the model predicting nothing. A correlation quoted without its *n* is not a result.

---

## Reproducing a number

```bash
# every score in this repo comes from the eval CSVs
ls eval_results/abl_*.csv

# the gate that protects everything in flight — must print dG=-0.0030 width=1092
python scripts/gate_g4_cpu.py

# per-protein calibration (b_p, a_p) from any eval CSV
python scripts/calc_bp.py

# reference-row integrity (this is what caught the 2K5H bug)
python scripts/gate_refrow.py
```

**Two formats of checkpoint exist.** Wrapper checkpoints carry model + optimizer + scheduler and
resume faithfully; bare `state_dict` checkpoints do not, and `--resume` **refuses** them rather than
silently producing a different trajectory.
