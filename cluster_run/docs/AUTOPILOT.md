# AUTOPILOT — state machine, decision rules, failure protocol

**Companion to `MASTER_PLAN_CODE.md`.** That document says *what* to build and *why*. This one says
*when to run it*, *how to decide what comes next*, and *what to do when something breaks*.

**Default posture: continue autonomously.** Every decision node below has a rule. If a rule fires,
follow it and record the decision — do not stop to ask. Only the four items in §7 halt.

**Constants used throughout**

```
SIGMA_POOLED = 0.0072      # measured, 5 seeds
SIGMA_PP     = 0.0037
EFFECT_MIN   = 2 * SIGMA_POOLED / sqrt(3)   # = 0.0083, resolvable at 3 seeds
REF_POOLED   = 0.591       # our calib_ctrl reproduction
REF_PP       = 0.731
REF_STD_B    = 0.223
CEILING      = 0.711       # offset-removed, measured over 12 checkpoints
RUN_HOURS    = 6           # 15 epochs, RTX 6000 class
MAX_CONCURRENT = 8
```

---

# 1. State machine

Persist state to `state.json` after **every** transition so a context reset resumes exactly.

```json
{"state":"S3_FACTORIAL","entered":"2026-09-06T10:00Z","cells_done":11,"cells_total":16,
 "decisions":[{"node":"D0","choice":"coil_kept","evidence":"var(E_u) noemb -18%"}],
 "blocked":[],"retries":{"job_18021044":1}}
```

```
S0  PREFLIGHT ──► S1 W0_ABLATION ──► D0 ──► S2 FIXES ──► S3 FACTORIAL ──► D1
                                                                           │
    ┌──────────────────────────────────────────────────────────────────────┤
    ▼                                                                      ▼
S4  REFINE ──► S5 BUILD_INFO ──► S6 INFO_GATES ──► D2 ──► S7 INFO_FACTORIAL
                                                                           │
                                                          ┌────────────────┤
                                                          ▼                ▼
                                              S8 FINAL_MODEL ──► S9 REPORT ──► DONE
```

## S0 — PREFLIGHT (no GPU, ~1 h)

1. `git fetch shaharec && git checkout -b auto-run origin/current-vers`
2. Assert the four levers are reachable:
   `grep -c "wt_anchor_weight\|designed_weight\|slope_weight\|flory_unfolded" train.py` ≥ 4.
   **If `flory_unfolded` is absent, apply FIX 1 from `FIXES.md` first.**
3. `bash cluster_run/scripts/00_env_check.sh` → all PASS
4. `bash cluster_run/scripts/01_data_check.sh` → `DATA: OK`, 368 / 28
5. Validate `calib_diag.py` on a committed eval CSV. **Record the offset-removed value.**
   Expected 0.70–0.72. **If it lands outside 0.65–0.78, halt (H1).** Inside that band but outside
   0.70–0.72, record and continue — the wider band is the honest tolerance.

**Exit:** all pass → S1.

## S1 — W0 channel ablation (1 h, no training)

Run `scripts/w0_unfolded_channel_ablation.py` on an existing checkpoint, 28 test proteins,
WT variant only. Four conditions on the **unfolded pass only**: `base`, `noemb`, `noOH`, `coil`.

Report per condition: `var(E_u)` across proteins, `corr(E_u, wt_err)`, `corr(E_u, length)`.

**Exit:** results written to `results/w0.json` → D0.

## D0 — decision: what is factor D?

Let `r_x = var(E_u | condition x) / var(E_u | base)`.

| Condition | Rule | Action |
|---|---|---|
| `r_noemb < 0.5` | ProtT5 is the offset channel | **Factor D := `--unfolded_emb`, levels {full, zero}.** Coil demoted to a 2-cell side arm |
| `r_coil < 0.5` and `r_noemb ≥ 0.5` | Geometry is the channel | **Factor D := `--flory_unfolded` as planned** |
| `r_noOH < 0.5` and both others ≥ 0.5 | Composition + extensive sum | **Factor D := `--flory_unfolded`**, and add `--dg_length_norm` as a 2-cell side arm |
| all ≥ 0.5 | Variance is in learned weights, not inputs | **Factor D := `--flory_unfolded`** (weakest evidence), and log `H4` for the write-up |
| two conditions both < 0.5 | Channels are entangled | Pick the smaller `r`; log the other as a side arm |

**Record the choice and its evidence in `state.json`. This is the only place factor D is decided.**

## S2 — FIXES (no GPU, ~2 h)

Apply in this order, each with its G1/G2 gate from `MASTER_PLAN_CODE.md` §IV:

1. **U5** coil-consistent edge topology (`--coil_edges`) — without it the coil is half-applied
2. **U6** per-half `ca_coords` — without it Phase 7 and the coil cancel
3. **U2** `--unfolded_emb {full,zero,mean}` — needed if D0 chose it
4. **U10** make span-1 bidirectional in the control arm

**Rule:** a fix whose G1 (`torch.equal` when off) fails is **reverted**, not debugged in place.
Log it and continue with the remaining fixes. **G1 failing means the flag is not inert, and an
inert default is what protects every baseline number.**

**Exit:** all applied fixes pass G1 and G2 → S3.

## S3 — CALIBRATION FACTORIAL

2⁴ = 16 cells × 3 seeds = **48 runs**. Factors: A `--wt_anchor_weight {0, 1.0}`,
B `--designed_weight {1, 3}`, C `--slope_weight {0, w*}`, D as decided at D0.

`w*` comes from the slope pilot already running. **If the pilot is not finished, take
`w* = 1.0`** and log the assumption.

**Submission:** waves of 8. Wave 1 = the 8 cells at seed 42. **Verify wave 1 before wave 2** — if
≥ 2 of 8 fail identically, halt (H2): that is a systematic error, not bad luck.

**Run tag:** `p3_a{A}_d{B}_s{C}_D{D}_seed{S}`.

**Per run:** attach `monitor.py`; epoch-2 reproduction check; `calib_diag.py` on the final eval.

**Exit:** ≥ 44 of 48 runs complete → D1. Fewer → H2.

## D1 — decision: which cell wins, and does anything work?

Fit main effects and all six two-way interactions on pooled ΔΔG. An effect is **real** if
`|effect| > EFFECT_MIN` (0.0083).

| Result | Action |
|---|---|
| ≥ 1 real main effect | `BEST` := the cell with the highest mean pooled. → S4 |
| No real main effect but a real interaction | `BEST` := the best interaction cell. → S4, and log that main effects were null |
| Nothing real | **Do not stop.** `BEST` := `calib_ctrl`. Skip S4, go straight to S5. Log `F1`: *calibration levers do not move pooled beyond noise* — a genuine negative result and a thesis finding |
| `BEST` pooled > `CEILING` + 0.02 | Suspicious. Re-run `BEST` at 2 new seeds before accepting. If it holds, accept and log |

**Also always report:** `std(b)` and the slope distribution per cell. **A cell that raises pooled
while collapsing slope below 0.3 median is not selected**, even if its pooled is highest — that is
the anchor's known failure mode and selecting it would defeat the purpose.

## S4 — REFINEMENT (~20 runs)

Sweep the factors that showed a real effect at finer levels, 5 seeds each:
anchor {0.3, 1.0, 3.0}, designed {3, 5}, slope around `w*`. Plus the U3/U4 coil sub-arms (2×2,
1 seed) if D0 kept the coil.

**Deliverable: the trade-off curve** — `std(b)`, slope distribution and PP as functions of anchor
weight. That curve does not exist anywhere and is a result in itself.

**Exit:** → S5.

## S5 — BUILD INFO LEVERS (no GPU)

Implement W5 burial, W6 descriptors, W7 edges, W8 pLDDT from `MASTER_PLAN_CODE.md` §III.
**Build all four before running any.** Order: W5, W8, W6, W7.

**Exit:** all four compile and pass G1 → S6.

## S6 — INFO GATES (no GPU, ~2 h)

Run every G2 assertion. **A lever that fails G2 does not enter the factorial** — it is logged as
`not-verified` and dropped. Specifically:

- W5: burial differs folded/unfolded; `|r|` vs Shrake-Rupley SASA > 0.7; no length confound
- W6: L→{I,V,M}, D→{E,N}, F→{Y,W} in descriptor space
- W7: `k(2Å) > k(8Å) > k(15Å)` strictly
- W7+coil: unfolded edge distances differ from folded

**Exit:** → D2.

## D2 — how many info levers enter the factorial?

| Levers passing G2 | Design |
|---|---|
| 3 | 2³ × 3 seeds = 24 runs |
| 2 | 2² × 3 = 12 runs |
| 1 | 2 arms × 5 seeds = 10 runs |
| 0 | Skip S7. Log `F2` and go to S8 with `BEST` |

W8 pLDDT runs as a separate 2-arm × 3-seed comparison regardless — it is cheap and its
tercile analysis is worth it independently.

## S7 — INFORMATION FACTORIAL

All cells on top of `BEST`, never the bare baseline. Same submission and monitoring rules as S3.

**Exit:** → S8.

## S8 — FINAL MODEL

Assemble from every component that cleared `EFFECT_MIN`. Run 5 seeds. **Average the five
predictions for the seed ensemble** — free, since the runs exist, and typically worth 0.02–0.03.

Full evaluation: pooled, PP, absolute WT ΔG, `std(b)`, slope distribution, designed-fold
breakdown, pLDDT-tercile breakdown.

## S9 — REPORT

Emit `results/FINAL.md` with: every cell's three-qualifier metric line, the factorial effect table
with interactions, the trade-off curve, the mechanism check for whichever D was chosen, all
negative results, and the full decision log from `state.json`.

**Then halt and hand over.** Writing is Nissim's.

---

# 2. Failure protocol

**Classify first, then act. Never retry blindly.**

| Symptom | Class | Action | Max retries |
|---|---|---|---|
| `CUDA out of memory` | resource | Resubmit on a ≥48 GB node (`cs-6000`, `ee-l40s`). Still failing → halve `mini_batch` **and mark the run `regime-changed`** so it is excluded from any comparison against 64 | 2 |
| Job `PENDING` > 12 h | queue | Resubmit to a different partition. Log the node class change | 2 |
| `s/it` > 2× the wave median | contention | `scancel`, resubmit excluding that node. **`ise-pheno` is 3.5× slower — exclude it permanently** | 3 |
| No log line for 60 min | hang | `scancel`, resume from the newest `epoch_*.pt` | 2 |
| NaN / Inf in loss | numerical | **Do not retry as-is.** Resubmit once with `--ranking_weight 0`. Still NaN → drop the cell, log `F3` | 1 |
| Training OK, eval fails | pipeline | Re-run eval only. Twice failed → check the checkpoint path is derived, not hardcoded | 2 |
| `monitor.py` fires `STOP: M4` (val PP < 0.30 at epoch 2) | model | **Not a fault.** The cell is bad. Record the result, do not retry | 0 |
| `monitor.py` fires `STOP: M5` (slope collapse) | model | Record. **This is data, not an error** — it is the failure mode under study | 0 |
| Same failure on ≥ 2 of 8 in a wave | systematic | **Halt H2** | — |
| Disk > 85 % | resource | Delete `epoch_0` … `epoch_7` for runs whose best epoch is known. Still full → halt H3 | — |
| `git push` refused | config | **Never force.** Check `git remote -v`. **If the target resolves to `shaharec`, abort immediately and log** | 0 |

**Retry accounting.** Increment `retries[job_id]` in `state.json`. On exceeding the max, mark the
cell `failed`, continue with the rest, and report it in S9. **A missing cell is a footnote; a
fabricated one is misconduct.**

**Never retry more than twice with an identical command.** The third attempt must change something
and say what changed.

---

# 3. Scheduling rules

- Max 8 concurrent jobs. Waves of 8; verify wave *n* before submitting *n+1*.
- **All seeds of one cell on the same node class.** Record the class with every result. Mixing
  classes within a cell is a confound.
- Prefer `cs-4090`, `cs-6000`, `ise-6000`, `ee-l40s`. **Never `ise-pheno`, `cs-2080`, `cs-1080`.**
- Chain eval as `--dependency afterok:<jobid>`.
- Poll every 10 min with `squeue -u $USER -o "%i %T %M %N"`. **Never `sleep` longer than the
  poll interval.**
- Every command's output through `head`/`tail`/`grep`. **Never dump a whole log or directory** —
  context is the scarce resource.

---

# 4. Bookkeeping, non-negotiable

**After every run**, append one line to `results/RESULTS.tsv`:

```
run_tag  state  node_class  seed  epochs  best_epoch  pooled  pp  abs_wt_dg  std_b  slope_med  slope_min  designed_pooled  wall_h  notes
```

**After every decision node**, append to `state.json.decisions`: node id, choice, the numeric
evidence, timestamp.

**Every reported number carries three qualifiers:** pooled or per-protein · ΔG or ΔΔG · which split
and which selection. Numbers come only from `run_calib_eval.sh` → `score_runs.py`.

**Every run over an hour** gets the six-line run card from `RUNBOOK.md` §1 pasted into its report.

---

# 5. Context-reset protocol

On resume, before any command:

1. Read `state.json` — current state, decisions, blocked items, retries
2. Read `cluster_run/docs/RUNBOOK.md` — the preflight discipline
3. Read `results/RESULTS.tsv` — what is already measured
4. `squeue -u $USER` — what is still running
5. Resume from `state.state`. **Do not re-run a completed state.**

**Never re-derive a decision already in `decisions`.** They were made with evidence that is
recorded; re-deciding without that evidence is how a stale premise gets re-adopted.

---

# 6. Self-audit, every 6 hours

Automatic, no prompting:

1. `RESULTS.tsv` row count equals completed jobs
2. No two rows share a `run_tag`
3. Every `pooled` is in [0, 1]; every `std_b` > 0
4. No cell's seeds span more than one node class
5. Disk under 85 %
6. `git remote -v` unchanged and `origin` is not `shaharec`
7. No run exceeded 2× the median wall-clock without being logged

**Any check failing → log it and apply the matching failure-protocol row. Two failing at once →
halt H2.**

---

# 7. The only four halts

Everything else continues autonomously.

**H1 — `calib_diag.py` outside 0.65–0.78 on a known CSV.** The measurement instrument is wrong.
Nothing downstream can be trusted.

**H2 — a systematic failure**: ≥ 2 of 8 in a wave failing identically, or two self-audit checks
failing together.

**H3 — resources exhausted**: disk full after cleanup, or the QOS revoked.

**H4 — a result contradicts a recorded fact by more than 3σ.** Example: reproducing `calib_ctrl`
at 0.75, or `std(b)` at 0.05. Do not build on it and do not explain it away. Report and halt.

**On a halt:** write `results/HALT.md` with the state, the evidence, what was tried, and the three
most likely causes. **Do not attempt a workaround.**

---

# 8. Explicitly out of scope for the agent

**Writing the thesis.** The central argument — that the affine oracle is not a valid ceiling, why
it fails, and what replaces it — is Nissim's contribution. Produce the tables, figures and numbers;
do not draft the argument.

**Changing the reference configuration.** `calib_ctrl` is fixed. If it will not reproduce, that is
H4.

**Adding a lever not in `MASTER_PLAN_CODE.md`.** The registry there lists fourteen directions
already closed on evidence. Proposing one of them means the registry was not read.

**Anything touching `shaharec/DeepPEF`** beyond `fetch` and read.
