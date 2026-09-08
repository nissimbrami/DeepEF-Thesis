
---

# CHECKPOINT 4 — 2026-09-06 ~22:00Z — POST-COMPACT RECONNECT + THE SCIENTIFIC GAPS

**Read this section FIRST. It supersedes CHECKPOINT 3's "what we are waiting for".**

## 0. How to reconnect (this cost a whole reconnect cycle once — do not repeat it)

The cluster password lives ONLY in the shell env as `SLURM_PW`. A compact kills the shell and the
password with it, and every `ssh` then fails with `Permission denied`. It is recoverable from
`C:/Users/User/Downloads/for claufe.txt` (an old chat transcript). Run everything as:

```
cd <scratchpad>
SLURM_PW='<pw>' python cluster.py exec <<'EOF'
cd /home/nissimb/DeepPEF
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate esm2_env_py38
export WANDB_MODE=disabled
<commands>
EOF
```
Upload: `MSYS_NO_PATHCONV=1 SLURM_PW='<pw>' python put.py <remote_dir> <local_file>`
(NOTE the argument order: remote dir FIRST. And `MSYS_NO_PATHCONV=1` or Git-Bash mangles the path.)

**The real fix, still not done:** an SSH key. It survives compaction and also unblocks the git push.

## 1. State verified by running it, not by trusting a report

- **5 trainings RUNNING, 104 queued.** Nothing was cancelled. The 48-run factorial is intact.
- **0 factorial eval CSVs.** A finished training is NOT a result. ~5-7 days remain.
- **autopilot ALIVE** — it had died with the login node (`rc=143`). Restarted with
  `setsid nohup` (NOT tmux) so it now survives a login-node drop.
- **`python scripts/gate_g4_cpu.py` prints `baseline ... dG=-0.0030 width=1092`.**
  This is THE guard: it means none of the new code has moved the feature vector, so the 48 runs
  in flight are still valid. Re-run it after ANY feature-vector change. If it moves, revert.
- **All 8 gate suites green**, including gate_u3u4 at **32/32** (see §2).

## 2. gate_u3u4 — fixed, and the diagnosis is worth keeping

It failed 2/32 for a long time and was written off as "known-false". It was neither false nor a
float-precision issue — my first two fixes (absolute tolerance, then relative tolerance) BOTH failed,
which is what forced an actual measurement:

- off-diagonal agreement: **1.2e-06** (perfect)
- the four DIAGONAL channels (0/5/10/15, the self-pairs): **1.38e-03**

Cause: `get_dist_matrix` flattens to (N*4,3) and runs ONE `cdist`; a self-distance goes through
`sqrt(x^2+x^2-2x.x)`, a catastrophic cancellation that returns ~1e-3 instead of exactly 0. The gate
was comparing that artifact and calling it a layout error. **Channel 5 IS CA-CA** (verified: it equals
`torch.cdist(x[:,1,:],x[:,1,:])` off-diagonal, and 1*4+1 = 5). Fix: compare off-diagonal for the
layout claim; assert the self-pair diagonal is near zero separately.

**Lesson to carry: do not guess at a tolerance twice. Measure the error's STRUCTURE first.**

## 3. What three workflows established (33 agents, all complete)

**Adversarial verification of the 9 worker items** → `results/VERIFY_9_ITEMS.md`.
- NOT-READY: **mordred, u5u6, w7edge, monitor** (`monitor` selects a GPU BY PARTITION NAME, which
  violates the site rule that has already cost us runs).
- READY-WITH-FIXES: w6wire, u3u4, w9, e2e, catalogue.
- The most dangerous find: **`PEMGraphTransformer` silently ignores the descriptor block.** Its
  slices stay in-bounds with a wider vector, so the run COMPLETES and reports a baseline number as a
  descriptor result. This is the project's signature failure mode — a silent wrong number.

**U5/U6 + W7-edge applied** to the cluster, G4 re-verified at width=1092.

## 4. THE SCIENTIFIC GAPS — this is the actual remaining work

**a. Mordred has NEVER been run.** The descriptor CSV on disk has the header
`"synthetic replica for verification"`. **Every W6 number so far was scored on a fake matrix.**

**b. The histidine SMILES is WRONG** in PHASES_5_9_WITH_CODE.md:230 —
`c1cc(nc1)...` is a ring with FOUR carbons and ONE nitrogen (C7H10N2O2). L-histidine is C6H9N3O2,
imidazol-4-yl, TWO ring nitrogens. Correct: `c1c(nc[nH]1)C[C@@H](C(=O)O)N`. Mordred would compute
~654 descriptors for the WRONG MOLECULE, and the existing gate cannot see it because it only tests
L, D and F. Fix requires a structural self-check (formula + stereocentres) over all 20.

**c. Ofir's contribution was discarded, precisely.** `aa_descriptors.py` hard-codes
`ORDER = list('ACDEFGHIKLMNPQRSTVWY')` and RAISES on anything else; lookup is `one_hot @ [20,K]`, so
**the alphabet is closed at the tensor level.** His claim is the opposite: the physicochemical space
is continuous, so a model trained on canonical residues predicts NON-CANONICAL effects. A 20-row
closed table throws away the entire point.
  - The sharpest objection to answer: the vector ALREADY has 1024-dim ProtT5 + 20-dim one-hot. Can a
    16-dim descriptor block add anything they do not carry? Test it: regress descriptors on ProtT5.
  - MegaScale has ~0 non-standard residues, so the generalisation claim needs a PROXY: hold out a
    canonical residue type (e.g. W) entirely and predict its mutations from descriptor space alone.

**d. The 100k catalogue was never exploited.** 100,246 x 21. It was correlated only against the
PRE-TRAINING DECOY LOSS (r = -0.001) and declared dead. That is the wrong metric — the same error as
the coil. The right question is whether protein-level properties (BSA, oligomeric state, ligands,
hydrophobicity, disulfides) predict **b_p** on the 28 test proteins. Caveat that must be stated:
n=28 means |r| < ~0.37 is indistinguishable from zero at p=0.05, over ~21 columns.

**e. Ligands as graph nodes — never built.** Only metal ions were considered. Any bound molecule
(ATP, NAD, cofactor, substrate) stabilises a protein by kcal/mol. Architecturally it is a node with
its own edges. HONEST CAVEAT: MegaScale is small in-vitro domains, mostly ligand-free — this is a
GENERALISATION lever, not a benchmark-number lever, and must be presented that way.

## 5. THE BIGGEST OMISSION: no dG arm has ever been submitted

Verified against `sacct`: every queued job is the ddG factorial. The coil-on-dG finding
(MAE 4.965 -> 3.952 with `--coil_b fixed`) is our strongest result on `b_p` and **zero runs test it.**
It needs GPU from the same 8-card cap the factorial is using, so it needs the user's approval.

## 6. THE RULE THAT KEEPS PAYING

**Score a lever on the metric it acts on.** ddG cancels everything identical between WT and mutant —
chain length, backbone positions, and therefore every purely geometric or whole-protein property.
It has now caught TWO directions that were nearly discarded on invalid measurements: the coil
(r=1.175 "harmful" on ddG; our best b_p lever on dG) and BSA (scored against a pre-training loss that
cannot predict ddG either way). Audit every new lever against this before spending a GPU-hour.
