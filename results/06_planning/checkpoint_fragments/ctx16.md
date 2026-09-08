
---

# CHECKPOINT 16 — 2026-09-08 — EVERYTHING BROKE OPEN, AND W6 IS LARGELY REDUNDANT

## The blockage cleared

| | before | now |
|---|---|---|
| factorial CSVs | **0/48** | **10/48** |
| GPUs running | 5 | **13** (5 public + 8 golden) |
| labmate holding golden | 8/8 | **0** |
| autopilot | dead behind a healthy log | **SLURM job 21107688, 7-day** |

The scoring fix worked: `cpu_evals` scored 8 cells on the CPU partition at zero GPU cost, and the
autopilot now logs `S3: 10/48 cells scored`.

**The autopilot is now a SLURM job**, not a login-node process. It had died twice more with
`rc=143` (SIGTERM when its login node churned). SLURM does not restart it, but the job itself
outlives login nodes entirely — which was the actual failure mode.

**Golden lane running 8/8:** gld_dg_coil, gld_slope, gld_w5_dg, gld_w7edge, gld_u10bidir,
gld_w7span4, gld_loroW_onehot, gld_loroW_desc. **Every lever that had never been tested is now on a
card.**

## THE EMBEDDING RESULT — this settles W6

`scripts/emb_probe.py`, 19 proteins, 1,056 residues, **leave-one-residue-type-out** (fit on 19
types, predict the held-out 20th — a row-random split is contaminated because every occurrence of a
residue shares one descriptor vector):

| property | held-out R^2 |
|---|---|
| **hydropathy** | **+0.704** |
| charge | +0.511 |
| volume | +0.421 |
| identity (row-random) | accuracy **1.000** vs chance 0.050 |

**ProtT5 predicts the hydropathy of a residue type it has NEVER seen, at R^2 = 0.70.**

This is the decisive fact for W6. A physicochemical descriptor block is a deterministic function of
residue identity, and:

1. **Identity is perfectly recoverable** from the embedding (accuracy 1.000). So the network can
   already reconstruct one-hot from ProtT5 — anything a per-residue-type table encodes is already
   spanned.
2. **The chemistry itself extrapolates to unseen residue types** (hydropathy R^2 = 0.704). This is
   precisely the generalisation Ofir attributes to descriptors — and ProtT5 already does it.

**W6 cannot be justified as adding information.** Its only defensible value is (a) regularisation —
sharing statistical strength across chemically similar residues — or (b) coverage of residues
absent from ProtT5's training data. And Ofir CONCEDES exactly this: his argument for descriptors is
coverage of non-canonical residues, not added signal. Our 28 test proteins are all canonical, so
that gap does not exist here.

**This should be stated in the thesis as a measured result, not an assumption.** It also predicts
the outcome of the LORO experiment now running: holding out W, the descriptor arm should NOT beat
the one-hot arm by much, because ProtT5 already carries W's chemistry. That prediction is on record
BEFORE the runs finish.

Caveat: 19 proteins, one variant each, a linear ridge probe. It bounds what a LINEAR reader can
extract; a nonlinear network might differ, though that cuts against descriptors too.

## Supporting facts from the same inspection

- **ProtT5 is CONTEXTUAL**: the same residue type at two positions has cosine similarity 0.13-0.18,
  nearly orthogonal. It encodes position and neighbourhood, not identity — yet identity is still
  perfectly decodable, so the information is present but distributed.
- **ProtT5 is recomputed PER VARIANT**: a single point mutation changes ALL 52 of 52 positions, so
  the mutation signal is global, and the embedding does NOT cancel in ddG.
- Tensors are per-variant: embeddings `[128, 52, 1024]`, one-hot `[1389, 52, 21]`. The 21st
  one-hot column is dropped at `train.py:317` before the graph is built, so every `[20, K]`
  descriptor table is correctly sized.
- The GCN branch reads `x[:, :32]` only, so **any block inserted at offset 48 is invisible to it**
  and reaches the GAT branch alone.
