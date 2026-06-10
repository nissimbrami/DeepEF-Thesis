# Experiment: GAT Distance Cutoff Edges

**Date:** 2026-03-27
**Branch:** main
**Status:** Ready to run

## Motivation

The PEM model's GAT branch used a **fully-connected graph** — every residue attended to every other residue. This has two problems:

1. **O(N^2) edges** — for a 500-residue protein, that's ~250K edges per sample. Memory and compute scale quadratically.
2. **Biophysically wrong** — residues >10-15A apart have negligible direct interactions. Attending to distant residues adds noise without signal.

Standard protein GNNs (ProteinMPNN, GVP-GNN, IPA in AlphaFold2) use **distance-cutoff** or **k-nearest neighbor** graphs with 10-30 neighbors per residue.

## Change Summary

### What changed

| File | Change |
|------|--------|
| `model/model_cfg.py` | Added `gat_cutoff = 10.0` (Angstroms) |
| `model/hydro_net.py` | `PEM.__init__` accepts `gat_cutoff`; `get_edge_index()` builds distance-based GAT edges using `torch.cdist` on CA atom coordinates when `ca_coords` is provided |
| `train-2_5_light_att.py` | `get_noised_proteins()` extracts CA coordinates (atom index 1) before `get_graph()` consumes 3D coords; passes `ca_coords` tensor through training/validation/DSM loops |

### How it works

1. CA coordinates (atom index 1 = C-alpha) are extracted from backbone coords **before** `get_graph()` converts them into distance features
2. A `ca_coords` tensor of shape `[batch, N, 3]` is constructed matching the 9-element batch (Xjf, Xju, Xd, Xcd, Xdu, Xcy1-4)
3. In `PEM.get_edge_index()`, pairwise CA-CA distances are computed via `torch.cdist`
4. Edges are created only between residues with CA-CA distance < `gat_cutoff` (default 10A)
5. GCN edges (sequential i->i+1) are unchanged

### Backward compatibility

- `ca_coords=None` (default) falls back to fully-connected GAT — all other scripts (inference, hpar_tuning, evaluate_train) work unchanged
- `gat_cutoff=None` also forces fully-connected, regardless of ca_coords

## Expected Impact

### Edge reduction

Tested with realistic protein-like coordinates (100 residues):

| Cutoff | Edges | % of Full | Avg Neighbors |
|--------|-------|-----------|---------------|
| 5.0A   | 680   | 3.4%      | 3.4           |
| 8.0A   | 960   | 4.8%      | 4.8           |
| **10.0A** | **1,200** | **6.1%** | **6.0** |
| 12.0A  | 2,480 | 12.5%     | 12.4          |
| 15.0A  | 3,648 | 18.4%     | 18.2          |
| 20.0A  | 4,820 | 24.3%     | 24.1          |

For real proteins the neighbor count will be higher due to 3D packing (globular fold), typically 20-30 neighbors at 10A.

### What to expect

- **Faster training** — fewer GAT edges means less message-passing compute per forward/backward pass
- **Better generalization** — the model can no longer rely on attending to irrelevant distant residues; forces it to learn local structural patterns
- **Lower memory** — enables training on longer proteins (currently limited to `seq_len=600`)
- **Risk:** If critical long-range interactions exist that GAT was capturing, accuracy could drop. Mitigate by trying 12A or 15A cutoff.

## Experiment Plan

### Baseline

Run with `gat_cutoff = None` (fully connected, current behavior) to establish baseline metrics:

```bash
python train-2_5_light_att.py --debug
```

### Variants to test

| Run | `gat_cutoff` | Hypothesis |
|-----|-------------|------------|
| A   | `None`      | Baseline (fully connected) |
| B   | `10.0`      | Standard cutoff — fewer edges, better inductive bias |
| C   | `15.0`      | Larger cutoff — more context, moderate edge reduction |
| D   | `8.0`       | Aggressive cutoff — maximum speedup, risk missing contacts |

To change cutoff without code edits, modify `CFG.gat_cutoff` in `model/model_cfg.py`.

### Metrics to compare

- `val_rank_Ejf_lt_Exd` — native vs sequence-decoy discrimination (primary metric)
- `val_rank_Ejf_lt_Ecd` — native vs structure-decoy discrimination
- `val_rank_Ejf_lowest` — native has lowest energy among all variants
- `val_gap_Exd_Ejf` — energy gap between decoy and native (larger = better)
- Training wall-clock time per epoch
- Peak GPU memory usage

### How to run

```bash
# Debug mode (quick test)
python train-2_5_light_att.py --debug

# Full training
python train-2_5_light_att.py

# To test different cutoffs, edit model/model_cfg.py:
#   gat_cutoff = 10.0  (or 8.0, 12.0, 15.0, None)
```

## Literature References

- **ProteinMPNN** (Dauparas et al. 2022): 10A CA-CA cutoff, k=48 neighbors
- **GVP-GNN** (Jing et al. 2021): k-NN with k=30, ~10A effective radius
- **AlphaFold2 IPA** (Jumper et al. 2021): uses distance-based attention weighting
- **SchNet** (Schutt et al. 2018): continuous-filter convolution with distance cutoff
