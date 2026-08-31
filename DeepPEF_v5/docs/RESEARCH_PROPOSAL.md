# DeepEF Research Proposal — Physics-Grounded Levers (A–F)

This document restates the thesis proposal's research direction as six concrete,
physics-grounded **levers** and maps each to its implementation flag in `DeepPEF_v5`.
Every lever is **flag-gated** and **ablatable**; the default value of every flag reproduces
the proven baseline (dG energy-difference, PNAS filtering, Huber+ranking loss, k-NN GAT,
`mini_batch 16`, PCC ≈ 0.5259) **bit-for-bit**.

## The model in one paragraph

DeepEF predicts protein folding free energy by contrasting a **folded** graph (real 3D
contacts) against an **unfolded** reference graph (chain-local contacts only). Residues are
nodes; each carries a distance-derived feature block, bonded features, a protein-language-model
(ProtT5) embedding, and a one-hot amino-acid code. A two-tower GNN (GCN + GATv2) with light
attention maps the graph to a per-residue energy; the protein energy is their sum.

```
dG   = E_unfolded - E_folded
ddG  = dG_mutant  - dG_wildtype
```

The six levers each target a specific weakness in this pipeline, motivated by protein
thermodynamics and polymer physics rather than generic ML tricks.

## The dimension contract (why nothing breaks)

The node-feature layout is:

```
[ D (dist_dim=16) | Fb (2*dist_dim=32) | burial (burial_dim) | emb (E) | one_hot (20) ]
```

**Rule:** every width is driven from ONE value on `model_cfg_v5.CFG`. The graph builder
(`train_utils.get_graph`) and the model (`hydro_net_v5.PEM.forward`) read the *same* numbers.
Consequently, when a lever's flag is at its default, the produced tensors and the energy math
are identical to the historical baseline.

## The six levers

| Lever | Flag(s) | Default (= baseline) | Physics motivation |
|-------|---------|----------------------|--------------------|
| **A. Energy decomposition** | `--energy_terms K` | `K=1` | Split the scalar energy into K interpretable per-residue components (e.g. contact, solvation, backbone), summed to the total. K=1 is a single scalar = today. |
| **B. RBF distance bank** | `--rbf_centers M` `--rbf_min` `--rbf_max` | `M=0` | Replace the single Gaussian distance kernel with a bank of M radial basis functions (a soft histogram of contact distances). Summed back to width 16 → dimension-preserving. M=0 = single Gaussian = today. |
| **C. Burial / solvation** | `--use_burial` `--burial_radius` | off, `burial_dim=0` | Add a per-residue CB neighbor-density scalar (buried vs. exposed) as a node feature inserted before `emb`. Buried and surface residues contribute differently to stability. |
| **D. Flory unfolded reference** | `--flory_unfolded` `--flory_nu` | off | Model the unfolded state as a random coil with distance scaling `~|i−j|^ν` (polymer theory) instead of a purely chain-local (tridiagonal) reference. Value-only change; shapes unchanged. |
| **E. Decoys + denoising head** | `--denoise_weight w` `--denoise_sigma` `--denoise_prob` | `w=0` | An auxiliary MLP predicts injected coordinate noise, teaching the energy surface to have a minimum at the native structure. Train-only aux loss; head absent when w=0. |
| **F. Edge features + extended connectivity** | `--gcn_span S` `--use_edge_features` | `S=1`, off | (i) GCN connects sequence offsets 1..S both directions (S=1 = i,i+1 only = today). (ii) GATv2 consumes a 41-dim edge feature `[onehot_src(20)|onehot_dst(20)|dist(1)]` for explicit pairwise chemistry. Node widths unchanged. |

## Composability

- **A** (energy head) and **E** (aux head off the pre-energy features) are **orthogonal** to
  node/edge widths.
- **B** is **dimension-preserving** (M RBFs summed back to width 16) → no downstream change.
- **D** is **value-only** (unfolded distances change; shapes do not).
- **C** changes the **node width** (adds `burial_dim`); **F** changes only the **edges**.
  Because all widths derive from `CFG`, C and F compose cleanly with each other and the rest.

## Verification

- CPU smoke test (`tests/smoke_test.py`) runs the baseline **and each lever** on synthetic
  data with no GPU and asserts finite outputs + correct shapes.
- Baseline regression: with all flags at default, produced shapes and energy math are
  identical to the pre-lever pipeline (asserted in the smoke test).
- GPU ablation ladder (`run_v5_evolution.sh`) measures PCC gains one lever at a time.
