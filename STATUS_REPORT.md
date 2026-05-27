# DeepPEF Project Status Report

**Last updated:** May 2026
**Author:** Nissim Brami (M.Sc. Thesis, continuation of Shahar Cohen's DeepEF)

---

## Project Goal

Predict protein thermodynamic stability changes (ddG) upon single-point mutations using a Graph Neural Network (GNN) + Protein Language Model (PLM) architecture.

**Target:** Pearson Correlation Coefficient (PCC) >= 0.70 on the MegaScale/PNAS 28-protein test set.

---

## Current Best Result

| Metric | Value |
|--------|-------|
| **PCC** | **0.510** |
| Test set | 28 proteins (ThermoMPNN split) |
| Training set | 340 proteins (~50K single-point mutations) |
| Config | k-NN GAT (k=30), Huber+Ranking loss, ProtT5 embeddings, 15 epochs |

---

## Architecture: PEM (Protein Energy Model)

```
Input: [seq_len, D(16) + Fb(32) + embedding(1024/1280) + one_hot(20)]
    |
    +-- GCN Branch (bonded features) --- 3 layers, fully-connected edges
    |           |
    +-- GAT Branch (non-bonded features) --- 3 layers, k-NN edges (k=30)
                |
        [Concatenate GCN + GAT outputs]
                |
        + Raw LLM embedding (late fusion)
                |
        Light Attention Pooling
                |
        MLP (128 -> 1)
                |
        dG prediction (folded vs unfolded -> ddG)
```

- **22M parameters**, batch_size=1 (protein-level batching)
- Dual-state energy: predicts folded energy & unfolded energy separately, ddG = dG_mut - dG_wt
- k-NN GAT: each residue connects to 30 nearest neighbors by CA distance

---

## What Changed From Baseline (Chronological)

| Step | Change | PCC | Gain |
|------|--------|-----|------|
| 0 | Baseline (L1 loss, fully-connected, from scratch) | 0.467 | - |
| 1 | Huber + Ranking loss (lambda=0.1, margin=0.1) | 0.486 | +0.019 |
| 2 | k-NN GAT (k=30, distance-based edges) | **0.510** | +0.024 |
| - | Stage 1 training improvements (cosine LR, 30 epochs) | 0.504 | -0.006 |

**Total improvement from baseline:** +0.043

---

## What Works (Validated by Controlled Ablation)

| Change | Effect | Status |
|--------|--------|--------|
| k-NN GAT (k=30) | +0.024 PCC | KEEP |
| Huber + Ranking loss | +0.019 PCC | KEEP |
| Training from scratch (no pretrained checkpoint) | Works, PCC=0.51 | Validated |
| ProtT5 per-mutant embeddings (1024-dim) | Stable baseline | Current default |

---

## What Failed (Do NOT Retry)

| Change | Effect | Reason |
|--------|--------|--------|
| Multi-RBF distance encoding | **-0.081 PCC** | Adds noise to distance features |
| Edge features in GAT (32-dim) | -0.003 PCC | Negligible benefit |
| Multi-RBF + k-NN + Edge combined | -0.047 PCC | Multi-RBF dominates negatively |
| ESM-2 embeddings (replace ProtT5) | PCC=0.25 (catastrophic) | WT-only embedding loses per-mutant info |
| Dual embeddings (ProtT5+ESM-2) | CRASH | Dimension mismatch |
| Phase 2 full bundle | CRASH | Multiple bugs |

---

## What Needs Re-run (Bugs Fixed)

### SaProt Embeddings (Structure-Aware, 1280-dim)
- **Status:** Bug in tokenization script produced [2, 1280] instead of [seq_len, 1280]
- **Root cause:** SaProt tokenizer treats each bigram (AA+3Di pair) as one token. Script fixed.
- **Expected gain:** +0.03-0.05 PCC (published SaProt Spearman 0.724 vs ProtT5 ~0.65)
- **Action:** Re-generate embeddings with `--force` flag, then train

### Per-Protein Fine-Tuning
- **Status:** CRASH due to checkpoint mismatch
- **Expected gain:** +0.03-0.05 PCC
- **Action:** Load checkpoint with correct flags (emb_projection="none")

---

## Improvement Plan (Stages 1-4)

### Stage 1: Training Optimization - DONE (PCC=0.504)
- Cosine annealing LR, gradient accumulation, weight decay
- Result: Marginal, model needs better input (Stage 2)

### Stage 2: SaProt Embeddings - NEXT (bug fixed)
- Replace ProtT5 (1024-dim) with SaProt (1280-dim, structure-aware)
- **Expected PCC: 0.55-0.60**

### Stage 3: Per-Protein Fine-Tuning
- Fine-tune per test protein (5-10 epochs each)
- **Expected PCC: 0.60-0.65**

### Stage 4: 5-Seed Ensemble
- Average 5 models (seeds: 42, 123, 456, 789, 1337)
- **Expected PCC: 0.65-0.72**

---

## Comparison with Published Methods

| Method | PCC (approx.) | Uses Structure? |
|--------|---------------|-----------------|
| BLOSUM62 baseline | ~0.30 | No |
| ESM-2 zero-shot | ~0.45 | No |
| **DeepPEF (this work)** | **0.510** | Yes |
| DeepEF original (with pretraining) | ~0.54 | Yes |
| RaSP | ~0.60 | Yes |
| ThermoMPNN | ~0.75 | Yes |

---

## Realistic Expectations

| Final PCC Range | Probability |
|-----------------|-------------|
| >= 0.75 | 20-30% |
| 0.70-0.75 | 25-35% |
| 0.65-0.70 | 30-35% |
| < 0.65 | 10-20% |

**Target with all improvements:** PCC 0.65-0.72 (strong thesis result).
