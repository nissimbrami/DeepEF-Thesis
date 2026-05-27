# QuickStart Guide: DeepPEF Training Pipeline

**Goal:** Get from PCC=0.510 to PCC>=0.70 on MegaScale/PNAS benchmark.

---

## Prerequisites

- GPU with >=8GB VRAM (RTX 4070, T4, A100)
- Linux (Ubuntu 22.04 WSL2 or Colab)
- Python 3.10+ with PyTorch 2.0+
- Data preprocessed in `data/MsDs/training_data/`

---

## Environment Setup

```bash
cd /home/nissimb/workspace/DeepPEF
source /home/nissimb/pytorch_env/bin/activate

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Stage 1: Train ProtT5 Baseline (Best Known Config)

```bash
python Megascale-fineTuning/pnas_train.py \
  --model_name stage1_prott5_seed42 \
  --seed 42 \
  --dataset_type pnas \
  --epochs 15 \
  --epochs_freeze 15 \
  --epochs_unfreeze 0 \
  --no_pretrained \
  --loss_type huber_rank \
  --ranking_weight 0.1 \
  --use_knn_gat \
  --one_mut \
  --dg_ml \
  --cosine_lr \
  --lr_min 1e-6 \
  --weight_decay 1e-5 \
  2>&1 | tee logs/stage1_prott5_seed42.log
```

**Flags explained:**
| Flag | Why |
|------|-----|
| `--no_pretrained` | No checkpoint available; train from scratch |
| `--loss_type huber_rank` | +0.019 PCC over L1 (validated) |
| `--use_knn_gat` | +0.024 PCC over fully-connected (validated) |
| `--ranking_weight 0.1` | Optimal ranking loss weight |
| `--one_mut` | Single-point mutations only |
| `--dg_ml` | Clamp dG to [-1, 5] |
| `--cosine_lr` | Smooth LR decay |
| `--weight_decay 1e-5` | Regularize 22M params |
| `--epochs 15` | Optimal (30 didn't help) |

**Expected PCC:** 0.50-0.52 | **Time:** ~2-3 hours

---

## Stage 2: SaProt Embeddings

### Step 2A: Regenerate embeddings (fix broken ones)

```bash
find data/MsDs/training_data -name "saprot_wt.pt" -delete
python data_creation/generate_saprot_embeddings.py --force
```

### Validate:
```python
import torch, os
for p in ['1A0N', '1BNI', '1CSP']:
    e = torch.load(f'data/MsDs/training_data/{p}/saprot_wt.pt')
    print(f"{p}: {e.shape}")  # Must be [seq_len, 1280], NOT [2, 1280]
```

### Step 2B: Train with SaProt

```bash
python Megascale-fineTuning/pnas_train.py \
  --model_name stage2_saprot_seed42 \
  --seed 42 \
  --dataset_type pnas \
  --epochs 15 \
  --epochs_freeze 15 \
  --epochs_unfreeze 0 \
  --no_pretrained \
  --loss_type huber_rank \
  --ranking_weight 0.1 \
  --use_knn_gat \
  --one_mut \
  --dg_ml \
  --cosine_lr \
  --lr_min 1e-6 \
  --weight_decay 1e-5 \
  --emb_type saprot \
  2>&1 | tee logs/stage2_saprot_seed42.log
```

**Expected PCC:** 0.55-0.60 | **Time:** ~2-3 hours

---

## Stage 4: Ensemble (5 Seeds)

```bash
for SEED in 42 123 456 789 1337; do
  python Megascale-fineTuning/pnas_train.py \
    --model_name ensemble_saprot_seed${SEED} \
    --seed ${SEED} \
    --dataset_type pnas \
    --epochs 15 \
    --epochs_freeze 15 \
    --epochs_unfreeze 0 \
    --no_pretrained \
    --loss_type huber_rank \
    --ranking_weight 0.1 \
    --use_knn_gat \
    --one_mut \
    --dg_ml \
    --cosine_lr \
    --lr_min 1e-6 \
    --weight_decay 1e-5 \
    --emb_type saprot \
    2>&1 | tee logs/ensemble_saprot_seed${SEED}.log
done
```

**Expected PCC:** +0.02-0.04 over single model | **Time:** ~12 hours total

---

## Decision Tree

```
After Stage 1 (PCC = ___):
  >= 0.52  -> Continue to Stage 2
  < 0.50   -> Check logs, data loading

After Stage 2 (PCC = ___):
  >= 0.60  -> Skip to Stage 4 (ensemble)
  0.55-0.60 -> Optional: Stage 3 per-protein FT
  < 0.55   -> Check saprot_wt.pt shapes

After Stage 4 (PCC = ___):
  >= 0.70  -> TARGET REACHED
  0.65-0.70 -> Strong thesis result
  < 0.65   -> Ceiling without pretraining
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Expected size X got Y` | SaProt embeddings wrong shape. Run `generate_saprot_embeddings.py --force` |
| `CUDA out of memory` | Add `--max_muts 64` |
| `KeyError: 'name'` | Wrong dataset_type, use `--dataset_type pnas` |
| wandb login | `export WANDB_MODE=disabled` |

---

## Key Files

| File | Purpose |
|------|---------|
| `Megascale-fineTuning/pnas_train.py` | Main training script |
| `Megascale-fineTuning/new_dataset.py` | Dataset (supports prott5/esm2/saprot) |
| `model/hydro_net.py` | PEM model (GCN + GAT + Light Attention) |
| `model/model_cfg.py` | Hyperparameters |
| `data_creation/generate_saprot_embeddings.py` | SaProt embedding generation |
| `train_utils.py` | Graph construction utilities |
