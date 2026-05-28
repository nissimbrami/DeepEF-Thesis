#!/bin/bash
# DeepPEF v3: ESM-IF1 Encoder Features + Our PEM Architecture (thermodynamic approach)
#
# KEY INSIGHT: ESM-IF1 encoder learns "what amino acids fit at each structural position"
# from millions of protein structures. This is the SAME structural knowledge that
# ThermoMPNN gets from ProteinMPNN — but we feed it INTO our own GNN as features,
# keeping our thermodynamic framework (ΔΔG = ΔG_unfolded - ΔG_folded).
#
# Architecture stays OURS:
#   - GCN branch (bonded + non-bonded features)
#   - GAT branch (k-NN, k=30)
#   - Light Attention pooling
#   - Embedding: ProtT5 (1024) + ESM-IF1 encoder (512) = 1536-dim
#   - Energy computation: ΔG(mut) - ΔG(wt)
#
# Expected PCC: 0.60-0.75 (based on ThermoMPNN showing +0.12 from structural pretraining)
#
# Pipeline:
#   Step 1: Generate ESM-IF1 encoder features (~1 hour, one pass per protein)
#   Step 2: Train with dual_esmif (ProtT5 + ESM-IF1 encoder, 1536-dim)
#   Step 3: Compare esmif_enc alone vs dual_esmif vs prott5
#   Step 4: 5-seed ensemble of best config
#
# Usage:
#   cd /home/nissimb/workspace/DeepPEF
#   source /home/nissimb/pytorch_env/bin/activate
#   nohup bash run_v3_esmif_encoder.sh > logs/v3_pipeline.log 2>&1 &
#   disown

set -e
cd /home/nissimb/workspace/DeepPEF
source /home/nissimb/pytorch_env/bin/activate
export WANDB_MODE=disabled
mkdir -p logs Megascale-fineTuning/models

# Clear Python bytecode cache to ensure latest code is used
find . -path './.git' -prune -o -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

echo "============================================================"
echo "DeepPEF v3: ESM-IF1 ENCODER + OUR PEM (thermodynamic)"
echo "Started at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo "============================================================"
echo ""
echo "This is OUR model with ESM-IF1 structural features added."
echo "Architecture: GCN + GAT + Light Attention + ProtT5 + ESM-IF1 encoder"
echo "Thermodynamic: ΔΔG = E_unfolded(mut) - E_folded(mut) - (E_unfolded(wt) - E_folded(wt))"
echo ""

# ==============================================================
# STEP 1: Generate ESM-IF1 encoder features (512-dim per residue)
# ==============================================================
echo "====== STEP 1: Generate ESM-IF1 encoder representations ======"
echo "  One forward pass per protein → [L, 512] structural features"
echo "  Started at $(date)"

python data_creation/generate_esmif_encoder_features.py 2>&1 | tee logs/esmif_encoder_gen.log

echo "  DONE at $(date)"
echo ""

# Quick validation
python -c "
import torch, os
td = './data/MsDs/training_data'
total = 0
ok = 0
for p in sorted(os.listdir(td)):
    sp = os.path.join(td, p, 'esmif_enc.pt')
    if os.path.exists(sp):
        ok += 1
    total += 1
print(f'ESM-IF1 encoder features: {ok}/{total} proteins generated')
if ok < total * 0.9:
    print('WARNING: Less than 90% generated. Check errors.')
else:
    print('OK: Coverage is good.')
# Show sample shapes
for p in sorted(os.listdir(td))[:3]:
    sp = os.path.join(td, p, 'esmif_enc.pt')
    if os.path.exists(sp):
        t = torch.load(sp, weights_only=True)
        print(f'  {p}: shape={t.shape}, dtype={t.dtype}')
"

# ==============================================================
# STEP 2: Train with ESM-IF1 encoder only (512-dim) — baseline
# ==============================================================
echo ""
echo "====== STEP 2: ESM-IF1 encoder only (512-dim) ======"
echo "  Tests whether structural features alone are sufficient"
echo "  Started at $(date)"

python Megascale-fineTuning/pnas_train.py \
  --model_name v3_esmif_enc_seed42 \
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
  --emb_type esmif_enc \
  2>&1 | tee logs/v3_esmif_enc_seed42.log
echo "  DONE at $(date)"
echo "  Result:" && grep -i "pearson\|pcc\|test.*corr" logs/v3_esmif_enc_seed42.log | tail -3

# ==============================================================
# STEP 3: Train with DUAL ProtT5 + ESM-IF1 encoder (1536-dim) — THE KEY
# ==============================================================
echo ""
echo "====== STEP 3: DUAL ProtT5 + ESM-IF1 encoder (1536-dim) ======"
echo "  This is the strongest config: per-mutation ProtT5 + structural ESM-IF1"
echo "  Started at $(date)"

python Megascale-fineTuning/pnas_train.py \
  --model_name v3_dual_esmif_seed42 \
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
  --emb_type dual_esmif \
  2>&1 | tee logs/v3_dual_esmif_seed42.log
echo "  DONE at $(date)"
echo "  Result:" && grep -i "pearson\|pcc\|test.*corr" logs/v3_dual_esmif_seed42.log | tail -3

# ==============================================================
# STEP 4: Compare all configs and pick best
# ==============================================================
echo ""
echo "====== STEP 4: Comparing configs ======"

python -c "
import re, os

def get_best_pcc(log_path):
    best = 0.0
    if not os.path.exists(log_path):
        return 0.0
    with open(log_path) as f:
        for line in f:
            matches = re.findall(r'(?:pearson|pcc|PCC)[^\\d]*(\\d+\\.\\d+)', line)
            for m in matches:
                val = float(m)
                if 0.1 < val < 1.0 and val > best:
                    best = val
    return best

configs = {
    'v1_prott5': get_best_pcc('logs/stage1_prott5_seed42.log'),
    'v3_esmif_enc': get_best_pcc('logs/v3_esmif_enc_seed42.log'),
    'v3_dual_esmif': get_best_pcc('logs/v3_dual_esmif_seed42.log'),
}

print('\\n=== PCC COMPARISON ===')
for name, pcc in sorted(configs.items(), key=lambda x: -x[1]):
    marker = ' *** BEST ***' if pcc == max(configs.values()) else ''
    print(f'  {name:20s}: {pcc:.4f}{marker}')

best_name = max(configs, key=configs.get)
# Map config name to emb_type argument
emb_map = {'v1_prott5': 'prott5', 'v3_esmif_enc': 'esmif_enc', 'v3_dual_esmif': 'dual_esmif'}
best_emb = emb_map.get(best_name, 'dual_esmif')
print(f'\\nBEST: {best_name} (PCC={configs[best_name]:.4f})')
print(f'Using emb_type={best_emb} for ensemble')
with open('logs/v3_best_emb_type.txt', 'w') as f:
    f.write(best_emb)
"

# ==============================================================
# STEP 5: 5-seed ensemble with best config
# ==============================================================
BEST_EMB=$(cat logs/v3_best_emb_type.txt)
echo ""
echo "====== STEP 5: 5-SEED ENSEMBLE with ${BEST_EMB} ======"
echo "  Started at $(date)"

for SEED in 42 123 456 789 1337; do
  echo ""
  echo "  --- Seed ${SEED} started at $(date) ---"
  python Megascale-fineTuning/pnas_train.py \
    --model_name v3_final_${BEST_EMB}_seed${SEED} \
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
    --emb_type ${BEST_EMB} \
    2>&1 | tee logs/v3_final_${BEST_EMB}_seed${SEED}.log
  grep -i "pearson\|pcc\|test.*corr" logs/v3_final_${BEST_EMB}_seed${SEED}.log | tail -1
done

# ==============================================================
# FINAL REPORT
# ==============================================================
echo ""
echo "============================================================"
echo "v3 PIPELINE COMPLETE at $(date)"
echo "============================================================"
echo ""
echo "RESULTS:"
echo "  v1 ProtT5 (1024-dim):"
grep -i "pearson\|pcc" logs/stage1_prott5_seed42.log 2>/dev/null | tail -1 || echo "    not available"
echo "  v3 ESM-IF1 encoder (512-dim):"
grep -i "pearson\|pcc" logs/v3_esmif_enc_seed42.log 2>/dev/null | tail -1 || echo "    not available"
echo "  v3 DUAL ProtT5+ESM-IF1 (1536-dim):"
grep -i "pearson\|pcc" logs/v3_dual_esmif_seed42.log 2>/dev/null | tail -1 || echo "    not available"
echo ""
echo "  ENSEMBLE (${BEST_EMB}, 5 seeds):"
for SEED in 42 123 456 789 1337; do
  echo -n "    Seed ${SEED}: "
  grep -i "pearson\|pcc" logs/v3_final_${BEST_EMB}_seed${SEED}.log 2>/dev/null | tail -1 || echo "N/A"
done
echo ""
echo "TARGET: PCC >= 0.70"
echo "============================================================"
