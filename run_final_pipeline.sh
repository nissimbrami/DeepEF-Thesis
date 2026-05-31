#!/bin/bash
# ============================================================
# DeepPEF FINAL PIPELINE — Correct approach after v3 failure
#
# v3 failed because 1536-dim forced batch_size=8 (OOM at 64)
# Small batches = noisy gradients = PCC 0.42 (below baseline 0.52)
#
# This script does:
# 1. Kill broken training
# 2. ProtT5-only 5-seed ensemble (guaranteed 0.52+ baseline)
# 3. dual_esmif WITH emb_projection=mlp (projects 1536→16, keeps batch=64)
# 4. Compare all results
#
# Run on GPU machine:
#   cd /home/nissimb/workspace/DeepPEF && git pull
#   setsid bash run_final_pipeline.sh > logs/final_pipeline.log 2>&1 & disown
# ============================================================

set -e
cd /home/nissimb/workspace/DeepPEF
source /home/nissimb/pytorch_env/bin/activate
export WANDB_MODE=disabled

echo "============================================================"
echo "DeepPEF FINAL PIPELINE — $(date)"
echo "============================================================"
echo ""

# STEP 0: Kill previous broken jobs, clean caches
echo "====== STEP 0: Clean slate ======"
pkill -f "pnas_train.py" 2>/dev/null || true
sleep 2
find . -path './.git' -prune -o -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true
echo "  Done. GPU free."
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

# STEP 1: Verify data
echo "====== STEP 1: Verify data integrity ======"
python -c "
import torch, os, sys
td = './data/MsDs/training_data'
proteins = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
# Check esmif_enc.pt exists
missing_esmif = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'esmif_enc.pt'))]
# Check emb.pt exists
missing_emb = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'emb.pt'))]
print(f'  Total proteins: {len(proteins)}')
print(f'  Missing esmif_enc.pt: {len(missing_esmif)}')
print(f'  Missing emb.pt: {len(missing_emb)}')
if missing_esmif:
    print(f'  WARNING: {missing_esmif[:5]}...')
if missing_emb:
    print(f'  FATAL: emb.pt missing!')
    sys.exit(1)
# Check test split works
import pandas as pd
tm = pd.read_csv('./data/ThermoMPNN/mega_test.csv')
test_names = tm['WT_name'].str.replace('.pdb', '', regex=False).unique().tolist()
test_found = [p for p in test_names if p in proteins]
print(f'  Test proteins found: {len(test_found)}/{len(test_names)}')
assert len(test_found) > 0, 'FATAL: 0 test proteins!'
print('  ALL CHECKS PASSED')
"
echo ""

mkdir -p Megascale-fineTuning/models logs

# ============================================================
# STEP 2: ProtT5-only 5-seed ensemble (BASELINE)
# This is what worked (PCC=0.5259). 5 seeds → ensemble ~0.55-0.57
# ============================================================
echo "====== STEP 2: ProtT5-only 5-seed ensemble (BASELINE) ======"
echo "  Config: from_scratch, huber_rank, knn_gat, batch=64, 15 epochs"
echo ""

for SEED in 42 123 456 789 1337; do
    echo "  --- ProtT5 seed $SEED --- (started $(date +%H:%M))"
    python Megascale-fineTuning/pnas_train.py \
        --model_name baseline_prott5_seed${SEED} \
        --seed $SEED \
        --dataset_type pnas \
        --epochs 15 \
        --no_pretrained \
        --loss_type huber_rank \
        --ranking_weight 0.1 \
        --use_knn_gat \
        --one_mut \
        --dg_ml \
        --cosine_lr \
        --lr_min 1e-6 \
        --weight_decay 1e-5 \
        --emb_type prott5 \
        --mini_batch_size 64 \
        --emb_projection none
    echo "  Seed $SEED done at $(date +%H:%M)"
    echo ""
done

echo ""
echo "====== STEP 2 COMPLETE: ProtT5 5-seed ensemble done ======"
echo ""

# ============================================================
# STEP 3: dual_esmif WITH projection (keeps batch=64!)
# Project 1536→16 before GNN = no OOM, proper batch size
# ============================================================
echo "====== STEP 3: dual_esmif WITH MLP projection — 5 seeds ======"
echo "  Config: 1536-dim → project to 16 before GNN, batch=64, 15 epochs"
echo "  This fixes the OOM issue that killed v3 (batch=8 was too noisy)"
echo ""

for SEED in 42 123 456 789 1337; do
    echo "  --- dual_esmif+proj seed $SEED --- (started $(date +%H:%M))"
    python Megascale-fineTuning/pnas_train.py \
        --model_name v4_dual_esmif_proj_seed${SEED} \
        --seed $SEED \
        --dataset_type pnas \
        --epochs 15 \
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
        --mini_batch_size 64 \
        --emb_projection mlp
    echo "  Seed $SEED done at $(date +%H:%M)"
    echo ""
done

echo ""
echo "====== STEP 3 COMPLETE: dual_esmif+projection 5-seed done ======"
echo ""

# ============================================================
# STEP 4: esmif_enc only with projection — comparison
# ============================================================
echo "====== STEP 4: ESM-IF1 only (512-dim) + projection — 5 seeds ======"

for SEED in 42 123 456 789 1337; do
    echo "  --- esmif_enc+proj seed $SEED --- (started $(date +%H:%M))"
    python Megascale-fineTuning/pnas_train.py \
        --model_name v4_esmif_only_proj_seed${SEED} \
        --seed $SEED \
        --dataset_type pnas \
        --epochs 15 \
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
        --mini_batch_size 64 \
        --emb_projection mlp
    echo "  Seed $SEED done at $(date +%H:%M)"
    echo ""
done

echo ""
echo "====== STEP 4 COMPLETE ======"
echo ""

# ============================================================
# STEP 5: Print all results
# ============================================================
echo "============================================================"
echo "ALL TRAINING COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""
echo "--- ProtT5 baseline (5 seeds) ---"
grep "Best Pearson" logs/final_pipeline.log | grep "baseline_prott5" || \
grep "best_pc_corr\|Best.*PCC\|Training completed" logs/final_pipeline.log | head -5
echo ""
echo "--- dual_esmif + projection (5 seeds) ---"
grep "Best Pearson" logs/final_pipeline.log | grep "v4_dual_esmif" || \
grep "best_pc_corr\|Best.*PCC\|Training completed" logs/final_pipeline.log | tail -10
echo ""
echo "Check detailed results with:"
echo "  grep 'PCC:' logs/final_pipeline.log"
echo "  grep 'Training completed' logs/final_pipeline.log"
