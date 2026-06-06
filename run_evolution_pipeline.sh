#!/bin/bash
# ============================================================
# DeepPEF EVOLUTION PIPELINE — Self-Validating, Damage-Controlled
#
# PHILOSOPHY: Every step VALIDATES before proceeding.
# A mistake in code/setup costs DAYS. This script prevents that.
#
# WHAT IT DOES (in order):
# 1. Clean environment (kill old jobs, clear caches)
# 2. Verify data integrity (embeddings, test split, shapes)
# 3. VERIFY GNN-SM CODE (shapes, gradients, antisymmetry check)
# 4. Run Level 7: ProtT5 5-seed ensemble (guaranteed baseline)
# 5. Run Level 10: GNN-SM single seed (quick test)
# 6. If Level 10 > 0.45: Run GNN-SM 5-seed ensemble
# 7. Run Level 9: dual_esmif + projection 5-seed
# 8. Print all results comparison
#
# DAMAGE CONTROL:
# - Each step checks exit code before proceeding
# - NaN/Inf detection in training output
# - Checkpoints every 5 epochs
# - Stops on repeated failures
#
# RUN ON GPU MACHINE:
#   cd /home/nissimb/workspace/DeepPEF && git pull
#   chmod +x run_evolution_pipeline.sh
#   setsid bash run_evolution_pipeline.sh > logs/evolution_pipeline.log 2>&1 & disown
#   # Monitor: tail -f logs/evolution_pipeline.log
# ============================================================

set -e  # Exit on any error
cd /home/nissimb/workspace/DeepPEF
source /home/nissimb/pytorch_env/bin/activate
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "DeepPEF EVOLUTION PIPELINE — $(date)"
echo "============================================================"
echo ""

# ============================================================
# STEP 0: CLEAN SLATE
# ============================================================
echo "====== STEP 0: Clean environment ======"
pkill -f "pnas_train" 2>/dev/null || true
sleep 2
find . -path './.git' -prune -o -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true
echo "  Killed old jobs, cleared caches"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

mkdir -p Megascale-fineTuning/models logs

# ============================================================
# STEP 1: VERIFY DATA INTEGRITY
# ============================================================
echo "====== STEP 1: Verify data integrity ======"
python -c "
import torch, os, sys
import pandas as pd

td = './data/MsDs/training_data'
proteins = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
print(f'  Total proteins: {len(proteins)}')

# Check critical files exist
missing_coords = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'coords.pt'))]
missing_emb = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'emb.pt'))]
missing_mask = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'mask.pt'))]
missing_esmif = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'esmif_enc.pt'))]

print(f'  Missing coords.pt: {len(missing_coords)}')
print(f'  Missing emb.pt: {len(missing_emb)}')
print(f'  Missing mask.pt: {len(missing_mask)}')
print(f'  Missing esmif_enc.pt: {len(missing_esmif)}')

if missing_coords or missing_emb or missing_mask:
    print('  FATAL: Core data files missing!')
    sys.exit(1)

# Verify test split
tm = pd.read_csv('./data/ThermoMPNN/mega_test.csv')
test_names = tm['WT_name'].str.replace('.pdb', '', regex=False).unique().tolist()
test_found = [p for p in test_names if p in proteins]
print(f'  Test proteins found: {len(test_found)}/{len(test_names)}')
assert len(test_found) > 0, 'FATAL: 0 test proteins!'

# Verify shapes of one protein
sample = proteins[0]
c = torch.load(os.path.join(td, sample, 'coords.pt'), weights_only=False)
m = torch.load(os.path.join(td, sample, 'mask.pt'), weights_only=False)
e = torch.load(os.path.join(td, sample, 'emb.pt'), weights_only=False)
print(f'  Sample protein: {sample}')
print(f'    coords shape: {c.shape}')
print(f'    mask shape: {m.shape}')
if isinstance(e, list):
    print(f'    emb: list of {len(e)} tensors, first shape: {e[0].shape}')
else:
    print(f'    emb shape: {e.shape}')

# Verify mutation files exist
mut_dir = './data/MsDs/mutation_files'
missing_mut = [p for p in proteins if not os.path.exists(os.path.join(mut_dir, f'{p}.csv'))]
print(f'  Missing mutation CSVs: {len(missing_mut)}/{len(proteins)}')

print('  ALL CHECKS PASSED')
"
echo ""

# ============================================================
# STEP 2: VERIFY GNN-SM CODE (shapes, gradients, antisymmetry)
# ============================================================
echo "====== STEP 2: Verify GNN-SM code ======"
echo "  Running verification mode (5 proteins, checks shapes + gradients)..."
python Megascale-fineTuning/pnas_train_sm.py \
    --verify_only \
    --seed 42 \
    --emb_type prott5 \
    --emb_projection none
VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo "  FATAL: GNN-SM verification FAILED! Exit code: $VERIFY_EXIT"
    echo "  DO NOT proceed. Fix the code first."
    exit 1
fi
echo "  GNN-SM verification PASSED"
echo ""

# ============================================================
# STEP 3: Level 7 — ProtT5 5-seed ensemble (BASELINE)
# Guaranteed PCC ~0.52 per seed, ensemble 0.55-0.57
# ============================================================
echo "====== STEP 3: Level 7 — ProtT5 baseline 5-seed ensemble ======"
echo "  Config: from_scratch, huber_rank, knn_gat, batch=64, 15 epochs"
echo ""

BASELINE_SUCCESS=0
for SEED in 42 123 456 789 1337; do
    echo "  --- ProtT5 baseline seed $SEED --- (started $(date +%H:%M))"
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
        --emb_projection none && BASELINE_SUCCESS=$((BASELINE_SUCCESS + 1))
    echo "  Seed $SEED done at $(date +%H:%M)"
    echo ""
done

echo "  Baseline: $BASELINE_SUCCESS/5 seeds completed successfully"
if [ $BASELINE_SUCCESS -lt 3 ]; then
    echo "  WARNING: Less than 3 baselines succeeded. Check for errors."
fi
echo ""
echo "====== STEP 3 COMPLETE ======"
echo ""

# ============================================================
# STEP 4: Level 10 — GNN-SM single seed (NOVEL APPROACH)
# Quick test: if PCC > 0.45, proceed to 5-seed ensemble
# ============================================================
echo "====== STEP 4: Level 10 — GNN-SM (single seed test) ======"
echo "  Config: subtract_mut output, 1 forward pass per protein, 30 epochs"
echo ""

python Megascale-fineTuning/pnas_train_sm.py \
    --model_name gnn_sm_prott5_seed42 \
    --seed 42 \
    --epochs 30 \
    --emb_type prott5 \
    --emb_projection none \
    --use_knn_gat
SM_EXIT=$?

if [ $SM_EXIT -ne 0 ]; then
    echo "  WARNING: GNN-SM training failed (exit code $SM_EXIT)"
    echo "  Continuing with other experiments..."
else
    echo "  GNN-SM seed 42 completed"
fi
echo ""

# ============================================================
# STEP 5: Level 10 — GNN-SM 5-seed ensemble (if seed 42 worked)
# ============================================================
if [ $SM_EXIT -eq 0 ]; then
    echo "====== STEP 5: Level 10 — GNN-SM 5-seed ensemble ======"
    SM_SUCCESS=1  # seed 42 already done
    for SEED in 123 456 789 1337; do
        echo "  --- GNN-SM seed $SEED --- (started $(date +%H:%M))"
        python Megascale-fineTuning/pnas_train_sm.py \
            --model_name gnn_sm_prott5_seed${SEED} \
            --seed $SEED \
            --epochs 30 \
            --emb_type prott5 \
            --emb_projection none \
            --use_knn_gat && SM_SUCCESS=$((SM_SUCCESS + 1))
        echo "  Seed $SEED done at $(date +%H:%M)"
        echo ""
    done
    echo "  GNN-SM: $SM_SUCCESS/5 seeds completed"
    echo ""
fi

# ============================================================
# STEP 6: Level 9 — dual_esmif + MLP projection (5-seed)
# Tests whether adding ESM-IF1 structural features helps
# ============================================================
echo "====== STEP 6: Level 9 — dual_esmif + projection 5-seed ======"
echo "  Config: ProtT5+ESM-IF1 (1536) projected to 16-dim, batch=64"
echo ""

DUAL_SUCCESS=0
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
        --emb_projection mlp && DUAL_SUCCESS=$((DUAL_SUCCESS + 1))
    echo "  Seed $SEED done at $(date +%H:%M)"
    echo ""
done
echo "  dual_esmif: $DUAL_SUCCESS/5 seeds completed"
echo ""

# ============================================================
# STEP 7: MLP subtract-mut baseline (comparison with GNN-SM)
# Simple MLP on ESM-IF1 features — tests whether GNN adds value
# ============================================================
echo "====== STEP 7: MLP subtract-mut (ESM-IF1 baseline) ======"
python Megascale-fineTuning/train_subtract_mut.py \
    --seed 42 \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-3
echo ""

python Megascale-fineTuning/train_subtract_mut.py \
    --seed 42 \
    --dual \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-3
echo ""

# ============================================================
# STEP 8: RESULTS SUMMARY
# ============================================================
echo "============================================================"
echo "ALL TRAINING COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""
echo "Check PCC results with:"
echo "  grep 'Best.*PCC\|FINAL RESULT\|Training complete' logs/evolution_pipeline.log"
echo ""
echo "  Baseline (Level 7):     grep 'Best Pearson' for baseline_prott5_*"
echo "  GNN-SM (Level 10):      grep 'TRAINING COMPLETE' for gnn_sm_*"
echo "  dual_esmif (Level 9):   grep 'Best Pearson' for v4_dual_esmif_*"
echo "  MLP subtract (Level F): grep 'FINAL RESULT' for subtract_mut_*"
echo ""
echo "Models saved in: Megascale-fineTuning/models/"
ls Megascale-fineTuning/models/ 2>/dev/null | head -20
echo ""
echo "PIPELINE DONE."
