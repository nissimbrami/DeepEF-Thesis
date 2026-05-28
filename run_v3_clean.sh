#!/bin/bash
# ============================================================
# DeepPEF v3 CLEAN LAUNCH — Kill broken jobs, run ESM-IF1 dual
# Run this on the GPU machine:
#   setsid bash run_v3_clean.sh > logs/v3_clean.log 2>&1 &
# ============================================================

set -e
cd /home/nissimb/workspace/DeepPEF
source /home/nissimb/pytorch_env/bin/activate

echo "============================================================"
echo "DeepPEF v3 CLEAN START — $(date)"
echo "============================================================"

# STEP 0: Kill all existing training processes
echo ""
echo "====== STEP 0: Kill broken/duplicate training jobs ======"
pkill -f "pnas_train.py" 2>/dev/null || true
pkill -f "run_mega_pipeline" 2>/dev/null || true
sleep 2
echo "  Killed existing jobs. Verifying..."
ps aux | grep pnas_train | grep -v grep && echo "  WARNING: jobs still running!" || echo "  All clear."
echo ""

# STEP 0.5: Pull latest code and clear caches
echo "====== STEP 0.5: Pull latest code, clear caches ======"
git pull origin main
find . -path './.git' -prune -o -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true
echo "  Code updated, caches cleared."
echo ""

# STEP 1: Verify ESM-IF1 encoder features exist
echo "====== STEP 1: Verify ESM-IF1 encoder features ======"
python -c "
import torch, os, sys
td = './data/MsDs/training_data'
proteins = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
missing = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'esmif_enc.pt'))]
if missing:
    print(f'ERROR: {len(missing)} proteins missing esmif_enc.pt: {missing[:5]}...')
    sys.exit(1)
# Check one shape
sample = torch.load(os.path.join(td, proteins[0], 'esmif_enc.pt'), weights_only=True)
print(f'  ESM-IF1 features verified: {len(proteins)} proteins, sample shape={sample.shape}')
assert sample.shape[1] == 512, f'Expected 512 dims, got {sample.shape[1]}'
print('  All good!')
"
echo ""

# STEP 2: Verify test protein detection works
echo "====== STEP 2: Verify dataset loads correctly ======"
python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './Megascale-fineTuning')
import new_dataset
new_dataset.EMB_TYPE = 'dual_esmif'
new_dataset.DS_TYPE = 'pnas'
new_dataset.DG_ML = True
new_dataset.ONE_MUT = True
new_dataset.UNSTABLE_MUT = True
ds = new_dataset.MSDataset(
    tensor_root_dir='./data/MsDs/training_data',
    mutations_root_dir='./data/MsDs/mutation_files',
    train=False
)
n_test = len(ds)
print(f'  Test dataset: {n_test} proteins')
assert n_test > 0, 'FATAL: 0 test proteins!'
print(f'  Loading first batch...')
batch = ds[0]
emb_shape = batch['prott5'].shape
print(f'  First batch embedding shape: {emb_shape}')
assert emb_shape[-1] == 1536, f'Expected 1536 dim (1024+512), got {emb_shape[-1]}'
print('  Dataset verification PASSED!')
"
echo ""

# STEP 3: Run v3 dual_esmif training (ProtT5 1024 + ESM-IF1 512 = 1536)
echo "====== STEP 3: Training dual_esmif (seed 42) ======"
echo "  Config: from_scratch, huber_rank, knn_gat, 15 epochs"
echo "  Embedding: ProtT5 (1024) + ESM-IF1 encoder (512) = 1536 dim"
echo "  Started at $(date)"
echo ""

export WANDB_MODE=disabled
mkdir -p Megascale-fineTuning/models

python Megascale-fineTuning/pnas_train.py \
    --model_name v3_dual_esmif_seed42 \
    --seed 42 \
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
    --emb_type dual_esmif

echo ""
echo "====== Training 1 (dual_esmif seed42) DONE at $(date) ======"
echo ""

# STEP 4: Run esmif_enc only (512-dim, structural only — for comparison)
echo "====== STEP 4: Training esmif_enc only (seed 42) ======"
python Megascale-fineTuning/pnas_train.py \
    --model_name v3_esmif_enc_only_seed42 \
    --seed 42 \
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
    --emb_type esmif_enc

echo ""
echo "====== Training 2 (esmif_enc only) DONE at $(date) ======"
echo ""

# STEP 5: 5-seed ensemble for dual_esmif
echo "====== STEP 5: 5-seed ensemble for dual_esmif ======"
for SEED in 42 123 456 789 1337; do
    echo "  --- Seed $SEED ---"
    python Megascale-fineTuning/pnas_train.py \
        --model_name v3_dual_esmif_seed${SEED} \
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
        --emb_type dual_esmif
    echo "  Seed $SEED done at $(date)"
done

echo ""
echo "============================================================"
echo "ALL v3 TRAINING COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "Check results with:"
echo "  grep 'Best Pearson' logs/v3_clean.log"
echo "  grep 'PCC:' logs/v3_clean.log | tail -20"
