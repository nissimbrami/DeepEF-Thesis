#!/bin/bash
# ============================================================
# ThermoMPNN-style subtract_mut training with frozen ESM-IF1
# This is the REAL 90% plan — replicates what actually works.
#
# Run AFTER run_v3_clean.sh finishes (or in parallel — this is tiny)
#   setsid bash run_subtract_mut.sh > logs/subtract_mut.log 2>&1 &
# ============================================================

set -e
cd /home/nissimb/workspace/DeepPEF
source /home/nissimb/pytorch_env/bin/activate
export WANDB_MODE=disabled

echo "============================================================"
echo "ThermoMPNN-style SUBTRACT_MUT with frozen ESM-IF1 encoder"
echo "Started: $(date)"
echo "============================================================"
echo ""
echo "Architecture: ESM-IF1(512) -> MLP(64,32) -> 21-dim -> ddG = score[mut] - score[wt]"
echo "This is TINY (< 10K params) — trains in minutes, not hours."
echo "All protein knowledge comes from pretrained ESM-IF1 (142M params, frozen)."
echo ""

# Clear caches
find . -path './.git' -prune -o -name '__pycache__' -type d -print -exec rm -rf {} + 2>/dev/null || true

# Verify ESM-IF1 features exist
python -c "
import torch, os, sys
td = './data/MsDs/training_data'
proteins = [d for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
missing = [p for p in proteins if not os.path.exists(os.path.join(td, p, 'esmif_enc.pt'))]
if missing:
    print(f'ERROR: {len(missing)} proteins missing esmif_enc.pt')
    sys.exit(1)
print(f'ESM-IF1 features verified: {len(proteins)} proteins')
"
echo ""

# =============================================
# EXPERIMENT 1: ESM-IF1 only (512-dim) — 5 seeds
# =============================================
echo "====== EXPERIMENT 1: ESM-IF1 only (512-dim) — 5 seeds ======"
for SEED in 42 123 456 789 1337; do
    echo "  --- Seed $SEED ---"
    python Megascale-fineTuning/train_subtract_mut.py \
        --seed $SEED \
        --epochs 50 \
        --lr 1e-3 \
        --batch_size 256 \
        --hidden_dims 64 32 \
        --weight_decay 1e-4
    echo ""
done

echo ""
echo "====== EXPERIMENT 1 DONE ======"
echo ""

# =============================================
# EXPERIMENT 2: Dual (ProtT5 + ESM-IF1 = 1536-dim) — 5 seeds
# =============================================
echo "====== EXPERIMENT 2: Dual ProtT5+ESM-IF1 (1536-dim) — 5 seeds ======"
for SEED in 42 123 456 789 1337; do
    echo "  --- Seed $SEED ---"
    python Megascale-fineTuning/train_subtract_mut.py \
        --seed $SEED \
        --epochs 50 \
        --lr 1e-3 \
        --batch_size 256 \
        --hidden_dims 128 64 \
        --dual \
        --weight_decay 1e-4
    echo ""
done

echo ""
echo "====== EXPERIMENT 2 DONE ======"
echo ""

# =============================================
# EXPERIMENT 3: Larger MLP (for ESM-IF1 only) — 5 seeds
# =============================================
echo "====== EXPERIMENT 3: Larger MLP [128, 64, 32] — 5 seeds ======"
for SEED in 42 123 456 789 1337; do
    echo "  --- Seed $SEED ---"
    python Megascale-fineTuning/train_subtract_mut.py \
        --seed $SEED \
        --epochs 50 \
        --lr 5e-4 \
        --batch_size 256 \
        --hidden_dims 128 64 32 \
        --weight_decay 1e-4
    echo ""
done

echo ""
echo "============================================================"
echo "ALL SUBTRACT_MUT EXPERIMENTS DONE — $(date)"
echo "============================================================"
echo ""
echo "Results summary:"
grep "FINAL RESULT" logs/subtract_mut.log 2>/dev/null || echo "(check log for results)"
