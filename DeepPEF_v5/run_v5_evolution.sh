#!/bin/bash
# DeepPEF v5 — Evolution ladder. Each rung adds ONE change to the winner of the previous rung.
# Run on the GPU machine from the REPO ROOT (so ./data/MsDs/... resolves).
# Copy DeepPEF_v5/model/*_v5.py and DeepPEF_v5/training/*.py so imports resolve, OR run this
# script which points python at the v5 training entrypoint.
#
# Memory rules baked in: mini_batch_size=16 (OOM-safe), WANDB disabled, cache cleanup.
set -e

export WANDB_MODE=disabled
export PYTHONPATH="$(pwd)/DeepPEF_v5:$(pwd)"

TRAIN=DeepPEF_v5/training/pnas_train_v5.py
COMMON="--dataset_type pnas --no_pretrained --one_mut --dg_ml \
  --loss_type huber_rank --ranking_weight 0.1 --use_knn_gat \
  --cosine_lr --lr_min 1e-6 --weight_decay 1e-5 --mini_batch_size 16 \
  --epochs 15 --seed 42"

# Clean known-bad caches (memory lesson)
rm -f ./data/MsDs/mutation_files/.cache.csv 2>/dev/null || true
rm -rf ./data/MsDs/training_data/.cache 2>/dev/null || true

echo "===== RUNG 0: baseline (reproduce ~0.5259) ====="
python $TRAIN $COMMON --model_name v5_rung0_baseline

echo "===== RUNG 1: + length_norm ====="
python $TRAIN $COMMON --length_norm --model_name v5_rung1_lengthnorm

echo "===== RUNG 2: + serial_fusion (on top of length_norm) ====="
python $TRAIN $COMMON --length_norm --serial_fusion --model_name v5_rung2_serialfusion

echo "===== RUNG 3: + correlation loss (pearson, weight 0.1) ====="
python $TRAIN $COMMON --length_norm --serial_fusion \
  --corr_weight 0.1 --corr_type pearson --model_name v5_rung3_corrloss

echo "===== RUNG 4: switch to dual_esmif embeddings (needs esmif_enc.pt per protein) ====="
echo "  Skipped unless esmif_enc.pt exists. Uncomment to run:"
echo "  python \$TRAIN \$COMMON --length_norm --serial_fusion --corr_weight 0.1 \\"
echo "    --emb_type dual_esmif --model_name v5_rung4_dualesmif"

echo "===== RUNG 5: 5-seed ensemble of the best rung (edit seeds/model_name) ====="
echo "  for s in 42 1 2 3 4; do python \$TRAIN \$COMMON --length_norm --serial_fusion \\"
echo "    --corr_weight 0.1 --seed \$s --model_name v5_rung5_seed\$s; done"

echo ""
echo "############################################################################"
echo "# PROPOSAL LEVER ABLATION (A–F): each rung adds ONE lever on top of BASELINE"
echo "# so its individual effect on PCC is isolated. Compare each vs RUNG 0."
echo "############################################################################"

echo "===== LEVER A: energy decomposition (K=4 terms) ====="
python $TRAIN $COMMON --energy_terms 4 --model_name v5_leverA_energy4

echo "===== LEVER B: RBF distance bank (16 centers) ====="
python $TRAIN $COMMON --rbf_centers 16 --model_name v5_leverB_rbf16

echo "===== LEVER C: burial / solvation feature ====="
python $TRAIN $COMMON --use_burial --burial_radius 10.0 --model_name v5_leverC_burial

echo "===== LEVER D: Flory unfolded reference (nu=0.5) ====="
python $TRAIN $COMMON --flory_unfolded --flory_nu 0.5 --model_name v5_leverD_flory

echo "===== LEVER E: denoising head (weight 0.1) ====="
python $TRAIN $COMMON --denoise_weight 0.1 --denoise_sigma 0.3 --denoise_prob 0.5 \
  --model_name v5_leverE_denoise

echo "===== LEVER F: edge features + extended GCN span (S=2) ====="
python $TRAIN $COMMON --gcn_span 2 --use_edge_features --model_name v5_leverF_edges

echo "All rungs + levers complete. Compare best_model PCC across v5_rung* and v5_lever* in Megascale-fineTuning/models/"
