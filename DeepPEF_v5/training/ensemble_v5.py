"""v5 seed-ensemble evaluation.

Loads the best_model.pt from several seed runs, averages their predicted dG per mutation, and
reports the ensembled PCC. Ensembling reliably cancels per-seed noise (+0.02-0.04 typical).

Usage (from repo root, GPU machine):
    PYTHONPATH=DeepPEF_v5:. python DeepPEF_v5/training/ensemble_v5.py \
        --model_dirs Megascale-fineTuning/models/v5_rung5_seed42 \
                     Megascale-fineTuning/models/v5_rung5_seed1  ... \
        --length_norm --serial_fusion --use_knn_gat

This reuses the Trainer/validate machinery from pnas_train_v5 to build predictions per model,
then averages the per-mutation predictions across models before computing PCC.

IMPORTANT: the flags you pass here MUST match how the seeds were trained. This script injects
those flags into pnas_train_v5's own argparse (via sys.argv) BEFORE importing it, so the module
globals it relies on (DEVICE, USE_KNN, KNN_K, CFG.emb_input_dim, model constructor args) are set
consistently with the training run. Otherwise the GAT graph topology or embedding dim would differ
from training and PCC would be wrong (or load_state_dict would raise a size mismatch).
"""

import os
import sys
import argparse
import numpy as np
import torch

_V5_DIR = os.path.dirname(os.path.abspath(__file__))
_V5_ROOT = os.path.dirname(_V5_DIR)
sys.path.insert(0, _V5_DIR)
sys.path.insert(0, _V5_ROOT)


def parse():
    p = argparse.ArgumentParser(description='v5 seed ensemble evaluation')
    p.add_argument('--model_dirs', nargs='+', required=True,
                   help='dirs each containing best_model.pt')
    # These MUST mirror the training flags so the model architecture + graph match the checkpoints.
    p.add_argument('--length_norm', action='store_true')
    p.add_argument('--serial_fusion', action='store_true')
    p.add_argument('--use_knn_gat', action='store_true',
                   help='set if the seeds were trained with --use_knn_gat (recommended default)')
    p.add_argument('--use_learned_aa', action='store_true')
    p.add_argument('--emb_type', type=str, default='prott5')
    p.add_argument('--emb_projection', type=str, default='none')
    # Levers that CHANGE parameter shapes (A, C, E, F, G-mutation_delta) MUST match training so
    # load_state_dict works. B (rbf), D (flory + afrc) and G-antisymmetry are shape-preserving /
    # value-only / train-only, so they do NOT affect state_dict loading and are omitted here.
    p.add_argument('--energy_terms', type=int, default=1)            # A
    p.add_argument('--use_burial', action='store_true')              # C
    p.add_argument('--burial_radius', type=float, default=10.0)      # C
    p.add_argument('--denoise_weight', type=float, default=0.0)      # E
    p.add_argument('--gcn_span', type=int, default=1)                # F
    p.add_argument('--use_edge_features', action='store_true')       # F
    p.add_argument('--mutation_delta', action='store_true')          # G (shape-changing)
    return p.parse_args()


def main():
    args = parse()

    # Build the argv that pnas_train_v5's module-level argparse expects, mirroring the training run.
    # We do this BEFORE importing pnas_train_v5 so its top-level parse_args() succeeds and sets its
    # globals (DEVICE, USE_KNN, KNN_K) and CFG.emb_input_dim to match training.
    train_argv = ['pnas_train_v5.py',
                  '--dataset_type', 'pnas', '--no_pretrained',
                  '--one_mut', '--dg_ml',
                  '--loss_type', 'huber_rank', '--ranking_weight', '0.1',
                  '--mini_batch_size', '16',
                  '--emb_type', args.emb_type,
                  '--emb_projection', args.emb_projection,
                  '--model_name', 'v5_ensemble_eval']
    if args.use_knn_gat:
        train_argv.append('--use_knn_gat')
    if args.length_norm:
        train_argv.append('--length_norm')
    if args.serial_fusion:
        train_argv.append('--serial_fusion')
    if args.use_learned_aa:
        train_argv.append('--use_learned_aa')
    # Shape-changing levers (A, C, E, F): forward to pnas_train_v5 so CFG + PEM widths match training.
    if args.energy_terms != 1:
        train_argv += ['--energy_terms', str(args.energy_terms)]
    if args.use_burial:
        train_argv += ['--use_burial', '--burial_radius', str(args.burial_radius)]
    if args.denoise_weight > 0:
        train_argv += ['--denoise_weight', str(args.denoise_weight)]
    if args.gcn_span != 1:
        train_argv += ['--gcn_span', str(args.gcn_span)]
    if args.use_edge_features:
        train_argv.append('--use_edge_features')
    # Lever G mutation_delta changes node widths (fc1_gcn/fc1_gat/gnn dims), so it MUST be
    # forwarded for load_state_dict to match. antisymmetry_weight / flory_afrc are train-only /
    # value-only and do NOT affect the state_dict, so they are omitted here.
    if args.mutation_delta:
        train_argv.append('--mutation_delta')

    saved_argv = sys.argv
    sys.argv = train_argv
    try:
        import pnas_train_v5 as T
    finally:
        sys.argv = saved_argv

    from model.hydro_net_v5 import PEM
    from model.model_cfg_v5 import CFG
    import config_paths

    tensor_root = config_paths.tensor_root()
    mut_root = config_paths.mut_root()
    from new_dataset import MSDataset
    from torch.utils.data import DataLoader
    test_ds = DataLoader(MSDataset(tensor_root, mut_root, train=False), batch_size=1, shuffle=False)

    all_true = None
    preds_per_model = []
    for d in args.model_dirs:
        ckpt = os.path.join(d, 'best_model.pt')
        # Construct the model with the SAME args used at training (from the configured T.args)
        # so load_state_dict matches the checkpoint's parameter shapes.
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                    dropout_rate=CFG.dropout_rate, light_attention=True,
                    emb_projection=T.args.emb_projection,
                    gat_cutoff=12.0,
                    serial_fusion=args.serial_fusion,
                    use_learned_aa=args.use_learned_aa,
                    length_norm=args.length_norm).to(T.DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE))
        trainer = T.Trainer(model, test_ds, test_ds)
        _, _, val_df = trainer.validate(0, test=True)
        val_df = val_df.reset_index(drop=True)
        preds_per_model.append(val_df['pred_deltaG'].to_numpy())
        if all_true is None:
            all_true = val_df['deltaG'].to_numpy()

    mean_pred = np.mean(np.vstack(preds_per_model), axis=0)
    pcc = np.corrcoef(all_true, mean_pred)[0, 1]
    print(f'Ensemble of {len(args.model_dirs)} models | dG PCC: {pcc:.4f}')


if __name__ == '__main__':
    main()
