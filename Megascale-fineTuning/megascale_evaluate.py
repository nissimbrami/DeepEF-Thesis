# Evaluate results on the whole megascale dataset (no training)
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import load_checkpoint
from new_dataset import MSDataset
from pnas_train import Trainer, LIGHT_ATTENTION, DEVICE, PRETRAINED, TRAINED_MODEL_PATH, MODEL_NAME, DROP_OUT
import argparse

def evaluate_and_save(trainer, ds, split_name):
    pc_corr, val_loss, val_df = trainer.validate(epoch=0, test=True)
    out_csv = f"{MODEL_NAME}_{split_name}_metrics.csv"
    val_df.to_csv(out_csv, index=False)
    print(f"Saved {split_name} metrics to {out_csv}")
    # Print overall metrics
    print(f"Overall {split_name} Pearson correlation: {pc_corr:.4f}, Loss: {val_loss:.4f}")
    # Per-protein correlation
    per_protein_corr = []
    for protein in val_df['protein'].unique():
        df_p = val_df[val_df['protein'] == protein]
        if len(df_p) > 1:
            try:
                corr = pd.Series(df_p['deltaG']).corr(pd.Series(df_p['pred_deltaG']))
            except Exception:
                corr = float('nan')
        else:
            corr = float('nan')
        per_protein_corr.append({'protein': protein, 'pearson_corr': corr})
        print(f"{split_name} protein: {protein}, Pearson correlation: {corr:.4f}")
    # Save per-protein correlations
    per_protein_df = pd.DataFrame(per_protein_corr)
    per_protein_csv = f"{MODEL_NAME}_{split_name}_per_protein_corr.csv"
    per_protein_df.to_csv(per_protein_csv, index=False)
    print(f"Saved per-protein correlations to {per_protein_csv}")
    return pc_corr, val_loss

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on megascale dataset')
    parser.add_argument('--tensor_root_dir', type=str, default='./data/MsDs/training_data', help='Tensor root directory')
    parser.add_argument('--mutations_root_dir', type=str, default='./data/MsDs/mutation_files', help='Mutations root directory')
    parser.add_argument('--trained_model_path', type=str, default=TRAINED_MODEL_PATH, help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    args = parser.parse_args()

    CFG.dropout_rate = DROP_OUT
    # Load datasets
    train_ds = MSDataset(tensor_root_dir=args.tensor_root_dir, mutations_root_dir=args.mutations_root_dir, train=True)
    val_ds = MSDataset(tensor_root_dir=args.tensor_root_dir, mutations_root_dir=args.mutations_root_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    # Combine all data
    all_ds = [*train_ds.protein_dirs, *val_ds.test_protein]
    # Model
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate, light_attention=LIGHT_ATTENTION).to(DEVICE)
    if PRETRAINED:
        try:
            model, _, _, _, _ = load_checkpoint(args.trained_model_path, model)
        except:
            model.load_state_dict(torch.load(args.trained_model_path))
    # Evaluate on train
    print('Evaluating on training data...')
    trainer_train = Trainer(model, train_loader, train_loader)
    evaluate_and_save(trainer_train, train_loader, 'train')
    # Evaluate on validation
    print('Evaluating on validation data...')
    trainer_val = Trainer(model, val_loader, val_loader)
    evaluate_and_save(trainer_val, val_loader, 'val')
    # Evaluate on all data
    print('Evaluating on all data...')
    all_ds_obj = MSDataset(tensor_root_dir=args.tensor_root_dir, mutations_root_dir=args.mutations_root_dir, train=True)
    all_ds_obj.protein_dirs = all_ds
    all_loader = DataLoader(all_ds_obj, batch_size=args.batch_size, shuffle=False)
    trainer_all = Trainer(model, all_loader, all_loader)
    evaluate_and_save(trainer_all, all_loader, 'all')

if __name__ == '__main__':
    main()
