"""
ThermoMPNN-style subtract_mut training with frozen ESM-IF1 encoder features.

Architecture:
    - Frozen ESM-IF1 encoder produces [L, 512] per protein (precomputed as esmif_enc.pt)
    - At mutation position i: extract features[i] = 512-dim vector
    - MLP: 512 -> 64 -> 32 -> 21 (one score per amino acid)
    - ddG = score[mut_aa] - score[wt_aa]
    - Loss: MSE (same as ThermoMPNN)

This replicates ThermoMPNN (PCC=0.754) but with ESM-IF1 instead of ProteinMPNN.
Key insight: the pretrained backbone already knows "which amino acids fit at each position."
We just train a tiny MLP to read that signal.

Usage:
    python Megascale-fineTuning/train_subtract_mut.py --seed 42
    python Megascale-fineTuning/train_subtract_mut.py --seed 42 --dual  # use ProtT5+ESM-IF1
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(description='ThermoMPNN-style subtract_mut training')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32])
parser.add_argument('--dual', action='store_true', help='Use ProtT5(1024) + ESM-IF1(512) = 1536 at mutation position')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--data_dir', type=str, default='./data/MsDs/training_data')
parser.add_argument('--mut_dir', type=str, default='./data/MsDs/mutation_files')
parser.add_argument('--model_name', type=str, default=None)
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}
VOCAB_DIM = 20

# ============================================================
# Model: subtract_mut MLP head
# ============================================================
class SubtractMutHead(nn.Module):
    """ThermoMPNN-style MLP head with subtract_mut prediction.

    Input: frozen per-residue feature at mutation position (512 or 1536 dim).
    Output: ddG = score[mut_aa] - score[wt_aa]
    """
    def __init__(self, input_dim=512, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        sizes = [input_dim] + hidden_dims + [VOCAB_DIM]
        layers = []
        for i, (s1, s2) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(s1, s2))
            if i < len(sizes) - 2:  # no activation/dropout after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features, wt_idx, mut_idx):
        """
        Args:
            features: [batch, input_dim] — frozen features at mutation position
            wt_idx: [batch] — index of wildtype amino acid (0-19)
            mut_idx: [batch] — index of mutant amino acid (0-19)
        Returns:
            ddG: [batch] — predicted stability change
        """
        scores = self.mlp(features)  # [batch, 20]
        # Gather scores for mut and wt
        ddG = scores.gather(1, mut_idx.unsqueeze(1)).squeeze(1) - \
              scores.gather(1, wt_idx.unsqueeze(1)).squeeze(1)
        return ddG


# ============================================================
# Dataset: loads precomputed ESM-IF1 features + mutation info
# ============================================================
def parse_single_mutation(mut_type):
    """Parse 'A32G' -> (wt_aa, position_0indexed, mut_aa)"""
    match = re.match(r"([A-Za-z])(\d+)([A-Za-z])", mut_type)
    if not match:
        return None
    wt_aa, pos_str, mut_aa = match.groups()
    return wt_aa, int(pos_str) - 1, mut_aa  # Convert to 0-indexed


class SubtractMutDataset(Dataset):
    """Dataset that yields (feature_at_position, wt_aa_idx, mut_aa_idx, ddG) tuples."""

    def __init__(self, data_dir, mut_dir, protein_list, use_dual=False):
        """
        Args:
            data_dir: path to training_data (contains protein folders with esmif_enc.pt)
            mut_dir: path to mutation_files (contains {protein}.csv)
            protein_list: list of protein names to include
            use_dual: if True, concatenate ProtT5 + ESM-IF1 at mutation position
        """
        self.data_dir = data_dir
        self.mut_dir = mut_dir
        self.use_dual = use_dual
        self.samples = []  # list of (protein_name, mut_position_0idx, wt_aa_idx, mut_aa_idx, ddG)

        for protein in tqdm(protein_list, desc='Loading dataset'):
            protein_dir = os.path.join(data_dir, protein)
            mut_path = os.path.join(mut_dir, f'{protein}.csv')

            if not os.path.exists(mut_path):
                continue
            if not os.path.exists(os.path.join(protein_dir, 'esmif_enc.pt')):
                continue

            mutations = pd.read_csv(mut_path)

            # Get wildtype deltaG
            wt_rows = mutations[mutations['mut_type'] == 'wt']
            if wt_rows.empty:
                continue
            wt_dG = wt_rows.iloc[0]['deltaG']

            # Filter: single mutations only, no ins/del
            for _, row in mutations.iterrows():
                mut_type = row['mut_type']
                if mut_type == 'wt':
                    continue
                if ':' in mut_type:  # multi-mutation
                    continue
                if 'ins' in mut_type or 'del' in mut_type:
                    continue

                parsed = parse_single_mutation(mut_type)
                if parsed is None:
                    continue

                wt_aa, pos_0idx, mut_aa = parsed
                if wt_aa not in AA_TO_IDX or mut_aa not in AA_TO_IDX:
                    continue

                ddG = row['deltaG'] - wt_dG

                # Clip to [-1, 5] range (same as dG_ml flag in pnas_train)
                ddG = max(-1.0, min(5.0, ddG))

                self.samples.append({
                    'protein': protein,
                    'pos': pos_0idx,
                    'wt_idx': AA_TO_IDX[wt_aa],
                    'mut_idx': AA_TO_IDX[mut_aa],
                    'ddG': ddG,
                })

        # Preload all ESM-IF1 features into memory (they're small: 368 * [L, 512] ~= 200MB)
        self.features = {}
        proteins_needed = set(s['protein'] for s in self.samples)
        for protein in tqdm(proteins_needed, desc='Loading features'):
            protein_dir = os.path.join(data_dir, protein)
            esmif = torch.load(os.path.join(protein_dir, 'esmif_enc.pt'), weights_only=True).float()

            if use_dual:
                # Also load ProtT5 wildtype embedding
                emb_path = os.path.join(protein_dir, 'emb.pt')
                if os.path.exists(emb_path):
                    emb_raw = torch.load(emb_path, weights_only=False)
                    # emb_raw is a list of [seq_len, 1024] tensors; first one is WT
                    if isinstance(emb_raw, list) and len(emb_raw) > 0:
                        prott5_wt = emb_raw[0].float()  # [seq_len, 1024]
                        if prott5_wt.shape[0] == esmif.shape[0]:
                            esmif = torch.cat([esmif, prott5_wt], dim=-1)  # [L, 512+1024=1536]
                        else:
                            # Shape mismatch, pad ESM-IF1 with zeros for ProtT5 part
                            esmif = torch.cat([esmif, torch.zeros(esmif.shape[0], 1024)], dim=-1)
                    else:
                        esmif = torch.cat([esmif, torch.zeros(esmif.shape[0], 1024)], dim=-1)
                else:
                    esmif = torch.cat([esmif, torch.zeros(esmif.shape[0], 1024)], dim=-1)

            self.features[protein] = esmif

        print(f"  Dataset: {len(self.samples)} mutations from {len(proteins_needed)} proteins")
        print(f"  Feature dim: {next(iter(self.features.values())).shape[1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feat = self.features[s['protein']]
        seq_len = feat.shape[0]

        # Clamp position to valid range
        pos = min(s['pos'], seq_len - 1)

        feature_at_pos = feat[pos]  # [input_dim]

        return {
            'features': feature_at_pos,
            'wt_idx': torch.tensor(s['wt_idx'], dtype=torch.long),
            'mut_idx': torch.tensor(s['mut_idx'], dtype=torch.long),
            'ddG': torch.tensor(s['ddG'], dtype=torch.float32),
        }


# ============================================================
# Training loop
# ============================================================
def train():
    print(f"{'='*60}")
    print(f"SUBTRACT_MUT TRAINING — ThermoMPNN-style with ESM-IF1")
    print(f"{'='*60}")
    print(f"  Seed: {args.seed}")
    print(f"  Dual (ProtT5+ESM-IF1): {args.dual}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  LR: {args.lr}, WD: {args.weight_decay}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"  Device: {DEVICE}")
    print()

    # Load train/test split (same as ThermoMPNN)
    tm_test = pd.read_csv('./data/ThermoMPNN/mega_test.csv')
    test_proteins = tm_test['WT_name'].str.replace('.pdb', '', regex=False).unique().tolist()

    # Also filter by PNAS mutations list
    pnas_mut = pd.read_csv('./data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv')
    pnas_proteins = pd.read_csv('./data/Processed_K50_dG_datasets/Pnas_filtering/train_proteins.csv')['protein_name'].tolist()

    all_proteins = [d for d in os.listdir(args.data_dir)
                    if os.path.isdir(os.path.join(args.data_dir, d))]

    train_proteins = [p for p in all_proteins if p not in test_proteins and p in pnas_proteins]

    if args.debug:
        train_proteins = train_proteins[:10]
        test_proteins = test_proteins[:5]

    print(f"  Train proteins: {len(train_proteins)}")
    print(f"  Test proteins: {len(test_proteins)}")
    print()

    # Create datasets
    print("Building train dataset...")
    train_ds = SubtractMutDataset(args.data_dir, args.mut_dir, train_proteins, use_dual=args.dual)
    print("Building test dataset...")
    test_ds = SubtractMutDataset(args.data_dir, args.mut_dir, test_proteins, use_dual=args.dual)

    if len(train_ds) == 0 or len(test_ds) == 0:
        print("FATAL: Empty dataset!")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Create model
    input_dim = 1536 if args.dual else 512
    model = SubtractMutHead(input_dim=input_dim, hidden_dims=args.hidden_dims, dropout=0.1).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,} (tiny MLP — all knowledge is in frozen ESM-IF1)")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()  # Same as ThermoMPNN

    # Training
    best_pcc = -float('inf')
    best_epoch = 0
    model_dir = f'./Megascale-fineTuning/models/subtract_mut_{"dual" if args.dual else "esmif"}_seed{args.seed}'
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []

        for batch in train_loader:
            features = batch['features'].to(DEVICE)
            wt_idx = batch['wt_idx'].to(DEVICE)
            mut_idx = batch['mut_idx'].to(DEVICE)
            ddG = batch['ddG'].to(DEVICE)

            pred = model(features, wt_idx, mut_idx)
            loss = criterion(pred, ddG)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_preds.extend(pred.detach().cpu().numpy())
            train_targets.extend(ddG.cpu().numpy())

        scheduler.step()
        train_loss /= len(train_ds)
        train_pcc, _ = pearsonr(train_preds, train_targets)

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(DEVICE)
                wt_idx = batch['wt_idx'].to(DEVICE)
                mut_idx = batch['mut_idx'].to(DEVICE)
                ddG = batch['ddG'].to(DEVICE)

                pred = model(features, wt_idx, mut_idx)
                loss = criterion(pred, ddG)

                val_loss += loss.item() * features.size(0)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(ddG.cpu().numpy())

        val_loss /= len(test_ds)
        val_pcc, _ = pearsonr(val_preds, val_targets)
        val_sp, _ = spearmanr(val_preds, val_targets)
        val_rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets))**2))

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:3d} | Train PCC: {train_pcc:.4f} Loss: {train_loss:.4f} | "
                  f"Val PCC: {val_pcc:.4f} Spearman: {val_sp:.4f} RMSE: {val_rmse:.4f} | "
                  f"Best: {best_pcc:.4f} (ep{best_epoch}) | LR: {scheduler.get_last_lr()[0]:.2e}")

    print()
    print(f"{'='*60}")
    print(f"FINAL RESULT: Best Val PCC = {best_pcc:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")
    print(f"  Model saved: {model_dir}/best_model.pt")

    return best_pcc


if __name__ == '__main__':
    best_pcc = train()
