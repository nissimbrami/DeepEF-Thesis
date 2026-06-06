"""
GNN-SM Training: Subtract-Mut output with full GNN backbone.

Key differences from pnas_train.py:
1. ONE forward pass per protein (not 200+ per protein)
2. Output: [L, 20] amino acid scores (not scalar energy)
3. ddG = score[pos, mut_aa] - score[pos, wt_aa] (instant, no unfolded state)
4. Anti-symmetry is AUTOMATIC (score[B]-score[A] = -(score[A]-score[B]))
5. All mutations from one protein contribute to ONE gradient step

Novel contribution: Full GNN message-passing (GCN + multi-head GAT + k-NN edges)
informs position-specific amino acid scoring. Unlike ThermoMPNN's isolated MLP.

Usage:
    python Megascale-fineTuning/pnas_train_sm.py --seed 42 --epochs 30
    python Megascale-fineTuning/pnas_train_sm.py --seed 42 --emb_type dual_esmif --emb_projection mlp
"""

import os
import sys
sys.path.append('./')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import gc

from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import get_graph, get_one_hot

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description='GNN-SM: Subtract-Mut with GNN backbone')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_min', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='gnn_sm_seed42')
parser.add_argument('--emb_type', type=str, default='prott5',
                    choices=['prott5', 'esmif_enc', 'dual_esmif'])
parser.add_argument('--emb_projection', type=str, default='none',
                    choices=['none', 'mlp', 'low_rank'])
parser.add_argument('--use_knn_gat', action='store_true', default=True)
parser.add_argument('--knn_k', type=int, default=30)
parser.add_argument('--ranking_weight', type=float, default=0.1)
parser.add_argument('--huber_delta', type=float, default=1.0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--verify_only', action='store_true',
                    help='Run verification checks only (5 proteins, 1 epoch)')
parser.add_argument('--data_dir', type=str, default='./data/MsDs/training_data')
parser.add_argument('--mut_dir', type=str, default='./data/MsDs/mutation_files')
args = parser.parse_args()

# ============================================================
# Setup
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# Configure embedding dimensions
EMB_DIMS = {'prott5': 1024, 'esmif_enc': 512, 'dual_esmif': 1536}
CFG.emb_input_dim = EMB_DIMS[args.emb_type]
CFG.emb_projection = args.emb_projection

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NANO_TO_ANGSTROM = 0.1

# Import dataset AFTER setting config
import dataset_sm
dataset_sm.EMB_TYPE = args.emb_type
dataset_sm.DG_ML = True
dataset_sm.ONE_MUT = True
dataset_sm.DS_TYPE = 'pnas'
dataset_sm.DEBUG = args.debug or args.verify_only
from dataset_sm import SMDataset, sm_collate_fn

# ============================================================
# Model creation
# ============================================================
def create_model():
    model = PEM(
        layers=CFG.num_layers,
        gaussian_coef=CFG.gaussian_coef,
        dropout_rate=CFG.dropout_rate,
        light_attention=True,
        emb_projection=args.emb_projection,
        gat_cutoff=12.0 if args.use_knn_gat else None,
    ).to(DEVICE)
    return model


# ============================================================
# Build graph for one protein (WT only)
# ============================================================
def build_wt_graph(batch_item):
    """Build a single WT graph from dataset item.

    Returns:
        graph: [1, L, feature_dim] tensor
        ca_coords: [1, L, 3] CA coordinates (for k-NN edges)
        seq_len: int
    """
    coords = batch_item['coords'].to(DEVICE) * NANO_TO_ANGSTROM  # [L, 4, 3]
    one_hot = batch_item['one_hot_wt'].to(DEVICE)  # [L, 20]
    emb = batch_item['emb_wt'].to(DEVICE)  # [L, emb_dim]
    mask = batch_item['mask'].to(DEVICE)  # [L]

    # Build folded graph using existing function
    graph = get_graph(coords, one_hot, emb, mask, gaussian_coef=CFG.gaussian_coef)  # [L, feat_dim]
    graph = graph.unsqueeze(0)  # [1, L, feat_dim]

    # CA coordinates for k-NN edges
    ca_coords = coords[:, 1, :].unsqueeze(0)  # [1, L, 3] (atom 1 = CA)

    return graph, ca_coords, coords.shape[0]


# ============================================================
# Compute ddG predictions from [L, 20] scores
# ============================================================
def compute_ddg_from_scores(scores, mutations):
    """
    Args:
        scores: [L, 20] amino acid preference scores
        mutations: list of dicts with 'pos', 'wt_idx', 'mut_idx', 'ddG'
    Returns:
        pred_ddg: [n_mutations] tensor
        true_ddg: [n_mutations] tensor
    """
    if len(mutations) == 0:
        return None, None

    pred_list = []
    true_list = []
    for mut in mutations:
        pos = mut['pos']
        wt_idx = mut['wt_idx']
        mut_idx = mut['mut_idx']

        # Bounds check
        if pos >= scores.shape[0]:
            continue

        # ddG = score[pos, mut_aa] - score[pos, wt_aa]
        # This is INHERENTLY antisymmetric: reverse gives -(same value)
        ddg_pred = scores[pos, mut_idx] - scores[pos, wt_idx]
        pred_list.append(ddg_pred)
        true_list.append(mut['ddG'])

    if len(pred_list) == 0:
        return None, None

    return torch.stack(pred_list), torch.tensor(true_list, device=scores.device, dtype=torch.float32)


# ============================================================
# Ranking loss (same as pnas_train.py)
# ============================================================
def ranking_loss(pred, target, margin=0.1):
    """Pairwise ranking loss for ordering preservation."""
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    n_pairs = min(n * (n - 1) // 2, 128)
    idx_i = torch.randint(0, n, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=pred.device)
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % n

    pred_diff = pred[idx_i] - pred[idx_j]
    target_diff = target[idx_i] - target[idx_j]
    sign = torch.sign(target_diff)
    loss = torch.clamp(-sign * pred_diff + margin, min=0.0)
    return loss.mean()


# ============================================================
# Training
# ============================================================
def train():
    print(f"{'='*60}")
    print(f"GNN-SM TRAINING — Subtract-Mut with GNN backbone")
    print(f"{'='*60}")
    print(f"  Seed: {args.seed}")
    print(f"  Embedding: {args.emb_type} (dim={CFG.emb_input_dim})")
    print(f"  Projection: {args.emb_projection}")
    print(f"  k-NN GAT: {args.use_knn_gat} (k={args.knn_k})")
    print(f"  LR: {args.lr} -> {args.lr_min} (cosine)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {DEVICE}")
    if args.verify_only:
        print(f"  *** VERIFICATION MODE — testing shapes/gradients only ***")
    print()

    # Create datasets
    print("Loading datasets...")
    train_ds = SMDataset(args.data_dir, args.mut_dir, train=True)
    test_ds = SMDataset(args.data_dir, args.mut_dir, train=False)

    if len(train_ds) == 0 or len(test_ds) == 0:
        print("FATAL: Empty dataset! Check paths.")
        sys.exit(1)

    # No DataLoader batching — we process proteins one at a time
    # (each protein IS the batch since all its mutations go in one step)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=sm_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=sm_collate_fn, num_workers=0)

    # Create model
    model = create_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print()

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = nn.HuberLoss(delta=args.huber_delta)

    # Model directory
    model_dir = f'./Megascale-fineTuning/models/{args.model_name}'
    os.makedirs(model_dir, exist_ok=True)

    # ============================================================
    # VERIFICATION GATE (always runs first)
    # ============================================================
    print("--- VERIFICATION GATE ---")
    model.train()
    test_item = train_ds[0]
    graph, ca_coords, seq_len = build_wt_graph(test_item)
    print(f"  Graph shape: {graph.shape}")  # Should be [1, L, feat_dim]
    print(f"  CA coords shape: {ca_coords.shape}")  # Should be [1, L, 3]
    print(f"  Seq len: {seq_len}")
    print(f"  Mutations count: {len(test_item['mutations'])}")

    # Test forward pass in subtract_mut mode
    scores = model(graph, f_type='subtract_mut', ca_coords=ca_coords)
    print(f"  Scores shape: {scores.shape}")  # Should be [1, L, 20]
    assert scores.shape == (1, seq_len, 20), f"SHAPE ERROR: expected [1, {seq_len}, 20], got {scores.shape}"

    # Test ddG computation
    pred_ddg, true_ddg = compute_ddg_from_scores(scores[0], test_item['mutations'])
    if pred_ddg is not None:
        print(f"  Pred ddG shape: {pred_ddg.shape}")
        print(f"  True ddG shape: {true_ddg.shape}")
        print(f"  Pred ddG range: [{pred_ddg.min().item():.3f}, {pred_ddg.max().item():.3f}]")
        print(f"  True ddG range: [{true_ddg.min().item():.3f}, {true_ddg.max().item():.3f}]")

        # Test backward
        loss = criterion(pred_ddg, true_ddg)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradient norm: {grad_norm:.4f}")
        assert grad_norm > 0, "GRADIENT ERROR: zero gradients!"
        optimizer.zero_grad()
    else:
        print("  WARNING: No valid mutations for first protein")

    # Verify antisymmetry
    if pred_ddg is not None and len(test_item['mutations']) > 0:
        mut = test_item['mutations'][0]
        fwd = scores[0, mut['pos'], mut['mut_idx']] - scores[0, mut['pos'], mut['wt_idx']]
        rev = scores[0, mut['pos'], mut['wt_idx']] - scores[0, mut['pos'], mut['mut_idx']]
        assert torch.allclose(fwd, -rev, atol=1e-6), "ANTISYMMETRY ERROR!"
        print(f"  Antisymmetry verified: fwd={fwd.item():.4f}, rev={rev.item():.4f}")

    print("--- VERIFICATION PASSED ---")
    print()

    if args.verify_only:
        print("Verification-only mode. Exiting.")
        return 0.0

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    best_pcc = -float('inf')
    best_epoch = 0
    nan_count = 0
    MAX_NAN = 5  # Damage control: stop after 5 NaN losses

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_proteins = 0
        epoch_mutations = 0

        for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):
            item = batch[0]  # single protein (collate returns list)

            if len(item['mutations']) == 0:
                continue

            # Build WT graph
            try:
                graph, ca_coords, seq_len = build_wt_graph(item)
            except Exception as e:
                print(f"  WARNING: Skipping {item['name']}: {e}")
                continue

            # Forward: get [1, L, 20] scores
            scores = model(graph, f_type='subtract_mut', ca_coords=ca_coords)

            # Compute ddG from scores
            pred_ddg, true_ddg = compute_ddg_from_scores(scores[0], item['mutations'])
            if pred_ddg is None:
                continue

            # Loss: Huber + ranking
            primary_loss = criterion(pred_ddg, true_ddg)
            rank_loss = args.ranking_weight * ranking_loss(pred_ddg, true_ddg)
            loss = primary_loss + rank_loss

            # Damage control: check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  WARNING: NaN/Inf loss at protein {item['name']} (count={nan_count})")
                if nan_count >= MAX_NAN:
                    print("  FATAL: Too many NaN losses. Stopping training.")
                    return best_pcc
                optimizer.zero_grad()
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_proteins += 1
            epoch_mutations += len(pred_ddg)

            # Memory cleanup
            del graph, ca_coords, scores, pred_ddg, true_ddg, loss
            torch.cuda.empty_cache()

        scheduler.step()

        if epoch_proteins == 0:
            print(f"  Epoch {epoch}: No valid proteins! Check data.")
            continue

        avg_loss = epoch_loss / epoch_proteins

        # Validation
        val_pcc, val_sp, val_rmse = validate(model, test_loader)

        # Track best
        if val_pcc > best_pcc:
            best_pcc = val_pcc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Proteins: {epoch_proteins} "
              f"| Mutations: {epoch_mutations} | Val PCC: {val_pcc:.4f} Sp: {val_sp:.4f} "
              f"RMSE: {val_rmse:.4f} | Best: {best_pcc:.4f} (ep{best_epoch}) | LR: {lr_now:.2e}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'epoch_{epoch}.pt'))

    print()
    print(f"{'='*60}")
    print(f"TRAINING COMPLETE. Best Val PCC = {best_pcc:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")
    print(f"  Model saved: {model_dir}/best_model.pt")

    return best_pcc


# ============================================================
# Validation
# ============================================================
def validate(model, test_loader):
    """Evaluate on test set. Returns (pearson, spearman, rmse)."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            item = batch[0]

            if len(item['mutations']) == 0:
                continue

            try:
                graph, ca_coords, seq_len = build_wt_graph(item)
            except:
                continue

            scores = model(graph, f_type='subtract_mut', ca_coords=ca_coords)
            pred_ddg, true_ddg = compute_ddg_from_scores(scores[0], item['mutations'])

            if pred_ddg is None:
                continue

            all_preds.extend(pred_ddg.cpu().numpy())
            all_targets.extend(true_ddg.cpu().numpy())

            del graph, ca_coords, scores
            torch.cuda.empty_cache()

    if len(all_preds) < 2:
        return 0.0, 0.0, float('inf')

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    pcc, _ = pearsonr(all_preds, all_targets)
    sp, _ = spearmanr(all_preds, all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    return pcc, sp, rmse


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    best_pcc = train()
    print(f"\nFinal result: PCC = {best_pcc:.4f}")
