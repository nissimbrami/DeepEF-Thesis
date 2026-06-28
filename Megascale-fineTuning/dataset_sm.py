"""
GNN-SM Dataset: Wildtype-only graphs with mutation metadata.

Key difference from new_dataset.py:
- Loads ONLY the wildtype structure/embedding per protein (1 graph, not 200+)
- Returns mutation metadata: list of (position, wt_aa_idx, mut_aa_idx, ddG)
- This enables subtract-mut scoring: ddG = score[pos, mut_aa] - score[pos, wt_aa]

Usage:
    from dataset_sm import SMDataset
    ds = SMDataset(data_dir, mut_dir, train=True)
    batch = ds[0]
    # batch['coords'] = [L, 4, 3]
    # batch['one_hot_wt'] = [L, 20]  (wildtype sequence)
    # batch['emb_wt'] = [L, 1024]  (wildtype ProtT5 embedding)
    # batch['mask'] = [L]
    # batch['mutations'] = list of dicts with pos, wt_idx, mut_idx, ddG
"""

import os
import sys
import re
sys.path.append('./')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from train_utils import get_one_hot

# Constants - same paths as pnas_train.py
TM_PATH = "./data/ThermoMPNN/mega_test.csv"
TM_TRAIN_PATH = "./data/ThermoMPNN/mega_train.csv"
PNAS_PROTEINS = "./data/Processed_K50_dG_datasets/Pnas_filtering/train_proteins.csv"
PNAS_MUT = "./data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv"

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}

# Module-level config (set by training script before dataset creation)
EMB_TYPE = 'prott5'
DG_ML = True
ONE_MUT = True
DS_TYPE = 'pnas'
DEBUG = False
FULL_MEGASCALE = False  # When True, skip PNAS mutation filtering → use ALL mutations


def parse_single_mutation(mut_type):
    """Parse 'A32G' -> (wt_aa, position_0indexed, mut_aa) or None if invalid."""
    match = re.match(r"([A-Za-z])(\d+)([A-Za-z])", mut_type.strip())
    if not match:
        return None
    wt_aa, pos_str, mut_aa = match.groups()
    wt_aa, mut_aa = wt_aa.upper(), mut_aa.upper()
    if wt_aa not in AA_TO_IDX or mut_aa not in AA_TO_IDX:
        return None
    return wt_aa, int(pos_str) - 1, mut_aa  # 0-indexed position


class SMDataset(Dataset):
    """Wildtype-only dataset for GNN subtract-mut training.

    Each item is one protein with:
    - WT coordinates, one-hot, embedding, mask (for building 1 graph)
    - List of all mutations with their ddG values
    """

    def __init__(self, tensor_root_dir, mutations_root_dir, train=True):
        self.tensor_root_dir = tensor_root_dir
        self.mutations_root_dir = mutations_root_dir
        self.train = train

        # Get all protein directories (skip hidden folders like .cache)
        all_proteins = [p for p in os.listdir(tensor_root_dir)
                        if os.path.isdir(os.path.join(tensor_root_dir, p))
                        and not p.startswith('.')]

        # Test proteins from ThermoMPNN benchmark
        tm_df = pd.read_csv(TM_PATH)
        test_protein_names = tm_df['WT_name'].str.replace('.pdb', '', regex=False).unique().tolist()
        self.test_proteins = [p for p in all_proteins if p in test_protein_names]
        train_candidates = [p for p in all_proteins if p not in test_protein_names]

        # Filter by PNAS protein list (removes homologs)
        if DS_TYPE == 'pnas' and not FULL_MEGASCALE:
            pnas_proteins = pd.read_csv(PNAS_PROTEINS)['protein_name'].tolist()
            # Also remove proteins in mega_train (homologs to test set)
            tm_train = pd.read_csv(TM_TRAIN_PATH)['WT_name'].str.replace('.pdb', '', regex=False).unique().tolist()
            train_candidates = [p for p in train_candidates
                                if p in pnas_proteins and p not in tm_train]
        elif FULL_MEGASCALE:
            # Full MegaScale: use ALL proteins EXCEPT test proteins
            # Note: mega_train.csv contains the ThermoMPNN TRAINING split (safe to use)
            # Only exclude the 28 test proteins (already excluded above via test_protein_names)
            train_candidates = train_candidates  # no additional filtering needed

        self.train_proteins = train_candidates
        self.pnas_mutations = pd.read_csv(PNAS_MUT) if not FULL_MEGASCALE else None
        self.test_mutations = pd.read_csv(TM_PATH)

        if DEBUG:
            self.train_proteins = self.train_proteins[:5]
            self.test_proteins = self.test_proteins[:3]

        self.proteins = self.train_proteins if train else self.test_proteins
        print(f"SMDataset ({'train' if train else 'test'}): {len(self.proteins)} proteins")

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein_name = self.proteins[idx]
        protein_dir = os.path.join(self.tensor_root_dir, protein_name)
        mut_path = os.path.join(self.mutations_root_dir, f'{protein_name}.csv')

        # Load structure data
        coords = torch.load(os.path.join(protein_dir, 'coords.pt'), weights_only=False)  # [L, 4, 3]
        mask = torch.load(os.path.join(protein_dir, 'mask.pt'), weights_only=False)  # [L]
        delta_g_all = torch.load(os.path.join(protein_dir, 'deltaG.pt'), weights_only=False)

        # Load WT embedding based on EMB_TYPE
        emb_wt = self._load_wt_embedding(protein_dir, coords.shape[0])

        # Load mutations CSV
        mutations_df = pd.read_csv(mut_path)

        # Get wildtype dG (first row)
        wt_dG = delta_g_all[0].item()

        # Get WT one-hot from first sequence
        wt_seq = mutations_df.iloc[0]['aa_seq']
        one_hot_wt = get_one_hot(wt_seq)  # [L, 20]

        # Parse valid single-point mutations
        mutations = self._parse_mutations(mutations_df, protein_name, wt_dG, coords.shape[0])

        return {
            'name': protein_name,
            'coords': coords,           # [L, 4, 3]
            'one_hot_wt': one_hot_wt,   # [L, 20]
            'emb_wt': emb_wt,           # [L, emb_dim]
            'mask': mask,               # [L]
            'mutations': mutations,     # list of dicts
        }

    def _load_wt_embedding(self, protein_dir, seq_len):
        """Load wildtype embedding. Returns [L, emb_dim] tensor."""
        if EMB_TYPE == 'esmif_enc':
            path = os.path.join(protein_dir, 'esmif_enc.pt')
            if os.path.exists(path):
                emb = torch.load(path, weights_only=True).float()
                return emb[:seq_len] if emb.shape[0] >= seq_len else emb
            # Fallback to ProtT5
            return self._load_prott5_wt(protein_dir, seq_len)

        elif EMB_TYPE == 'dual_esmif':
            prott5 = self._load_prott5_wt(protein_dir, seq_len)  # [L, 1024]
            esmif_path = os.path.join(protein_dir, 'esmif_enc.pt')
            if os.path.exists(esmif_path):
                esmif = torch.load(esmif_path, weights_only=True).float()[:seq_len]
                return torch.cat([prott5, esmif], dim=-1)  # [L, 1536]
            return torch.cat([prott5, torch.zeros(seq_len, 512)], dim=-1)

        elif EMB_TYPE == 'proteinmpnn':
            path = os.path.join(protein_dir, 'proteinmpnn_feat.pt')
            if os.path.exists(path):
                emb = torch.load(path, weights_only=True).float()
                return emb[:seq_len] if emb.shape[0] >= seq_len else emb
            # Fallback to ProtT5
            print(f"  WARNING: proteinmpnn_feat.pt not found in {protein_dir}, falling back to ProtT5")
            return self._load_prott5_wt(protein_dir, seq_len)

        elif EMB_TYPE == 'dual_proteinmpnn':
            prott5 = self._load_prott5_wt(protein_dir, seq_len)  # [L, 1024]
            mpnn_path = os.path.join(protein_dir, 'proteinmpnn_feat.pt')
            if os.path.exists(mpnn_path):
                mpnn = torch.load(mpnn_path, weights_only=True).float()[:seq_len]
                return torch.cat([prott5, mpnn], dim=-1)  # [L, 1408]
            return torch.cat([prott5, torch.zeros(seq_len, 384)], dim=-1)

        else:
            # Default: ProtT5 (1024-dim)
            return self._load_prott5_wt(protein_dir, seq_len)

    def _load_prott5_wt(self, protein_dir, seq_len):
        """Load WT ProtT5 embedding from emb.pt (first entry)."""
        emb_path = os.path.join(protein_dir, 'emb.pt')
        if not os.path.exists(emb_path):
            return torch.zeros(seq_len, 1024)
        emb_raw = torch.load(emb_path, weights_only=False)
        if isinstance(emb_raw, list):
            # List of [L, 1024] tensors, first is WT
            wt_emb = emb_raw[0].float() if len(emb_raw) > 0 else torch.zeros(seq_len, 1024)
        elif isinstance(emb_raw, torch.Tensor):
            if emb_raw.dim() == 3:
                wt_emb = emb_raw[0].float()  # [n_muts, L, 1024] -> [L, 1024]
            elif emb_raw.dim() == 2:
                wt_emb = emb_raw.float()  # Already [L, 1024]
            else:
                wt_emb = torch.zeros(seq_len, 1024)
        else:
            wt_emb = torch.zeros(seq_len, 1024)
        return wt_emb[:seq_len] if wt_emb.shape[0] >= seq_len else wt_emb

    def _parse_mutations(self, mutations_df, protein_name, wt_dG, seq_len):
        """Parse mutations CSV into list of {pos, wt_idx, mut_idx, ddG} dicts."""
        mutations = []

        # Filter by PNAS mutations list if in pnas mode (skip if full_megascale)
        if self.train and DS_TYPE == 'pnas' and not FULL_MEGASCALE and self.pnas_mutations is not None:
            valid_names = set(self.pnas_mutations['name'].tolist())
        elif not self.train and DS_TYPE in ('pnas', 'deepef1'):
            valid_names = set(self.test_mutations['name'].tolist())
        else:
            valid_names = None

        for _, row in mutations_df.iterrows():
            mut_type = row['mut_type']

            # Skip wildtype row
            if mut_type == 'wt':
                continue
            # Skip insertions/deletions
            if 'ins' in str(mut_type) or 'del' in str(mut_type):
                continue
            # Skip multi-mutations if ONE_MUT
            if ONE_MUT and ':' in str(mut_type):
                continue
            # Skip if not in PNAS filter
            if valid_names is not None and row.get('name', '') not in valid_names:
                continue

            # Parse mutation
            parsed = parse_single_mutation(mut_type)
            if parsed is None:
                continue

            wt_aa, pos_0idx, mut_aa = parsed

            # Validate position is within sequence
            if pos_0idx >= seq_len or pos_0idx < 0:
                continue

            # Compute ddG
            ddG = row['deltaG'] - wt_dG

            # Clip to [-1, 5] range (same as dG_ml)
            if DG_ML:
                ddG = max(-1.0, min(5.0, ddG))

            mutations.append({
                'pos': pos_0idx,
                'wt_idx': AA_TO_IDX[wt_aa],
                'mut_idx': AA_TO_IDX[mut_aa],
                'ddG': ddG,
            })

        return mutations


def sm_collate_fn(batch):
    """Custom collate: don't stack proteins (different lengths). Return as list."""
    return batch
