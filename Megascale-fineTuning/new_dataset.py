import os
import sys
sys.path.append('./')
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from train_utils import get_one_hot


NANO_TO_ANGSTROM = 0.1
# Constants
COORDS = 'coords.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'emb.pt'
VAL_RATIO = 0.2
RANDOM_SEED = 42
NANO_TO_ANGSTROM = 0.1
TM_PATH = "./data/ThermoMPNN/mega_test.csv"
TM_TRAIN_PATH = "./data/ThermoMPNN/mega_train.csv"
TENSOR_STORE = './data/MsDs/training_data'
MUTATIONS_STORE = './data/MsDs/mutation_files'
PNAS_PROTEINS = "./data/Processed_K50_dG_datasets/Pnas_filtering/train_proteins.csv"
PNAS_MUT = "./data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv"
DEBUG = False
ONE_MUT = True
DG_ML = True
DS_TYPE = 'pnas'
UNSTABLE_MUT = True
# Embedding type: 'prott5' (default), 'esmif_enc', 'dual_esmif' (prott5+esmif_enc)
EMB_TYPE = 'prott5'


class MSDataset(Dataset):

    def __init__(self, tensor_root_dir, mutations_root_dir, train  = True, one_mut = True):
        self.tensor_root_dir = tensor_root_dir
        self.train = train
        self.mutations_root_dir = mutations_root_dir
        self.protein_dirs = [protein for i, protein in enumerate(os.listdir(self.tensor_root_dir))]
        print(f"Found {len(self.protein_dirs)} proteins")
        self.one_mut = one_mut # remove the mutations with more than one mutation
        self.unstable_mut = UNSTABLE_MUT
        self.ds_type = DS_TYPE
        self.dG_ml = DG_ML
        # remove TM proteins
        tm_proteins = pd.read_csv(TM_PATH)
        tm_proteins = tm_proteins['WT_name'].str.replace('.pdb', '', regex=False).unique().tolist()
        # Fix: populate test_protein before filtering protein_dirs
        self.test_protein = [protein for protein in self.protein_dirs if protein in tm_proteins]
        self.protein_dirs = [protein for protein in self.protein_dirs if protein not in tm_proteins]
        print(f"Found {len(self.test_protein)} test proteins")
        print(f"Found {len(self.protein_dirs)} training proteins")
        self.pnas_mutations = pd.read_csv(PNAS_MUT)
        self.test_mutations = pd.read_csv(TM_PATH)
        if DEBUG:
            self.protein_dirs = self.protein_dirs[:5]
        self.remove_homologs()

    def __len__(self):
        if self.train:
            return len(self.protein_dirs)
        else:
            return len(self.test_protein)

    def __getitem__(self, idx):
        if self.train:
            return self.load_protein_data(idx)
        else:
            return self.load_test_protein_data(idx)
   
    def remove_homologs(self):
        """Remove the homologs from the training set"""
        if self.train:  # Only remove homologs for training data
            pnas_proteins = pd.read_csv(TM_TRAIN_PATH)['WT_name'].str.replace('.pdb', '', regex=False).unique()
            self.protein_dirs = [protein for protein in self.protein_dirs if protein not in pnas_proteins]
     
    def _load_and_concat_embeddings(self, protein_dir, coords_tensor, embedding_tensor_raw):
        """Load embeddings based on EMB_TYPE and optionally concatenate ESM-IF1 encoder features.

        Returns: embedding_tensor [n_muts, seq_len, emb_dim]
        """
        seq_len = coords_tensor.shape[0]

        if EMB_TYPE == 'esmif_enc':
            # Use ONLY ESM-IF1 encoder features (512-dim, same for all mutations)
            esmif_path = os.path.join(protein_dir, 'esmif_enc.pt')
            if not os.path.exists(esmif_path):
                # Fallback to ProtT5 if ESM-IF1 features not yet generated
                emb_list = [emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len]
                return torch.stack(emb_list) if emb_list else None
            esmif_enc = torch.load(esmif_path, weights_only=True).float()  # [L, 512]
            if esmif_enc.shape[0] != seq_len:
                esmif_enc = esmif_enc[:seq_len]
            # Expand to all mutations (same structural features for each)
            n_muts = len([emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len])
            return esmif_enc.unsqueeze(0).expand(n_muts, -1, -1).clone()

        elif EMB_TYPE == 'dual_esmif':
            # Concatenate ProtT5 (1024) + ESM-IF1 encoder (512) = 1536-dim
            emb_list = [emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len]
            if not emb_list:
                return None
            prott5_tensor = torch.stack(emb_list)  # [n_muts, L, 1024]

            esmif_path = os.path.join(protein_dir, 'esmif_enc.pt')
            if not os.path.exists(esmif_path):
                # Fallback: pad with zeros where ESM-IF1 features missing
                zeros = torch.zeros(prott5_tensor.shape[0], seq_len, 512)
                return torch.cat([prott5_tensor, zeros], dim=-1)

            esmif_enc = torch.load(esmif_path, weights_only=True).float()  # [L, 512]
            if esmif_enc.shape[0] != seq_len:
                esmif_enc = esmif_enc[:seq_len]
            # Expand ESM-IF1 to match number of mutations
            esmif_expanded = esmif_enc.unsqueeze(0).expand(prott5_tensor.shape[0], -1, -1)
            # Concatenate: [n_muts, L, 1024+512=1536]
            return torch.cat([prott5_tensor, esmif_expanded], dim=-1)

        elif EMB_TYPE == 'saprot':
            # SaProt WT-only embeddings (1280-dim)
            saprot_path = os.path.join(protein_dir, 'saprot_wt.pt')
            if not os.path.exists(saprot_path):
                emb_list = [emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len]
                return torch.stack(emb_list) if emb_list else None
            saprot_wt = torch.load(saprot_path, weights_only=True).float()  # [L, 1280]
            if saprot_wt.shape[0] != seq_len:
                saprot_wt = saprot_wt[:seq_len]
            n_muts = len([emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len])
            return saprot_wt.unsqueeze(0).expand(n_muts, -1, -1).clone()

        elif EMB_TYPE == 'saprot_pm':
            # SaProt per-mutant embeddings (1280-dim, one per mutation)
            saprot_path = os.path.join(protein_dir, 'saprot_emb.pt')
            if not os.path.exists(saprot_path):
                emb_list = [emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len]
                return torch.stack(emb_list) if emb_list else None
            saprot_list = torch.load(saprot_path, weights_only=False)
            emb_list = [emb.float() for emb in saprot_list if emb.shape[0] == seq_len]
            return torch.stack(emb_list) if emb_list else None

        elif EMB_TYPE == 'dual_saprot_pm':
            # ProtT5 (1024) + SaProt per-mutant (1280) = 2304-dim
            prott5_list = [emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len]
            if not prott5_list:
                return None
            prott5_tensor = torch.stack(prott5_list)

            saprot_path = os.path.join(protein_dir, 'saprot_emb.pt')
            if not os.path.exists(saprot_path):
                zeros = torch.zeros(prott5_tensor.shape[0], seq_len, 1280)
                return torch.cat([prott5_tensor, zeros], dim=-1)
            saprot_list = torch.load(saprot_path, weights_only=False)
            saprot_embs = [emb.float() for emb in saprot_list if emb.shape[0] == seq_len]
            if not saprot_embs:
                zeros = torch.zeros(prott5_tensor.shape[0], seq_len, 1280)
                return torch.cat([prott5_tensor, zeros], dim=-1)
            saprot_tensor = torch.stack(saprot_embs)
            # Match lengths (take min of prott5 and saprot counts)
            n = min(prott5_tensor.shape[0], saprot_tensor.shape[0])
            return torch.cat([prott5_tensor[:n], saprot_tensor[:n]], dim=-1)

        else:
            # Default: ProtT5 only (1024-dim)
            emb_list = [emb for emb in embedding_tensor_raw if emb.shape[0] == seq_len]
            return torch.stack(emb_list) if emb_list else None

    def load_test_protein_data(self, idx):

        protein_dir = os.path.join(self.tensor_root_dir, self.test_protein[idx])
        mutations_path = os.path.join(self.mutations_root_dir, f'{self.test_protein[idx]}.csv')
        mutations = pd.read_csv(mutations_path)
        # Get ins and del mutations indexes:
        ins_del_index = mutations[mutations['mut_type'].str.contains('ins|del')].index

        # Load and preprocess the data for each protein
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS),weights_only=False)
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G),weights_only=False)
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS),weights_only=False)
        embedding_tensor_raw = torch.load(os.path.join(protein_dir, PROTT5_EMBEDDINGS),weights_only=False)

        # remove the ins and del mutations
        ins_del_mask = torch.ones(delta_g_tensor.shape[0], dtype=bool)
        ins_del_mask[ins_del_index.tolist()] = False

        # remove the ins and del mutations
        mutations = mutations.drop(ins_del_index).reset_index(drop=True)
        delta_g_tensor = delta_g_tensor[ins_del_mask]
        # Load embeddings based on EMB_TYPE
        embedding_tensor = self._load_and_concat_embeddings(protein_dir, coords_tensor, embedding_tensor_raw)
        if embedding_tensor is None:
            embedding_tensor_res = [emb for emb in embedding_tensor_raw if emb.shape[0] == coords_tensor.shape[0]]
            embedding_tensor = torch.stack(embedding_tensor_res)
        one_hot_tensor = torch.stack([get_one_hot(x) for x in mutations['aa_seq']])
        
        # If dG_ml is check save the threshold of -1 and 5
        if self.dG_ml:
            threshold = [-1.0, 5.0]
            delta_g_tensor = torch.where(delta_g_tensor > threshold[0], delta_g_tensor, threshold[0])
            delta_g_tensor = torch.where(delta_g_tensor < threshold[1], delta_g_tensor, threshold[1])
        
        indexes = set(mutations.index)
        # remove unstable mut
        if not self.unstable_mut:
            indexes -= set(mutations[mutations['ddG_ML'] == '-'].index)
             
        # remove the mutations with more than one mutation
        if self.one_mut:
            indexes -= set(mutations[mutations['mut_type'].str.contains(':')].index)
        
        if self.ds_type in ('pnas', 'deepef1'):
            # get the pnas mutations indexes
            indexes -= set(mutations[~mutations['name'].isin(self.test_mutations['name'])].index)
        
        # Ensure iloc[0] is not removed by adding it as the first index for ddg calc
        indexes.add(0)

        indexes = list(indexes)
        mutations = mutations.loc[indexes]
        delta_g_tensor = delta_g_tensor[indexes]
        one_hot_tensor = one_hot_tensor[indexes]
        embedding_tensor = embedding_tensor[indexes]
            
        mutations_data = {
            'name': self.test_protein[idx],
            'mutations': mutations['mut_type'].to_list(),
            'prott5': embedding_tensor,
            'coords': coords_tensor,
            'one_hot': one_hot_tensor,
            'delta_g': delta_g_tensor,
            'masks': mask_tensor
        }

        return mutations_data

    
    def load_protein_data(self, idx):
        protein_dir = os.path.join(self.tensor_root_dir, self.protein_dirs[idx])
        mutations_path = os.path.join(self.mutations_root_dir, f'{self.protein_dirs[idx]}.csv')
        mutations = pd.read_csv(mutations_path)
        # Get ins and del mutations indexes:
        ins_del_index = mutations[mutations['mut_type'].str.contains('ins|del')].index

        # Load and preprocess the data for each protein
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS),weights_only=False)
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G),weights_only=False)
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS),weights_only=False)
        embedding_tensor_raw = torch.load(os.path.join(protein_dir, PROTT5_EMBEDDINGS),weights_only=False)

        # remove the ins and del mutations
        ins_del_mask = torch.ones(delta_g_tensor.shape[0], dtype=bool)
        ins_del_mask[ins_del_index.tolist()] = False

        # remove the ins and del mutations
        mutations = mutations.drop(ins_del_index).reset_index(drop=True)
        delta_g_tensor = delta_g_tensor[ins_del_mask]
        # Load embeddings based on EMB_TYPE
        embedding_tensor = self._load_and_concat_embeddings(protein_dir, coords_tensor, embedding_tensor_raw)
        if embedding_tensor is None:
            embedding_tensor_res = [emb for emb in embedding_tensor_raw if emb.shape[0] == coords_tensor.shape[0]]
            embedding_tensor = torch.stack(embedding_tensor_res)
        one_hot_tensor = torch.stack([get_one_hot(x) for x in mutations['aa_seq']])
        
        # If dG_ml is check save the threshold of -1 and 5
        if self.dG_ml:
            threshold = [-1.0, 5.0]
            delta_g_tensor = torch.where(delta_g_tensor > threshold[0], delta_g_tensor, threshold[0])
            delta_g_tensor = torch.where(delta_g_tensor < threshold[1], delta_g_tensor, threshold[1])
        
        
        indexes = set(mutations.index)
        # remove unstable mut
        if not self.unstable_mut:
            indexes -= set(mutations[mutations['ddG_ML'] == '-'].index)
             
        # remove the mutations with more than one mutation
        if self.one_mut:
            indexes -= set(mutations[mutations['mut_type'].str.contains(':')].index)
        
        if self.ds_type == 'pnas':
            # get the pnas mutations indexes
            indexes -= set(mutations[~mutations['name'].isin(self.pnas_mutations['name'])].index)
        
        # Ensure iloc[0] is not removed by adding it as the first index for ddg calc
        indexes.add(0)
        indexes = list(indexes)
        
        mutations = mutations.loc[indexes]
        delta_g_tensor = delta_g_tensor[indexes]
        one_hot_tensor = one_hot_tensor[indexes]
        embedding_tensor = embedding_tensor[indexes]
            
        mutations_data = {
            'name': self.protein_dirs[idx],
            'mutations': mutations['mut_type'].to_list(),
            'prott5': embedding_tensor,
            'coords': coords_tensor,
            'one_hot': one_hot_tensor,
            'delta_g': delta_g_tensor,
            'masks': mask_tensor
        }

        return mutations_data

   
    
if __name__ == "__main__":
    Ds = MSDataset(TENSOR_STORE, MUTATIONS_STORE)
    # set dataloader
    dataloader = DataLoader(Ds, batch_size=1, shuffle=True)
    for batch in dataloader:
        # print(batch)
        break

    # --- Dataset statistics ---
    import collections
    total_mutations = 0
    total_insertions = 0
    total_deletions = 0
    real_proteins = 0
    engineered_proteins = 0
    protein_types = {}

    # Helper: classify protein as real or engineered (customize as needed)
    def classify_protein(name):
        # Example: if name starts with 'P' and is followed by digits, treat as real (e.g., UniProt IDs)
        import re
        if re.match(r"^P\\d+", name):
            return 'real'
        else:
            return 'engineered'

    for idx in range(len(Ds)):
        data = Ds.load_protein_data(idx) if Ds.train else Ds.load_test_protein_data(idx)
        muts = data['mutations']
        total_mutations += len(muts)
        # Count insertions and deletions
        for mut in muts:
            if 'ins' in mut:
                total_insertions += 1
            if 'del' in mut:
                total_deletions += 1
        # Classify protein
        ptype = classify_protein(data['name'])
        protein_types.setdefault(ptype, set()).add(data['name'])

    real_proteins = len(protein_types.get('real', set()))
    engineered_proteins = len(protein_types.get('engineered', set()))
    total_proteins = real_proteins + engineered_proteins

    print("\n--- Dataset Statistics ---")
    print(f"Total proteins: {total_proteins}")
    print(f"  Real proteins: {real_proteins}")
    print(f"  Engineered proteins: {engineered_proteins}")
    print(f"Total mutations (deltaG values): {total_mutations}")
    print(f"Insertions: {total_insertions}")
    print(f"Deletions: {total_deletions}")