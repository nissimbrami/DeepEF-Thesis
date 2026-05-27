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


class MSDataset(Dataset):

    def __init__(self, tensor_root_dir, mutations_root_dir, train=True, one_mut=True, emb_type='prott5'):
        self.tensor_root_dir = tensor_root_dir
        self.train = train
        self.mutations_root_dir = mutations_root_dir
        self.emb_type = emb_type  # 'prott5', 'esm2', or 'dual'
        self.protein_dirs = [protein for i, protein in enumerate(os.listdir(self.tensor_root_dir))]
        print(f"Found {len(self.protein_dirs)} proteins")
        self.one_mut = one_mut # remove the mutations with more than one mutation
        self.unstable_mut = UNSTABLE_MUT
        self.ds_type = DS_TYPE
        self.dG_ml = DG_ML
        # remove TM proteins (strip .pdb suffix so it matches directory names)
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

    def _load_embeddings(self, protein_dir, indexes, seq_len, raw_emb_list=None):
        """Return embedding tensor [n_muts, seq_len, dim] based on emb_type.

        Supported emb_type values and output dims:
          prott5        → 1024  (per-mutant ProtT5)
          esm2          → 1280  (WT-only ESM-2 650M)
          saprot        → 1280  (WT-only SaProt 650M, structure-aware)
          dual          → 2304  (prott5 + esm2)
          esm2_saprot   → 2560  (esm2 + saprot)
        """
        n = len(indexes)

        if self.emb_type in ('prott5', 'dual'):
            emb_list = raw_emb_list if raw_emb_list is not None else \
                torch.load(os.path.join(protein_dir, PROTT5_EMBEDDINGS), weights_only=False)
            emb_filtered = [e for e in emb_list if e.shape[0] == seq_len]
            prott5 = torch.stack(emb_filtered)[indexes]  # [n, seq_len, 1024]

        if self.emb_type in ('esm2', 'dual', 'esm2_saprot'):
            wt = torch.load(os.path.join(protein_dir, 'esm2_wt.pt'), weights_only=True)
            esm2 = wt.unsqueeze(0).expand(n, -1, -1).contiguous()  # [n, seq_len, 1280]

        if self.emb_type in ('saprot', 'esm2_saprot'):
            wt_s = torch.load(os.path.join(protein_dir, 'saprot_wt.pt'), weights_only=True)
            saprot = wt_s.unsqueeze(0).expand(n, -1, -1).contiguous()  # [n, seq_len, 1280]

        if self.emb_type == 'prott5':
            return prott5
        elif self.emb_type == 'esm2':
            return esm2
        elif self.emb_type == 'saprot':
            return saprot
        elif self.emb_type == 'dual':
            return torch.cat([prott5, esm2.to(prott5.dtype)], dim=-1)   # [n, seq_len, 2304]
        else:  # esm2_saprot
            return torch.cat([esm2, saprot.to(esm2.dtype)], dim=-1)      # [n, seq_len, 2560]

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
        # one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT))
        raw_emb_list = torch.load(os.path.join(protein_dir, PROTT5_EMBEDDINGS), weights_only=False) \
            if self.emb_type in ('prott5', 'dual') else None

        # remove the ins and del mutations
        ins_del_mask = torch.ones(delta_g_tensor.shape[0], dtype=bool)
        ins_del_mask[ins_del_index.tolist()] = False

        mutations = mutations.drop(ins_del_index).reset_index(drop=True)
        delta_g_tensor = delta_g_tensor[ins_del_mask]
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
            indexes -= set(mutations[~mutations['name'].isin(self.test_mutations['name'])].index)

        # Ensure iloc[0] is not removed by adding it as the first index for ddg calc
        indexes.add(0)
        indexes = list(indexes)

        mutations = mutations.loc[indexes]
        delta_g_tensor = delta_g_tensor[indexes]
        one_hot_tensor = one_hot_tensor[indexes]
        embedding_tensor = self._load_embeddings(protein_dir, indexes, coords_tensor.shape[0], raw_emb_list)

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
        raw_emb_list = torch.load(os.path.join(protein_dir, PROTT5_EMBEDDINGS), weights_only=False) \
            if self.emb_type in ('prott5', 'dual') else None

        # remove the ins and del mutations
        ins_del_mask = torch.ones(delta_g_tensor.shape[0], dtype=bool)
        ins_del_mask[ins_del_index.tolist()] = False

        mutations = mutations.drop(ins_del_index).reset_index(drop=True)
        delta_g_tensor = delta_g_tensor[ins_del_mask]
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
            indexes -= set(mutations[~mutations['name'].isin(self.pnas_mutations['name'])].index)

        # Ensure iloc[0] is not removed by adding it as the first index for ddg calc
        indexes.add(0)
        indexes = list(indexes)

        mutations = mutations.loc[indexes]
        delta_g_tensor = delta_g_tensor[indexes]
        one_hot_tensor = one_hot_tensor[indexes]
        embedding_tensor = self._load_embeddings(protein_dir, indexes, coords_tensor.shape[0], raw_emb_list)

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