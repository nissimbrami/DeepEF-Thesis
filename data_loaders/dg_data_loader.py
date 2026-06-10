import glob
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data_loaders.constants import COORDS, ONE_HOT, MASKS, DELTA_G, PROTT5_EMBEDDINGS

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

'''
Need three data loaders:
1. first one is for evaluation - get all as supervised_model.
2. Split each protein into train, test, supervised_model
3. Split into train/test/val the entire dataset.
'''


class AllProteinValidationDataset(Dataset):

    def __init__(self, tensor_root_dir, mutations_root_dir):
        self.tensor_root_dir = tensor_root_dir
        self.mutations_root_dir = mutations_root_dir
        self.protein_dirs = {i: protein for i, protein in enumerate(os.listdir(self.tensor_root_dir))}

    def __len__(self):
        return len(self.protein_dirs)

    def __getitem__(self, idx):
        protein_dir = os.path.join(self.tensor_root_dir, self.protein_dirs[idx])
        mutations_path = os.path.join(self.mutations_root_dir, f'{self.protein_dirs[idx]}.csv')
        mutations = pd.read_csv(mutations_path)
        mutations = mutations[~mutations['mut_type'].str.contains('ins|del')]
        # Load and preprocess the data for each protein
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS))
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G))
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS))
        one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT))
        embedding_tensor = self.load_embedding_tensor(os.path.join(protein_dir, PROTT5_EMBEDDINGS))

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

    def load_embedding_tensor(self, embeddings_dir):
        embeddings = []
        all_embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, 'prott5_embedding_*.pt')),
                                     key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        for filename in all_embedding_files:
            if filename.endswith('.pt'):
                embedding_tensor = torch.load(filename, map_location=device)
                embeddings.append(embedding_tensor)
        return torch.vstack(embeddings)


def test_protein(prot):
    for k, v in prot.items():
        try:
            print(f'{k}: {v.shape}')
        except AttributeError as e:
            print(f'{k}: {v[0]}')
    print()


def test_protein_by_idx(idx):
    prot0 = protein_dataset[idx]
    test_protein(prot0)


if __name__ == '__main__':
    tensor_root_dir = r'../data/Processed_K50_dG_datasets/training_data'
    mutations_root_dir = r'../data/Processed_K50_dG_datasets/mutation_datasets'
    protein_dataset = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                                  mutations_root_dir=mutations_root_dir)
    # test_protein_by_idx(idx=0)

    protein_dataloader = DataLoader(protein_dataset, batch_size=1, shuffle=False)
    for batch in protein_dataloader:
        test_protein(batch)
