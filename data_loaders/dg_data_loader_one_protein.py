from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

'''
Need three data loaders:
1. first one is for evaluation - get all as supervised_model.
2. Split each protein into train, test, supervised_model
3. Split into train/test/val the entire dataset.
'''


class ProteinMutationDataset(Dataset):

    def __init__(self, mutation_data: Dict, subindex_list=None):
        self.mutation_data = mutation_data
        self.mutation_data['index'] = list(range(len(self.mutation_data['mutations']))) \
            if not subindex_list else sorted(subindex_list)

        if subindex_list:
            self.__process_mutation_data()

    def __len__(self):
        return self.mutation_data['one_hot'].size(1)

    def __getitem__(self, idx):
        return {
            'index': idx,
            'name': self.mutation_data['name'],
            'mutations': [x for i, x in enumerate(self.mutation_data['mutations']) if i == idx],
            'prott5': self.mutation_data['prott5'][:, idx, ...],
            'coords': self.mutation_data['coords'],
            'one_hot': self.mutation_data['one_hot'][:, idx, ...],
            'delta_g': self.mutation_data['delta_g'][:, idx, ...],
            'masks': self.mutation_data['masks']
        }

    def __process_mutation_data(self):
        self.mutation_data['delta_g'] = self.mutation_data['delta_g'][:, self.mutation_data['index'], ...]
        self.mutation_data['mutations'] = np.array(self.mutation_data['mutations'])[self.mutation_data['index'], ...]
        self.mutation_data['prott5'] = self.mutation_data['prott5'][:, self.mutation_data['index'], ...]
        self.mutation_data['one_hot'] = self.mutation_data['one_hot'][:, self.mutation_data['index'], ...]


def custom_collate_fn(batch):
    names = [sample['name'] for sample in batch]
    mutations = [sample['mutations'] for sample in batch]

    # Stack tensors for other keys
    prott5 = torch.stack([sample['prott5'] for sample in batch]).swapaxes(0, 1)
    coords = batch[0]['coords']
    one_hot = torch.stack([sample['one_hot'] for sample in batch]).swapaxes(0, 1)
    delta_g = torch.stack([sample['delta_g'] for sample in batch]).swapaxes(0, 1)
    masks = batch[0]['masks']

    return {
        'name': names,
        'mutations': mutations,
        'prott5': prott5,
        'coords': coords,
        'one_hot': one_hot,
        'delta_g': delta_g,
        'masks': masks,
    }
