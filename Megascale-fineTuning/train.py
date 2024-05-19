# fine tuning the existing model with mega-scale data
# Path: Megascale-fineTuning/train.py

import os
import sys
sys.path.append('./')
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import get_graph, get_unfolded_graph




COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'prott5_embeddings'
VAL_RATIO = 0.2
RANDOM_SEED = 42
NANO_TO_ANGSTROM = 0.1
DEBUG = False
EPOCHS = 50 if not DEBUG else 1
FREEZE_LAYERS = True

def normalize_batch(batch, LLM_EMB = True):
    batch['one_hot'] = batch['one_hot'][:, :, :, :-1]
    batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM
    if not LLM_EMB: # zero prot5 embedding
         batch['prott5'] = torch.zeros_like(batch['prott5'])
    return batch    

class AllProteinValidationDataset(Dataset):

    def __init__(self, tensor_root_dir, mutations_root_dir, train  = True):
        self.tensor_root_dir = tensor_root_dir
        self.mutations_root_dir = mutations_root_dir
        self.protein_dirs = [protein for i, protein in enumerate(os.listdir(self.tensor_root_dir))]
        if DEBUG:
            self.protein_dirs = self.protein_dirs[:2]
        # Train test split
        self.training_protein, self.val_proteins = train_test_split(self.protein_dirs, test_size=VAL_RATIO, random_state=RANDOM_SEED)
        if train:
            self.protein_dirs = self.training_protein

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
                embedding_tensor = torch.load(filename)
                embeddings.append(embedding_tensor)
        return torch.vstack(embeddings)


# Trainer class

class Trainer():
    def __init__(self, model, train_ds, val_ds, device = 'cuda'):
        self.model = model.to(device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.to(self.device)
        self.mini_batch_size = 256

    def train(self, epochs = 10):
        # Freeze the layers and only train the last layer
        if FREEZE_LAYERS:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc2.parameters():
                param.requires_grad = True
        running_loss = 0
        for epoch in range(epochs):
            self.model.train()
            for i, batch in enumerate(self.train_ds):
                batch = normalize_batch(batch, True)
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    self.optimizer.zero_grad()
                    output = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                    loss = self.criterion(output, delta_g)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    
            
            self.validate()

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_ds):
                batch = normalize_batch(batch, True)
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    output = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                    loss = self.criterion(output,delta_g)
                    val_loss += loss.item()
        val_loss /= len(self.val_ds)
        print(f'Validation Loss: {val_loss}')
        self.model.train()
        
    def get_deltaG(self, batch, i):
        # move all to the same device
        one_hot_minibatch = batch['one_hot'][0, i: i + self.mini_batch_size].to(self.device)
        prott5_embedding_minibatch = batch['prott5'][0, i: i + self.mini_batch_size].to(self.device)
        batch['coords'] = batch['coords'].to(self.device)
        batch['masks'] = batch['masks'].to(self.device)
        # get the graph
        folded_graph_minibatch = torch.stack(
            [get_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(), prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])
        unfolded_graph_minibatch = torch.stack(
            [get_unfolded_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(), prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])
      

        folded_energy = self.model(folded_graph_minibatch)
        unfolded_energy = self.model(unfolded_graph_minibatch)
        
        return unfolded_energy - folded_energy

if __name__ == '__main__':
    tensor_root_dir = r'./data/Processed_K50_dG_datasets/training_data'
    mutations_root_dir = r'./data/Processed_K50_dG_datasets/mutation_datasets'
    protein_train = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                                  mutations_root_dir=mutations_root_dir, train=True)
    protein_val = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                                  mutations_root_dir=mutations_root_dir, train=False)
    # test_protein_by_idx(idx=0)

    train_ds = DataLoader(protein_train, batch_size=1, shuffle=False)
    val_ds = DataLoader(protein_val, batch_size=1, shuffle=False)
    
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef)
    
    trainer = Trainer(model, train_ds, val_ds)
    
    trainer.train(epochs = 50)
    