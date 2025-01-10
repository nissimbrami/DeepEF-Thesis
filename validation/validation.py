import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.dg_data_loader import AllProteinValidationDataset
from data_loaders.utils import load_checkpoint
from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import get_graph, get_unfolded_graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NANO_TO_ANGSTROM = 0.1

def get_wt_data(batch, wt_index):
    coords = batch['coords'].to(device)
    mask = batch['masks'].to(device)
    wt_one_hot = batch['one_hot'].to(device)[:, wt_index, ...]
    wt_prott5_embedding = batch['prott5'].to(device)[:, wt_index, ...]

    return coords.squeeze(0), mask.squeeze(0), wt_one_hot.squeeze(0), wt_prott5_embedding.squeeze(0)


def normalize_batch(batch, LLM_EMB = True):
    batch['one_hot'] = batch['one_hot'][:, :, :, :-1]
    batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM
    if not LLM_EMB: # zero prot5 embedding
         batch['prott5'] = torch.zeros_like(batch['prott5'])
    return batch


def run_thermodynamics_model(model, batch, mini_batch=256,debug = False):
    folded_energies_list = []
    unfolded_energies_list = []
    wt_index = [i for i, x in enumerate(batch['mutations']) if 'wt' in x][0]
    coords, mask, _, _ = get_wt_data(batch, wt_index)
    one_hot = batch['one_hot'].to(device)
    prott5_embedding = batch['prott5'].to(device)
    number_of_mutations = one_hot.size(1) if not debug else 4
    batch['mutations'] = batch['mutations'][:number_of_mutations]
    mini_batch = min(mini_batch, number_of_mutations)
    for i in range(0, number_of_mutations, mini_batch):
        one_hot_minibatch = one_hot[0, i: i + mini_batch]
        prott5_embedding_minibatch = prott5_embedding[0, i: i + mini_batch]
        folded_graph_minibatch = torch.stack(
            [get_graph(coords, one_hot_minibatch[i], prott5_embedding_minibatch[i], mask) for i in
             range(prott5_embedding_minibatch.size(0))])
        unfolded_graph_minibatch = torch.stack(
            [get_unfolded_graph(coords, one_hot_minibatch[i], prott5_embedding_minibatch[i], mask) for i in
             range(prott5_embedding_minibatch.size(0))])
        mini_batch_mutation = one_hot_minibatch.size(0)
        all_proteins = torch.cat((folded_graph_minibatch, unfolded_graph_minibatch), dim=0)
        with torch.no_grad():
            energys = model(all_proteins).cpu().numpy()

        folded_energies_list.append(energys[:mini_batch_mutation])
        unfolded_energies_list.append(energys[mini_batch_mutation:])

    protein_mutations_energies = np.vstack((np.hstack(folded_energies_list),
                                            np.hstack(unfolded_energies_list)))
    energy_out_df = pd.DataFrame(
        np.vstack((np.array([x[0] for x in batch['mutations']]), protein_mutations_energies))).T
    energy_out_df.columns = ['mut_type', 'folded_energies', 'unfolded_energies']
    return energy_out_df


def evaluate_mutations(model, data_loader, root_dir,model_path = CFG.model_path,debug = CFG.debug, mini_batch_size=256):
    model.eval()
    # model.train()
    mutation_output_dir = Path(root_dir) / 'mutation_outputs' / os.path.join(model_path.split('/')[-2],model_path.split('/')[-1]) # model_path.split('/')[-2]
    os.makedirs(mutation_output_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch = normalize_batch(batch, True)
            energy_out_df = run_thermodynamics_model(model = model, batch = batch, debug = debug, mini_batch = mini_batch_size)
            out_file = mutation_output_dir / f"{batch['name'][0]}.csv"
            energy_out_df.to_csv(out_file, index=False)
            if debug:
                break


def run_validation(root_dir, mode='evaluation', model_path=CFG.model_path, mini_batch_size = 256,model=None,debug = CFG.debug):
    tensor_root_dir = Path(root_dir) / 'training_data'
    mutations_root_dir = Path(root_dir) / 'mutation_datasets'
    protein_dataset = AllProteinValidationDataset(tensor_root_dir, mutations_root_dir)
    if mode == 'evaluation':
        if model is None:
            model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,light_attention=CFG.light_attention).to(CFG.device)
            try:
                model.load_state_dict(torch.load(model_path)['model_state_dict'])
            except:
                model.load_state_dict(torch.load(model_path))
        data_loader = DataLoader(protein_dataset, batch_size=1, shuffle=False)
        evaluate_mutations(model, data_loader, root_dir,model_path, mini_batch_size =mini_batch_size,debug=debug)
    elif mode == 'split_single_protein':
        pass
    elif mode == 'split_whole_set':
        pass
