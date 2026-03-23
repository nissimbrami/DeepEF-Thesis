import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.model_cfg import CFG
from supervised_model.utils import get_wt_data, normalize_batch
from train_utils import get_graph, get_unfolded_graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_naive_thermodynamics_model(model, batch, mini_batch=256):
    folded_energies_list = []
    unfolded_energies_list = []
    wt_index = [i for i, x in enumerate(batch['mutations']) if 'wt' in x][0]
    coords, mask, _, _ = get_wt_data(batch, wt_index)
    one_hot = batch['one_hot'].to(device)
    prott5_embedding = batch['prott5'].to(device)
    for i in range(0, batch['prott5'].size(1), mini_batch):
        one_hot_minibatch = one_hot[0, i: i + mini_batch]
        prott5_embedding_minibatch = prott5_embedding[0, i: i + mini_batch]
        folded_graph_minibatch = torch.stack(
            [get_graph(coords, one_hot_minibatch[i], prott5_embedding_minibatch[i], mask) for i in
             range(prott5_embedding_minibatch.size(0))])
        unfolded_graph_minibatch = torch.stack(
            [get_unfolded_graph(coords, one_hot_minibatch[i], prott5_embedding_minibatch[i], mask) for i in
             range(prott5_embedding_minibatch.size(0))])
        folded_energies = model(folded_graph_minibatch).cpu().numpy()
        unfolded_energies = model(unfolded_graph_minibatch).cpu().numpy()

        folded_energies_list.append(folded_energies)
        unfolded_energies_list.append(unfolded_energies)

    protein_mutations_energies = np.vstack((np.hstack(folded_energies_list),
                                            np.hstack(unfolded_energies_list)))
    energy_out_df = pd.DataFrame(
        np.vstack((np.array([x[0] for x in batch['mutations']]), protein_mutations_energies))).T
    energy_out_df.columns = ['mut_type', 'folded_energies', 'unfolded_energies']
    return energy_out_df


def evaluate_mutations(model, data_loader, root_dir, ckpt_path):
    model.eval()
    mutation_output_dir = Path(root_dir) / 'mutation_outputs' / ckpt_path
    os.makedirs(mutation_output_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch = normalize_batch(batch)
            energy_out_df = run_naive_thermodynamics_model(model, batch, mini_batch=32)
            out_file = mutation_output_dir / f"{batch['name'][0]}.csv"
            energy_out_df.to_csv(out_file, index=False)
