import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from data_loaders.dg_data_loader import AllProteinValidationDataset
from data_loaders.utils import load_checkpoint
from model.hydro_net import PEM
from model.model_cfg import CFG
from supervised_model.training.train_single_protein import train_all_single_proteins
from supervised_model.utils import load_config
from supervised_model.vaildation.validation import evaluate_mutations

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = load_config()

parser = argparse.ArgumentParser(description="Run model with supervised data.")
parser.add_argument("root_data_dir", type=str, nargs="?", default="../data/Processed_K50_dG_datasets",
                    help="Path to the root data directory.")
parser.add_argument("--mode", type=str, default="evaluation",
                    choices=["train_dataset", "train_single_proteins", "evaluation"],
                    help="Mode for running the model (default: evaluation).")

args = parser.parse_args()


def run_model_with_supervised_data(root_data_dir, mode='evaluation'):
    tensor_root_dir = Path(root_data_dir) / 'training_data'
    mutations_root_dir = Path(root_data_dir) / 'mutation_datasets'
    protein_dataset = AllProteinValidationDataset(tensor_root_dir, mutations_root_dir)
    if mode == 'evaluation':
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef).to(CFG.device)
        model = load_checkpoint(model, device, config.training.pretrained_ckpt_path)
        data_loader = DataLoader(protein_dataset, batch_size=1, shuffle=False)
        evaluate_mutations(model, data_loader, root_data_dir, config.training.pretrained_ckpt_path)
    elif mode == 'train_dataset':
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef).to(CFG.device)
        model = load_checkpoint(model, device, CFG.model_path)
        ...
    elif mode == 'train_single_proteins':
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef).to(CFG.device)
        model = load_checkpoint(model, device, config.training.pretrained_ckpt_path)
        optimizer = optim.Adam(model.parameters(), lr=config.optimizer.lr)
        data_loader = DataLoader(protein_dataset, batch_size=1, shuffle=False)
        train_all_single_proteins(model, optimizer, data_loader)


if __name__ == '__main__':
    args = parser.parse_args()

    run_model_with_supervised_data(args.root_data_dir, mode=args.mode)
