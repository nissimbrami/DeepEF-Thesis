import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf

from constants import NANO_TO_ANGSTROM
from train_utils import get_graph, get_unfolded_graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_config():
    file = 'configs/config.yaml'
    if os.path.exists(file):
        return OmegaConf.load(file)
    else:
        return OmegaConf.load(os.path.join('supervised_model', file))


def set_wandb_params(config, protein_name):
    wandb_conf = config.wandb_logger
    if wandb_conf.enabled:
        pretrained_model_name = Path(config.training.pretrained_ckpt_path).stem
        run = wandb.init(project=wandb_conf.project,
                         name=f'{wandb_conf.per_protein_run_prefix}_{protein_name}',
                         config={
                             "model_name": config.model.name,
                             "base_model": config.model.base_model,
                             "optimizer": config.optimizer.name,
                             "learning_rate": config.optimizer.lr,
                             "freeze_pretrained": config.training.freeze_pretrained,
                             "pretrained_ckpt": pretrained_model_name,
                             "batch_size": config.training.single_protein_batch_size,
                             "train_split": config.training.train_size
                         },
                         tags=[protein_name, pretrained_model_name]
                         )
        return run


def get_wt_data(batch, wt_index):
    coords = batch['coords'].to(device)
    mask = batch['masks'].to(device)
    wt_one_hot = batch['one_hot'].to(device)[:, wt_index, ...]
    wt_prott5_embedding = batch['prott5'].to(device)[:, wt_index, ...]

    return coords.squeeze(0), mask.squeeze(0), wt_one_hot.squeeze(0), wt_prott5_embedding.squeeze(0)


def normalize_batch(batch):
    batch['one_hot'] = batch['one_hot'][:, :, :, :-1]
    batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM
    # ['coords'] = batch['coords'][:, :, [0, 2, 1, 3], :]
    # batch['masks'] = batch['masks'].to(torch.int) ^ 1  # Xor to reverse current output

    return batch


def preprocess_for_pem(batch):
    coords = batch['coords'].squeeze()
    one_hot_minibatch = batch['one_hot'].squeeze()
    prott5_embedding_minibatch = batch['prott5'].squeeze()
    mask = batch['masks'].squeeze()
    folded_graph_minibatch = torch.stack(
        [get_graph(coords, one_hot_minibatch[i], prott5_embedding_minibatch[i], mask) for i in
         range(prott5_embedding_minibatch.size(0))])
    unfolded_graph_minibatch = torch.stack(
        [get_unfolded_graph(coords, one_hot_minibatch[i], prott5_embedding_minibatch[i], mask) for i in
         range(prott5_embedding_minibatch.size(0))])
    return folded_graph_minibatch, unfolded_graph_minibatch


def get_batch_preprocessing_function(base_model_type='CFG'):
    return preprocessing_mapping.get(base_model_type)


preprocessing_mapping = {
    'PEM': preprocess_for_pem
}
