import os
from pathlib import Path
from statistics import mean

import wandb
from accelerate import Accelerator
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.dg_data_loader_one_protein import ProteinMutationDataset, custom_collate_fn
from supervised_model.training.criteria import criterion
from supervised_model.utils import normalize_batch, load_config, get_batch_preprocessing_function, set_wandb_params

config = load_config()
accelerator = Accelerator()


def mock_batch_print_shapes(batch):
    print("Batch Names:", batch['name'])
    print("Batch Mutations:", batch['mutations'])
    print("Batch prott5 Shape:", batch['prott5'].shape)
    print("Batch coords Shape:", batch['coords'].shape)
    print("Batch one_hot Shape:", batch['one_hot'].shape)
    print("Batch delta_g Shape:", batch['delta_g'].shape)
    print("Batch masks Shape:", batch['masks'].shape)
    print("-" * 40)


def preprocess_batch(batch):
    batch = normalize_batch(batch)
    preprocessing_function = get_batch_preprocessing_function(config.model.base_model)
    if preprocessing_function:
        inputs = preprocessing_function(batch)
    else:
        inputs = batch
    return inputs


def run_model_on_batch(batch, model, criterion, experiment_delta_gs):
    folded_graph_minibatch, unfolded_graph_minibatch = preprocess_batch(batch)
    folded_graph_minibatch.requires_grad = True
    unfolded_graph_minibatch.requires_grad = True
    folded_energies = model(folded_graph_minibatch)
    unfolded_energies = model(unfolded_graph_minibatch)
    loss_dict = criterion([folded_energies, unfolded_energies], experiment_delta_gs, folded_graph_minibatch)

    return loss_dict


def get_loss_weights(weights, loss_dict):
    if weights is None or weights.shape != len(loss_dict):
        weights = torch.ones(len(loss_dict)) / len(loss_dict)
    return torch.tensor(weights, requires_grad=True)


def train_batch(model, optimizer, criterion, batch, weights=None):
    experiment_delta_gs = batch['delta_g'].squeeze()
    optimizer.zero_grad()
    loss_dict = run_model_on_batch(batch, model, criterion, experiment_delta_gs)
    weights = get_loss_weights(weights, loss_dict)
    loss = weights @ torch.tensor(list(loss_dict.values()))
    accelerator.backward(loss)
    optimizer.step()
    logging_dict = {k: v.item() for k, v in loss_dict.items()}
    logging_dict['total_loss'] = loss.item()
    return logging_dict


def evaluate_single_protein(model, criterion, test_loader, epoch, weights=None):
    model.eval()
    test_loss = 0.0
    batch_losses = {}
    for i, batch in enumerate(tqdm(test_loader, leave=False), 0):
        model.zero_grad()
        experiment_delta_gs = batch['delta_g'].squeeze()
        loss_dict = run_model_on_batch(batch, model, criterion, experiment_delta_gs)
        weights = get_loss_weights(weights, loss_dict)

        loss = weights @ torch.tensor(list(loss_dict.values()))

        logging_dict = {k: v.item() for k, v in loss_dict.items()}
        logging_dict['total_loss'] = loss.item()
        batch_losses[i] = logging_dict

        test_loss += logging_dict['total_loss']
    mean_test_loss = test_loss / len(test_loader)
    print(f'Test loss: Epoch {epoch + 1}: {mean_test_loss}')
    loss_dict_for_logging = {key: mean(entry[key] for entry in batch_losses.values()) for key in batch_losses[0]}
    return loss_dict_for_logging


def log_loss_dict(loss_dict):
    if config.wandb_logger.enabled:
        wandb.log(loss_dict)


def train_epoch(model, criterion, optimizer, train_data_loader, epoch):
    model.train()
    running_loss = 0.0
    batch_losses = {}
    len_train_data_loader = len(train_data_loader)
    for i, batch in tqdm(enumerate(train_data_loader), total=len_train_data_loader):
        batch_loss_dict = train_batch(model, optimizer, criterion, batch)
        running_loss += batch_loss_dict['total_loss']
        batch_losses[i] = batch_loss_dict
    epoch_loss = running_loss / len_train_data_loader
    print(f'Train loss: epoch {epoch + 1}: {epoch_loss}')
    loss_dict_for_logging = {key: mean(entry[key] for entry in batch_losses.values()) for key in batch_losses[0]}
    return loss_dict_for_logging


def save_best_model(model, out_path):
    best_model_state = model.state_dict()
    if best_model_state is not None:
        torch.save(best_model_state, out_path)


def create_train_test_split(protein_dataset, random_state=42, split_mode='regular'):
    # fix for data leakage if similar mutations in train and test.
    if split_mode == 'regular' or split_mode is None:
        train_idx, test_val_idx = train_test_split(range(len(protein_dataset)),
                                                   test_size=1 - config.training.train_size,
                                                   random_state=random_state)
        test_idx, val_idx = train_test_split(test_val_idx, test_size=0.5, random_state=random_state)
        train_set = ProteinMutationDataset(protein_dataset.mutation_data.copy(), train_idx)
        test_set = ProteinMutationDataset(protein_dataset.mutation_data.copy(), test_val_idx)
        val_set = ProteinMutationDataset(protein_dataset.mutation_data.copy(), val_idx)
        return train_set, test_set, val_set
    elif split_mode == 'variant_type':
        return None
    elif split_mode == 'structural_arrangement':
        return None  # split mode by alpha_helix_beta_sheet


def set_training_false_on_feature_extractor(model):
    print('set training false on pretrained')
    for param in model.parameters():
        param.requires_grad = False
    print('changing training true on fc1 and fc2')
    for name, param in model.named_parameters():
        if name.split('.')[0] == 'fc1' or name.split('.')[0] == 'fc2':
            param.requires_grad = True


def train_single_protein(protein_name, model, optimizer, mutations_data, num_epochs):
    run = set_wandb_params(config, protein_name)
    with run:
        single_protein_dataset = ProteinMutationDataset(mutations_data)
        train_set, test_set, val_set = create_train_test_split(single_protein_dataset)
        best_test_loss = float('inf')
        if config.training.freeze_pretrained:
            set_training_false_on_feature_extractor(model)
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            train_data_loader = DataLoader(train_set, batch_size=config.training.single_protein_batch_size, shuffle=True,
                                           collate_fn=custom_collate_fn)
            test_data_loader = DataLoader(test_set, batch_size=config.training.single_protein_batch_size, shuffle=False,
                                          collate_fn=custom_collate_fn)
            model, optimizer = accelerator.prepare(model, optimizer)
            train_data_loader, test_data_loader = accelerator.prepare(train_data_loader, test_data_loader)
            train_losses = train_epoch(model, criterion, optimizer, train_data_loader, epoch)
            test_losses = evaluate_single_protein(model, criterion, test_data_loader, epoch)

            unified_epoch_losses = {"train": train_losses, "val": test_losses}
            log_loss_dict(unified_epoch_losses)

            curr_test_loss = test_losses['total_loss']
            if curr_test_loss < best_test_loss:
                print(f'curr_loss = {curr_test_loss}')
                best_test_loss = curr_test_loss
                model_path = Path(config.training.trained_ckpt_path) / config.model.name / protein_name / 'best_model.pth'
                os.makedirs(str(model_path.parent), exist_ok=True)
                save_best_model(model, str(model_path))
        run.detach()


def train_all_single_proteins(model, optimizer, data_loader):
    for mutations_data in iter(data_loader):
        train_single_protein(mutations_data['name'][0], model, optimizer, mutations_data, config.training.num_epochs)
        break
