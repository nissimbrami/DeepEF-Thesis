import os
import sys
sys.path.append('./')
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_utils import get_graph, get_unfolded_graph, load_checkpoint
from tqdm import tqdm
from dataset import AllProteinValidationDataset, normalize_batch


TRAINED_MODEL_PATH = "./models/AlphaFold2.pt"

def get_training_results(model,datapath,device):
    """Get the training results for the model."""
    model.eval()
    val_loss = 0
    val_dg = torch.tensor([],device=device)
    val_dg_pred = torch.tensor([],device=device)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(self.val_ds,desc=f'Validation Epoch: {epoch}')):
            batch = normalize_batch(batch, True)
            batch_loss = 0
            batch_idx = 1
            for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                batch_idx += 1
                output,u_energy,f_energy = self.get_deltaG(batch, j)
                delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                loss = self.criterion(output,delta_g)
                energys = torch.cat((u_energy,f_energy),dim=0)
                energy_reg = REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                loss += energy_reg
                batch_loss += loss.item()
                val_dg = torch.cat((val_dg, delta_g), dim=0)
                val_dg_pred = torch.cat((val_dg_pred, output), dim=0)
            batch_loss /= batch_idx
        val_loss += batch_loss
    val_loss /= len(self.val_ds)
    print(f'Validation Loss: {val_loss}')
    pc_corr = torch.corrcoef(torch.cat((val_dg[None,:],val_dg_pred[None,:])))[0, 1]
    wandb_log({'val_loss': val_loss,'epoch': epoch, 'pc_corr': pc_corr},run)
    self.model.train()
    return pc_corr