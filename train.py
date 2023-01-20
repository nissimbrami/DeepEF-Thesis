from model.data_loader import PEFDataset,fetch_dataloader
from model.model_cfg import CFG
from model.net import ProteinEnergyNet
import torch
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc
import time

# define one epoch train
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (seq, id, coordsAlpha,coordsBeta, coordsC, coordsCa, coordsN, 
             coordsAlpha_native, coordsBeta_native, coordsC_native, 
             coordsCa_native, coordsN_native, mask, nativemask, esm_embed) in pbar:         
    
        if (step + 1) % 1 == 0:
            start_time = time.time()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
        E_d,E_n = model() 
        # running_loss += (loss.item() * batch_size)
        # dataset_size += batch_size
        
    #     epoch_loss = running_loss / dataset_size
        
    #     mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
    #     current_lr = optimizer.param_groups[0]['lr']
    #     pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
    #                     lr=f'{current_lr:0.5f}',
    #                     gpu_mem=f'{mem:0.2f} GB')
    # torch.cuda.empty_cache()
    # gc.collect()
    
    # return epoch_loss
    return 0

def main():
    train_loader, valid_loader,test_loader = fetch_dataloader(data_dir=CFG.data_path,num_workers =CFG.num_workers,
                                                  batch_size=CFG.batch_size,cuda=CFG.cuda)
    
    # Build the model
    model = ProteinEnergyNet(embedding_size = CFG.embedding_size,filters = CFG.filters, layers = CFG.num_layers,
                             cord_size = CFG.coords_emb,h = CFG.h) 
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    # Run training
    
    return 1
