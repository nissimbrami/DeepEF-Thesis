from model.data_loader import PEFDataset,fetch_dataloader
from model.data_loader import params as data_params
from model.model_cfg import CFG
from model.net import ProteinEnergyNet
from model.net import params as model_params
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
    print('***Start main function***')
    print('***load the data with dataloader***')
    d_params = data_params(num_workers =CFG.num_workers, batch_size=CFG.batch_size,cuda=CFG.cuda,debug=CFG.debug)
    train_loader, valid_loader,test_loader = fetch_dataloader(data_dir=CFG.data_path, params=d_params)
    
    # Build the model
    print('***Build the model***')
    m_params = model_params(embedding_size = CFG.embedding_size,filters = CFG.filters, layers = CFG.num_layers,
                             cord_size = CFG.coords_emb,h = CFG.h,device=CFG.device)
    model = ProteinEnergyNet(m_params).to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    # Run training
    print('***Start training***')
    x_test = torch.randn(2,10,4,3).to(CFG.device)
    x_test_native = torch.randn(2,10,4,3).to(CFG.device)
    x_test_embed = torch.randn(2,10,480).to(CFG.device)
    y_pred = model(x_test,x_test_native,x_test_embed)
    print(y_pred.shape)
    return 1

if __name__ == '__main__':
    main()