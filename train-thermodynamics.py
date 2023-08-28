from model.data_loader import fetch_dataloader,fetch_inference_loader
from model.data_loader import params as data_params
from model.model_cfg import CFG
# from model.net import ProteinEnergyNet
from model.hydro_net import PEM
from model.net import params as model_params
from train_utils import *
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc
import time
import sys
import pandas as pd
import wandb

# Set the default data type to float32
torch.set_default_dtype(CFG.torch_default_dtype)
# Set wandb
if not CFG.debug:
    wandb.init(project="Thermodynamic cycle")

            

# define validation function
def validation(model, dataloader, device,epoch,N,optimizer,val_type = 'robust'):
    """
    Validation function for the model.
    """
    valid_loss = 0
    Exd_list = []
    Exn_list = []
    seq_len = []
    lossg_list = []
    lossd_list = []
    ids_list = []
    n_skips = 0
    model.eval() 
    with tqdm(dataloader, unit="batch") as tepoch:
        for index, data in (enumerate(tepoch)):
            # set progress bar description
            tepoch.set_description(f"Validation: Epoch {epoch}")
            # Clean the GPU cache
            if(device.type == "cuda" or device.type == "mps"):    
                torch.cuda.empty_cache()
            gc.collect()
            # get the inputs; data is a list of [inputs, labels]   
            id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang, proT5_emb, proT5_mut,seq_mut = data
            
            Xjf = crd_backbone.to(device) # wilde type structure folded
            Xkf = torch.clone(Xjf).to(device) # mutant structure folded
            Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
            Xku = torch.clone(Xjf).to(device) # mutant structure unfolded

            mask = mask.to(device)
            mask_decoy = torch.clone(mask).to(device)
            
            seq_one_hot = seq_one_hot.to(device) # [batch_size,seq_len,20]
            seq_one_hot_mut = get_one_hot(seq_mut[0]).to(device) # [batch_size,seq_len,20]
            
            if seq_one_hot.shape[1] >CFG.seq_len : # if the sequence is too long, skip it(GPU limitation)
                n_skips += 1
                continue
            
            emb = seq_one_hot.to(device)
            emb_decoy = seq_one_hot_mut.to(device)
            # move proT5_emb to device
            proT5_mut, proT5_emb = proT5_mut.to(device), proT5_emb.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # squeeze the data
            Xjf, Xkf, Xju, Xku = Xjf.squeeze(), Xkf.squeeze(), Xju.squeeze(), Xku.squeeze()
            emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
            mask_decoy, mask= mask_decoy.squeeze(), mask.squeeze()
            proT5_mut, proT5_emb = proT5_mut.squeeze(), proT5_emb.squeeze()
            # get folded graph  
            Xjf,Xkf = get_graph(Xjf, emb, proT5_emb, mask), get_graph(Xkf, emb, proT5_mut, mask)
            # get unfolded graph
            Xju,Xku = get_unfolded_graph(Xju, emb, proT5_emb, mask), get_unfolded_graph(Xku, emb, proT5_mut, mask)
            # calculate the energy for the folded unfolded structures
            Ejf, Ekf, Eju, Eku = model(Xjf), model(Xkf), model(Xju), model(Xku)
            
            loss ,lossd, lossg = criterion(Ejf, Ekf, Eju, Eku)
            valid_loss += loss.item() 
            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            if index % 1000 == 999:
                print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}, n_skips: {n_skips}")
                # validation_plots(Exd_list,Exn_list,seq_len,val_type,epoch)
            #tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(valid_loss/(index + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"Exn":round(Exn.item(),3),"Exd":round(Exd.item(),3)})
            
            # Exd_list.append(Exd.item())
            # Exn_list.append(Exn.item())
            # seq_len.append(Xd.shape[0])
            lossg_list.append(lossg.item())
            lossd_list.append(lossd.item())
            ids_list.append(id)
    
    # validation_plots(Exd_list,Exn_list,seq_len,val_type,epoch)
    # df = pd.DataFrame({'id':ids_list,'Exd':Exd_list,'Exn':Exn_list,'seq_len':seq_len,'lossg':lossg_list,'lossd':lossd_list})
    # df.to_csv(f'./res/results/epoch_{epoch}-validation_{val_type}.csv')
    # print(f"Finished Validation {val_type} epoch {epoch}")
            
    return valid_loss/len(dataloader)

def train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader,best_val=1000,scheduler=None):
    """
    Training function for the model.
    
    """
    epoch_train_loss = []
    ephoch_val_loss = []
    model.train()
    running_loss = 0.0
    n_skips = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for index, data in enumerate(tepoch):
            # set progress bar description
            tepoch.set_description(f"Epoch {epoch}")
            # Clean the GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            # get the inputs; data is a list of [inputs, labels]   
            id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang, proT5_emb, proT5_mut,seq_mut = data
            
            Xjf = crd_backbone.to(device) # wilde type structure folded
            Xkf = torch.clone(Xjf).to(device) # mutant structure folded
            Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
            Xku = torch.clone(Xjf).to(device) # mutant structure unfolded

            mask = mask.to(device)
            mask_decoy = torch.clone(mask).to(device)
            
            seq_one_hot = seq_one_hot.to(device) # [batch_size,seq_len,20]
            seq_one_hot_mut = get_one_hot(seq_mut[0]).to(device) # [batch_size,seq_len,20]
            
            if seq_one_hot.shape[1] >CFG.seq_len : # if the sequence is too long, skip it(GPU limitation)
                n_skips += 1
                continue
            
            emb = seq_one_hot.to(device)
            emb_decoy = seq_one_hot_mut.to(device)
            # move proT5_emb to device
            proT5_mut, proT5_emb = proT5_mut.to(device), proT5_emb.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # squeeze the data
            Xjf, Xkf, Xju, Xku = Xjf.squeeze(), Xkf.squeeze(), Xju.squeeze(), Xku.squeeze()
            emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
            mask_decoy, mask= mask_decoy.squeeze(), mask.squeeze()
            proT5_mut, proT5_emb = proT5_mut.squeeze(), proT5_emb.squeeze()
            # get folded graph  
            Xjf,Xkf = get_graph(Xjf, emb, proT5_emb, mask), get_graph(Xkf, emb, proT5_mut, mask)
            # get unfolded graph
            Xju,Xku = get_unfolded_graph(Xju, emb, proT5_emb, mask), get_unfolded_graph(Xku, emb, proT5_mut, mask)
            # calculate the energy for the folded unfolded structures
            Ejf, Ekf, Eju, Eku = model(Xjf), model(Xkf), model(Xju), model(Xku)
            
            loss ,lossd, lossg = criterion(Ejf, Ekf, Eju, Eku)
            
            loss.backward()
            # print_par(model) # print the parameters of the model
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if index % 1000 == 999 :    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 1000:.3f}')
                print(f"skipped {n_skips}")
                epoch_train_loss.append(running_loss/1000)
                if not CFG.debug:
                    wandb.log({"epoch": epoch,"running_loss": running_loss/1000})
                running_loss = 0.0

            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(running_loss/(index%1000 + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3)})
            # Log metrics
            if not CFG.debug:
                wandb.log({"epoch": epoch, "loss": loss.item(),"lossd":lossd.item(),"lossg":lossg.item(),"Eku":Eku.item(),"Ekf":Ekf.item(),"Eju": Eju.item(), "Ejf":Ejf.item(), "sequence_len": Xjf.shape[0]})
            
        print(f"skipped {n_skips}")
        save_checkpoint(epoch, model, optimizer, loss,0,CFG.model_path+str(epoch)+"_final_model.pt")
        # evaluate the model
        val_loss = validation(model, valid_loader,CFG.device,epoch, CFG.N, optimizer , val_type = 'robust')
         # update wandb metrics
        if not CFG.debug:
            wandb.log({"epoch" : epoch ,"validation loss": val_loss, "learning rate": optimizer.param_groups[0]["lr"]})
         # Update the learning rate based on the validation loss
        scheduler.step(val_loss)
        print (f"validation loss: {val_loss}")
        if val_loss<best_val:
            print('saving model with valid loss: ',val_loss)
            save_checkpoint(epoch, model, optimizer, loss,val_loss,CFG.model_path+"best_model.pt")
            best_val = val_loss
       
        
                
    return model, epoch_train_loss,val_loss

# define one epoch train
def training (model, optimizer, dataloader,valid_loader, device,N,EPOCH,valid_loss,scheduler):
    """
    Training function for the model.
    Args:
        model (torch.model): model to train
        optimizer (torch.optim): optimizer to use
        dataloader (torch.utils.data.DataLoader): dataloader for the training set
        valid_loader (torch.utils.data.DataLoader): dataloader for the validation set
        device (torch.device): device to use ('cpu' or 'cuda' or 'mps')
        N (int): The number of iterations for the iterative optimization
        epoch (int): The current epoch
    """
    model.train()
    
    for epoch in (range(EPOCH,CFG.num_epochs+EPOCH)):  # loop over the dataset multiple times

        
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        model,epoch_train_loss,valid_loss = train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader,valid_loss, scheduler)
        
        
    print('Finished Training')

def preform_energy_optimization(X_decoy,partial_dx_decoy):
    """
    Preform an iterative optimization on the decoy structure, by using the energy partial derivative on the decoy structure,
    Args:
        X_decoy (tensor): A tensor containing the decoy structure [batch_size,seq_len,4,3]
        partial_dx_decoy (tensor): A tensor containing the partial derivative of the energy with respect to the decoy structure [batch_size,seq_len,4,3]
    output:
        lossc (tensor): The dRMSD of the end and the start of the optimization.
    """
    return 0

def criterion(Ejf, Ekf, Eju, Eku):
    """
    The loss function for the model coressponds to 2 main losses:
    1. lossg: delta energy betweeen the folded and unfolded structures
    2. lossd: the thermodynamic cycle loss
    Args:
        Ejf (tensor): The energy of the folded structure
        Ekf (tensor): The energy of the folded structure with mutation
        Eju (tensor): The energy of the unfolded structure
        Eku (tensor): The energy of the unfolded structure with mutation
    output:
        loss (tensor): The loss of the model
    """
    
    # Ejf, Ekf, Eju, Eku = torch.tanh(Ejf), torch.tanh(Ekf), torch.tanh(Eju), torch.tanh(Eku) # clip the energy to be between -1 and 1
    delta_g1, delta_g2, delta_g3, delta_g4 = Ejf-Eju, Ekf-Ejf, Eku-Eju, Ekf-Eku # themodynamic cycle, from the paper
    lossg = ((delta_g1+delta_g2)-(delta_g3+delta_g4))**2
    # lossd = torch.log(torch.exp(Ekf-Eku) +1)+ torch.log(torch.exp(Ejf-Eju) +1)
    lossd = energy_softplus(Ejf, Ekf, Eju, Eku)
    # lossd = torch.sigmoid(Ekf-Eku) + torch.sigmoid(Ejf-Eju)
    
    return lossd+lossg , lossd, lossg  

def energy_softplus(Ejf, Ekf, Eju, Eku, beta = 1):
    """Energy softplus,
    As we know the energy diffrence between an unfolded protein and folded protein is positive.
    Therefore we will add it to the loss as lossd"""
    softplus = torch.nn.Softplus(beta=beta)
    
    lossd1 = softplus(Ekf-Eku)
    lossd1 =  torch.where(lossd1 < 0.05, torch.tensor(0.0).to(lossd1.device), torch.min(torch.tensor(10.0).to(lossd1.device), lossd1))
    
    lossd2 = softplus(Ejf-Eju)
    lossd2 = torch.where(lossd2 < 0.05, torch.tensor(0.0).to(lossd2.device), torch.min(torch.tensor(10.0).to(lossd2.device), lossd2))
    
    return lossd1+lossd2

def trainAndTest(model,train_loader,valid_loader,test_loader,optimizer,device,N,epoch,scheduler):
    "train and test the model"
    valid_loss = 100
    if epoch > 0:
        model,optimizer,epoch,loss,valid_loss = load_checkpoint(CFG.model_path+f"best_model.pt", model, optimizer,CFG.device)
    training(model, optimizer, train_loader,valid_loader, CFG.device,CFG.N,epoch,valid_loss,scheduler)
    # load the best model and check the validation
    load_checkpoint(CFG.model_path+f"best_model.pt", model, optimizer,CFG.device)
    validation(model, valid_loader,CFG.device,-1, CFG.N, optimizer , val_type = 'robust')
    validation(model, train_loader,CFG.device,-1, CFG.N, optimizer, val_type = 'train')  
    # amino acid inference
    # A_inference(model, amino_inference_loader, CFG.device, CFG.N,optimizer,val_type = 'robust') 
    # create diffucion data
    # diff_data(model, optimizer, train_loader,valid_loader, CFG.device,CFG.N,epoch)
    
def main():
    print('***Start main function***')
    print('***load the data with dataloader***')
    d_params = data_params(num_workers =CFG.num_workers, batch_size=CFG.batch_size,cuda=CFG.cuda,constraint=CFG.constraint, debug=CFG.debug,dataset='scn')
    train_loader, valid_loader,test_loader = fetch_dataloader(data_dir=CFG.data_path, params=d_params)
    # amino_inference_loader = fetch_inference_loader(data_dir=CFG.inference_path, params=d_params)
    # Build the model
    print('***Build the model***')
    # m_params = model_params(embedding_size = CFG.embedding_size,filters = CFG.filters, layers = CFG.num_layers,
    #                          h = CFG.h,device=CFG.device)
    model = PEM(dim_in=36,dim_h=64,dim_out=36,layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef).to(CFG.device)
    model.name = "PEM-thermodynamic cycle"
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    # Define the learning rate scheduler based on loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    # configurate wandb
    wandb_config(wandb, model, optimizer, scheduler, train_loader)
    # Run training
    print('***Start training***')
    epoch = 0
    trainAndTest(model,train_loader,valid_loader,test_loader,optimizer,CFG.device,CFG.N,epoch, scheduler)
    return 1

    
def print_par(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
   
if __name__ == '__main__':
    if len(sys.argv)>1:
        # CFG.model_path = sys.argv[1]
        CFG.SM = False#int(sys.argv[2])
    main()
