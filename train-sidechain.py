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
    wandb.init(project="deepmeshi")


# define amino acid inference
def A_inference(model, dataloader, device,N,optimizer,val_type = 'robust'):
    """
    Validation function for the model.
    """
    valid_loss = 0
    Exd_A_list = []
    Exn_A_list = []
    ids_list = []
    model.eval() # cant use eval because of the loss function calculation
    with tqdm(dataloader, unit="batch") as tepoch:
        for index, data in enumerate(dataloader):
            # Clean the GPU cache
            if(device.type == "cuda" or device.type == "mps"):    
                torch.cuda.empty_cache()
            gc.collect()
            # zero the parameter gradients
            optimizer.zero_grad()
            # get the inputs; data is a list of [inputs, labels]   
            seq_one_hot,seq_decoy ,id, Xd,Xn, mask, nativemask, esm_embed = data
            # Xd = Xd.to(device)
            # Take native structure
            Xd = torch.clone(Xn).to(device)
            Xn = Xn.to(device)
            esm_embed = esm_embed.to(device)
            seq_one_hot = seq_one_hot.to(device) # [batch_size,20,seq_len]
            seq_one_hot = torch.swapaxes(seq_one_hot,1,2) # swap the axes to [batch_size,seq_len,20]
            if(seq_one_hot.shape[1]>1000): # skip long sequences due to GPU memory
                continue
            if val_type == 'robust' or val_type == 'train':
                seq_decoy = torch.swapaxes(seq_decoy,1,2)
            else: 
                seq_decoy = torch.clone(seq_one_hot).to(device)
                seq_decoy[:,torch.randperm(seq_decoy.shape[1])[:1],:] = seq_decoy[:,torch.randperm(seq_decoy.shape[1])[:1],:]
            #emb = torch.cat((esm_embed,seq),dim=2)
            emb = seq_one_hot
            emb_decoy = seq_decoy.to(device)
            
            Xd = Xd.squeeze()
            Xn = Xn.squeeze()
            
            emb_decoy = emb_decoy.squeeze()
            emb = emb.squeeze()
            
            
            E_amino = model(Xd,emb_decoy,Xn,emb,f_type = 'A_inference')
            x_decoy,x_native = E_amino[0],E_amino[1]
            Exd_A_list.append(x_decoy.detach().numpy().squeeze())
            Exn_A_list.append(x_native.detach().numpy().squeeze())
            ids_list.append(id)
            ids_list.append(id)

            # valid_loss += loss.item() 
            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            if index % 1000 == 99:
                print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}")
            
    df= pd.DataFrame(Exd_A_list)
    df = pd.concat([df,pd.DataFrame(Exn_A_list)])  
    df['id'] = ids_list
    df.to_csv(f'./res/results/A_inference_{val_type}.csv')
    print(f"Finished amino acid inference {val_type}")
            

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
    # model.eval() # cant use eval because of the loss function calculation
    with tqdm(dataloader, unit="batch") as tepoch:
        for index, data in (enumerate(tepoch)):
            # set progress bar description
            tepoch.set_description(f"Validation: Epoch {epoch}")
            # Clean the GPU cache
            if(device.type == "cuda" or device.type == "mps"):    
                torch.cuda.empty_cache()
            gc.collect()
            # zero the parameter gradients
            optimizer.zero_grad()
            id, Xn, mask, seq_one_hot, seq,ang_backbone, ang,proT5_emb, dist_matrix = data
            
            # Take native structure.
            Xd = torch.clone(Xn).to(device)
            Xn = Xn.to(device)
            seq_one_hot = seq_one_hot.to(device) # [batch_size,20,seq_len]
            
            # create decoy sequence
            seq_decoy,mask_decoy, proT5_emb_decoy = mix_A_acid(seq_one_hot = seq_one_hot, emb=proT5_emb, mask = mask,val_type='train',device=device)
            
            if seq_decoy.shape[1] >CFG.seq_len : # if the sequence is too long, skip it(GPU limitation)
                n_skips += 1
                continue
            #emb = torch.cat((esm_embed,seq),dim=2)
            emb = seq_one_hot.to(device)
            emb_decoy = seq_decoy.to(device)
            # move proT5_emb to device
            proT5_emb_decoy, proT5_emb = proT5_emb_decoy.to(device), proT5_emb.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # squeeze the data
            Xd, Xn= Xd.squeeze(), Xn.squeeze()
            emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
            mask_decoy, mask= mask_decoy.squeeze(), mask.squeeze()
            proT5_emb_decoy, proT5_emb = proT5_emb_decoy.squeeze(), proT5_emb.squeeze()
              
            X_native = get_graph(Xn, emb, proT5_emb, mask)
            X_decoy = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy)
            X_native.requires_grad = True
            
            Exn = model(X_native)
            Exd = model(X_decoy)
            outputs = torch.cat((Exd.unsqueeze(0),Exn.unsqueeze(0)),dim=0)
            
            loss ,lossd, lossg,Exn,Exd = criterion(outputs,X_decoy,X_native,model,N,CFG.h)
            valid_loss += loss.item() 
            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            if index % 1000 == 999:
                print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}, n_skips: {n_skips}")
                validation_plots(Exd_list,Exn_list,seq_len,val_type,epoch)
            #tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(valid_loss/(index + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"Exn":round(Exn.item(),3),"Exd":round(Exd.item(),3)})
            
            Exd_list.append(Exd.item())
            Exn_list.append(Exn.item())
            seq_len.append(Xd.shape[0])
            lossg_list.append(lossg.item())
            lossd_list.append(lossd.item())
            ids_list.append(id)
    
    validation_plots(Exd_list,Exn_list,seq_len,val_type,epoch)
    df = pd.DataFrame({'id':ids_list,'Exd':Exd_list,'Exn':Exn_list,'seq_len':seq_len,'lossg':lossg_list,'lossd':lossd_list})
    df.to_csv(f'./res/results/epoch_{epoch}-validation_{val_type}.csv')
    print(f"Finished Validation {val_type} epoch {epoch}")
            
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
            id, Xn, mask, seq_one_hot, seq,ang_backbone, ang, proT5_emb, dist_matrix = data
            # Take native structure
            Xd = torch.clone(Xn).to(device)
            Xn = Xn.to(device)
            
            seq_one_hot = seq_one_hot.to(device) # [batch_size,seq_len,20]
            # create decoy sequence
            seq_decoy,mask_decoy, proT5_emb_decoy = mix_A_acid(seq_one_hot = seq_one_hot, emb=proT5_emb, mask = mask,val_type='train',device=device)
            
            if seq_decoy.shape[1] >CFG.seq_len : # if the sequence is too long, skip it(GPU limitation)
                n_skips += 1
                continue
            #emb = torch.cat((esm_embed,seq),dim=2)
            emb = seq_one_hot.to(device)
            emb_decoy = seq_decoy.to(device)
            # move proT5_emb to device
            proT5_emb_decoy, proT5_emb = proT5_emb_decoy.to(device), proT5_emb.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # squeeze the data
            Xd, Xn= Xd.squeeze(), Xn.squeeze()
            emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
            mask_decoy, mask= mask_decoy.squeeze(), mask.squeeze()
            proT5_emb_decoy, proT5_emb = proT5_emb_decoy.squeeze(), proT5_emb.squeeze()
              
            X_native = get_graph(Xn, emb, proT5_emb, mask)
            X_decoy = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy)
            X_native.requires_grad = True
            
            Exn = model(X_native)
            Exd = model(X_decoy)
            outputs = torch.cat((Exd.unsqueeze(0),Exn.unsqueeze(0)),dim=0)
            
            loss ,lossd, lossg,Exn,Exd = criterion(outputs,X_decoy,X_native,model,N,CFG.h)
            
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
            tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(running_loss/(index%1000 + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"Exn":round(Exn.item(),3),"Exd":round(Exd.item(),3)})
            # Log metrics
            if not CFG.debug:
                wandb.log({"epoch": epoch, "loss": loss.item(),"lossd":lossd.item(),"lossg":lossg.item(),"Exn":Exn.item(),"Exd":Exd.item(),"Edelta": (Exd-Exn).item(),"sequence_len": X_decoy.shape[0]})
            
        print(f"skipped {n_skips}")
        save_checkpoint(epoch, model, optimizer, loss,0,CFG.model_path+str(epoch)+"_final_model.pt")
        # evaluate the model
        r_val = validation(model, valid_loader,CFG.device,epoch, CFG.N, optimizer , val_type = 'robust')
        s_val = validation(model, valid_loader,CFG.device,epoch , CFG.N, optimizer, val_type = 'soft')
         # update wandb metrics
        if not CFG.debug:
            wandb.log({"epoch" : epoch ,"robust validation loss": r_val,"soft validation loss": s_val, "learning rate": optimizer.param_groups[0]["lr"]})
         # Update the learning rate based on the validation loss
        scheduler.step(r_val)
        print (f"robust validation loss: {r_val}")
        print (f"soft validation loss: {s_val}")
        if r_val<best_val:
            print('saving model with valid loss: ',r_val)
            save_checkpoint(epoch, model, optimizer, loss,r_val,CFG.model_path+"best_model.pt")
            best_val = r_val
       
        
                
    return model, epoch_train_loss,r_val

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

def criterion(E,X_decoy,X_native,model,N,h):
    """
    The loss function for the model coressponds to 3 main losses:
    1. lossg: the partial derivateve of the energy with respect to the native structure
    2. lossd: the energy of the native structure divided by the decoy energy
    3. lossc: After preforming an iterative optimization on the decoy structure, by using the energy partial derivative on the decoy structure, 
              we calculate the dRMSD of the end and the start of the optimization.
    Args:
        E (tensor): A tensor containing the energy of the native and the decoy structure Exd,Exn [2]
        X_native (tensor): A tensor containing the native structure [batch_size,seq_len,4,3]
        X_decoy (tensor): A tensor containing the decoy structure [batch_size,seq_len,4,3]
        model (torch.model): model that was trained
        N (int): The number of iterations for the iterative optimization
        h (float): The step size for the numerical derivative
    output:
        loss (tensor): The loss of the model
    """
    # print('***Start criterion function***')
    partial_dx_native = torch.autograd.grad(outputs=E[1], inputs=X_native, grad_outputs=torch.ones_like(E[1]), create_graph=True)[0]
    part_dx_native_norm = 0.5*torch.norm(partial_dx_native,p=2)**2
   
    lossg = 2/(1+torch.exp(-part_dx_native_norm)) -1
    lossd = (torch.log((E[1]+1) / (E[0]+1) +1)).mean()
    
    # lossc = preform_energy_optimization(X_decoy,partial_dx_decoy)
    # print(f"loss g: {round(lossg.item(),4)} loss d: {round(lossd.item(),4)}")
    return lossd+lossg , lossd, lossg,E[1],E[0]   

def trainAndTest(model,train_loader,valid_loader,test_loader,optimizer,device,N,epoch,scheduler):
    "train and test the model"
    valid_loss = 100
    if epoch > 0:
        model,optimizer,epoch,loss,valid_loss = load_checkpoint(CFG.model_path+f"best_model.pt", model, optimizer,CFG.device)
    training(model, optimizer, train_loader,valid_loader, CFG.device,CFG.N,epoch,valid_loss,scheduler)
    #load the best model and check the validation
    load_checkpoint(CFG.model_path+f"best_model.pt", model, optimizer,CFG.device)
    validation(model, valid_loader,CFG.device,-1, CFG.N, optimizer , val_type = 'robust')
    validation(model, valid_loader,CFG.device,-1, CFG.N, optimizer, val_type = 'soft')
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
    model.name = "PEM-With LLM embedding"
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    # Define the learning rate scheduler based on loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
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
        CFG.model_path = sys.argv[1]
        CFG.SM = False#int(sys.argv[2])
    main()
