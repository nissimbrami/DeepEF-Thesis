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
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from tqdm import tqdm
import gc
import time
import sys
import pandas as pd
import wandb

# Set the default data type to float32
torch.set_default_dtype(CFG.torch_default_dtype)
# torch.autograd.set_detect_anomaly(True)
# Set wandb
if not CFG.debug:
    wandb.init(project="Thermodynamic+decoy",name = 'epoch 0 negative loss')
if CFG.debug:
   CFG.model_path = "./res/debug/"
   CFG.results_path = './res/results-debug/'
   print('**** Debug mode ****')



# define validation function
def validation(model, dataloader, device,epoch,N,optimizer,val_type = 'robust'):
    """
    Validation function for the model.
    """
    valid_loss = 0
    valid_lossd = 0
    valid_lossg = 0
    valid_lossc = 0
    Exd_list = []
    Exn_list = []
    seq_len = []
    lossg_list = []
    lossd_list = []
    ids_list = []
    n_skips = 0
    model.eval() # cant use eval because of the loss function calculation
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Validation: Epoch {epoch}")
        for index, data in (enumerate(tepoch)):
            # Clean the GPU cache
            if(device.type == "cuda" or device.type == "mps"):    
                torch.cuda.empty_cache()
            gc.collect()
            # zero the parameter gradients
            optimizer.zero_grad()
            id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy = data
            
            # wild type and mutant type
            Xjf = crd_backbone.to(device) # wilde type structure folded
            Xkf = torch.clone(Xjf).to(device) # mutant structure folded
            Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
            Xku = torch.clone(Xjf).to(device) # mutant structure unfolded
            Xcd = crd_decoy.to(device) # decoy structure
            #Add random step to the native structure
            Xjf_h1,Xjf_h2 = Add_random_step(Xjf, CFG.h)
            
            # change the decoy len to match the native len
            if Xcd.shape[1] > Xjf.shape[1]:
                Xcd = Xcd[:,:Xjf.shape[1],:,:]
                mask_crd_decoy = mask_crd_decoy[:,:Xjf.shape[1]]
            elif Xcd.shape[1] < Xjf.shape[1]:
                # add zeros to the end of the decoy
                Xcd = torch.cat((Xcd, torch.zeros(Xjf.shape[0],Xjf.shape[1] - Xcd.shape[1], *Xcd.shape[2:]).to(device)), dim=1)
                mask_crd_decoy = torch.cat((mask_crd_decoy, torch.zeros(mask_crd_decoy.shape[0],mask.shape[1] - mask_crd_decoy.shape[1])), dim=1)
              
            # native structure and decoy structure
            Xd = torch.clone(Xjf).to(device)
            Xdu = torch.clone(Xjf).to(device)
            seq_one_hot = seq_one_hot.to(device) # [batch_size,20,seq_len]
            
            # create decoy sequence
            seq_decoy,mask_decoy, proT5_emb_decoy = mix_A_acid(seq_one_hot = seq_one_hot, emb=proT5_emb, mask = mask,val_type='train',device=device)
            
            if seq_decoy.shape[1] >CFG.seq_len : # if the sequence is too long, skip it(GPU limitation)
                n_skips += 1
                continue
            #emb = torch.cat((esm_embed,seq),dim=2)
            emb = seq_one_hot.to(device)
            emb_decoy = seq_decoy.to(device)
            emb_mut = get_one_hot(seq_mut[0]).to(device)
            
            # move proT5_emb to device
            proT5_mut, proT5_emb_decoy, proT5_emb = proT5_mut.to(device), proT5_emb_decoy.to(device), proT5_emb.to(device)
           
            # zero the parameter gradients
            optimizer.zero_grad()
            # squeeze the data
            Xd, Xjf, Xkf, Xju, Xku, Xcd, Xdu, Xjf_h1, Xjf_h2 = Xd.squeeze(), Xjf.squeeze(), Xkf.squeeze(), Xju.squeeze(), Xku.squeeze(), Xcd.squeeze(), Xdu.squeeze(), Xjf_h1.squeeze(), Xjf_h2.squeeze()
            emb_decoy, emb, emb_mut = emb_decoy.squeeze(), emb.squeeze(), emb_mut.squeeze()
            mask_decoy, mask, mask_crd_decoy= mask_decoy.squeeze(), mask.squeeze(), mask_crd_decoy.squeeze()
            proT5_emb_decoy, proT5_emb, proT5_mut = proT5_emb_decoy.squeeze(), proT5_emb.squeeze(), proT5_mut.squeeze()
            
            # get folded graph  
            Xjf,Xkf = get_graph(Xjf, emb, proT5_emb, mask), get_graph(Xkf, emb_mut, proT5_mut, mask)
            # get unfolded graph
            Xju,Xku = get_unfolded_graph(Xju, emb, proT5_emb, mask), get_unfolded_graph(Xku, emb_mut, proT5_mut, mask)
            # get decoy graph
            Xd, Xcd, Xdu = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy), get_graph(Xcd, emb, proT5_emb, mask_crd_decoy), get_unfolded_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy)
            # get h1 and h2 graph
            Xjf_h1, Xjf_h2 = get_graph(Xjf_h1, emb, proT5_emb, mask), get_graph(Xjf_h2, emb, proT5_emb, mask)
            # create a batch of Xjf,Xkf,Xju,Xku,x_decoy
            Xjf,Xkf,Xju,Xku,Xd,Xcd,Xdu,Xjf_h1,Xjf_h2= Xjf.unsqueeze(0),Xkf.unsqueeze(0),Xju.unsqueeze(0),Xku.unsqueeze(0),Xd.unsqueeze(0), Xcd.unsqueeze(0), Xdu.unsqueeze(0), Xjf_h1.unsqueeze(0), Xjf_h2.unsqueeze(0)  
            X = torch.cat((Xjf,Xkf,Xju,Xku,Xd,Xcd,Xdu,Xjf_h1,Xjf_h2),dim=0)
            
            with torch.no_grad():
                # half precision validation
                with torch.amp.autocast(device_type="cuda", dtype=CFG.precision):
                    # calculate the energy for the folded unfolded and decoy structure
                    E = model(X)
                    Ejf, Ekf, Eju, Eku, Exd, Ecd, Exdu, Eh1, Eh2 = E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7], E[8]
                    # calculate the loss   
                    loss ,lossd, lossg,lossc = criterion(Ejf, Ekf, Eju, Eku, Exd, Xjf, Ecd, Exdu, Eh1, Eh2, with_grad = True)
                
            # Add gradient penalty
            Ejf_grad = torch.tensor(0.0).to(device)
            
            valid_loss += loss.item() 
            valid_lossd += lossd.item()
            valid_lossg += lossg.item()
            valid_lossc += lossc.item()
            # delete all the variables
            del Xjf,Xkf,Xju,Xku,Xd,emb_decoy, emb, emb_mut, mask_decoy, mask, proT5_emb_decoy, proT5_emb, proT5_mut, X, E, Ejf, Ekf, Eju, Eku, Exd
            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            if index % 1000 == 999:
                print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}, n_skips: {n_skips}")
            
    return valid_loss/len(dataloader),valid_lossd/len(dataloader),valid_lossg/len(dataloader),valid_lossc/len(dataloader)

def train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader,best_val=1000,scheduler=None,scaler=None):
    """
    Training function for the model.
    
    """
    model.train()
    epoch_train_loss = []
    ephoch_val_loss = []
    model.train()
    running_loss = 0.0
    n_skips = 0
    ds_length = len(dataloader)
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Epoch {epoch}")
        for index, data in enumerate(tepoch):
            # Clean the GPU cache
            torch.cuda.empty_cache()
            gc.collect()
             # zero the parameter gradients
            optimizer.zero_grad()
            id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy = data
            
            # wild type and mutant type
            Xjf = crd_backbone.to(device) # wilde type structure folded
            Xkf = torch.clone(Xjf).to(device) # mutant structure folded
            Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
            Xku = torch.clone(Xjf).to(device) # mutant structure unfolded
            Xcd = crd_decoy.to(device) # decoy structure
            
            Xjf_h1,Xjf_h2 = Add_random_step(Xjf, CFG.h)
            
            # change the decoy len to match the native len
            if Xcd.shape[1] > Xjf.shape[1]:
                Xcd = Xcd[:,:Xjf.shape[1],:,:]
                mask_crd_decoy = mask_crd_decoy[:,:Xjf.shape[1]]
            elif Xcd.shape[1] < Xjf.shape[1]:
                # add zeros to the end of the decoy
                Xcd = torch.cat((Xcd, torch.zeros(Xjf.shape[0],Xjf.shape[1] - Xcd.shape[1], *Xcd.shape[2:]).to(device)), dim=1)
                mask_crd_decoy = torch.cat((mask_crd_decoy, torch.zeros(mask.shape[0],mask.shape[1] - mask_crd_decoy.shape[1])), dim=1)
                
            
            # native structure and decoy structure
            Xd = torch.clone(Xjf).to(device)
            Xdu = torch.clone(Xjf).to(device)
            seq_one_hot = seq_one_hot.to(device) # [batch_size,20,seq_len]
            
            # create decoy sequence
            seq_decoy,mask_decoy, proT5_emb_decoy = mix_A_acid(seq_one_hot = seq_one_hot, emb=proT5_emb, mask = mask,val_type='train',device=device)
            
            if seq_decoy.shape[1] >CFG.seq_len : # if the sequence is too long, skip it(GPU limitation)
                n_skips += 1
                continue
            #emb = torch.cat((esm_embed,seq),dim=2)
            emb = seq_one_hot.to(device)
            emb_decoy = seq_decoy.to(device)
            emb_mut = get_one_hot(seq_mut[0]).to(device)
            
            # move proT5_emb to device
            proT5_mut, proT5_emb_decoy, proT5_emb = proT5_mut.to(device), proT5_emb_decoy.to(device), proT5_emb.to(device)
           
            # zero the parameter gradients
            optimizer.zero_grad()
            # squeeze the data
            Xd, Xjf, Xkf, Xju, Xku, Xcd, Xdu, Xjf_h1, Xjf_h2 = Xd.squeeze(), Xjf.squeeze(), Xkf.squeeze(), Xju.squeeze(), Xku.squeeze(), Xcd.squeeze(), Xdu.squeeze(), Xjf_h1.squeeze(), Xjf_h2.squeeze()
            emb_decoy, emb, emb_mut = emb_decoy.squeeze(), emb.squeeze(), emb_mut.squeeze()
            mask_decoy, mask, mask_crd_decoy= mask_decoy.squeeze(), mask.squeeze(), mask_crd_decoy.squeeze()
            proT5_emb_decoy, proT5_emb, proT5_mut = proT5_emb_decoy.squeeze(), proT5_emb.squeeze(), proT5_mut.squeeze()
            
            # get folded graph  
            Xjf,Xkf = get_graph(Xjf, emb, proT5_emb, mask), get_graph(Xkf, emb_mut, proT5_mut, mask)
            # get unfolded graph
            Xju,Xku = get_unfolded_graph(Xju, emb, proT5_emb, mask), get_unfolded_graph(Xku, emb_mut, proT5_mut, mask)
            # get decoy graph
            Xd, Xcd, Xdu = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy), get_graph(Xcd, emb, proT5_emb, mask_crd_decoy), get_unfolded_graph(Xdu, emb_decoy, proT5_emb_decoy, mask_decoy)
            # get h1 and h2 graph
            Xjf_h1, Xjf_h2 = get_graph(Xjf_h1, emb, proT5_emb, mask), get_graph(Xjf_h2, emb, proT5_emb, mask)
            
            # create a batch of Xjf,Xkf,Xju,Xku,x_decoy
            Xjf,Xkf,Xju,Xku,Xd,Xcd,Xdu,Xjf_h1,Xjf_h2 = Xjf.unsqueeze(0),Xkf.unsqueeze(0),Xju.unsqueeze(0),Xku.unsqueeze(0),Xd.unsqueeze(0), Xcd.unsqueeze(0), Xdu.unsqueeze(0), Xjf_h1.unsqueeze(0), Xjf_h2.unsqueeze(0)
            X = torch.cat((Xjf,Xkf,Xju,Xku,Xd,Xcd,Xdu,Xjf_h1,Xjf_h2),dim=0)
            
            # half precision training
            with torch.amp.autocast(device_type="cuda", dtype=CFG.precision):
                # calculate the energy for the folded unfolded and decoy structure
                E = model(X)
                Ejf, Ekf, Eju, Eku, Exd, Ecd, Exdu, Eh1, Eh2 = E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7], E[8]
                # calculate the loss   
                loss ,lossd, lossg,lossc = criterion(Ejf, Ekf, Eju, Eku, Exd, Xjf, Ecd, Exdu,Eh1 ,Eh2, with_grad = True)
            
            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            # Clip gradients to a maximum norm of max_grad_norm to prevent exploding gradients
            if CFG.clip_grad_norm:
                clip_grad_norm(model.parameters(), CFG.max_grad_norm)
            
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            # Add gradient penalty
            Ejf_grad = torch.tensor(0.0).to(device)
                
            # print statistics
            running_loss += loss.item()
            if index % 1000 == 999 :    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 1000:.3f}')
                print(f"skipped {n_skips}")
                epoch_train_loss.append(running_loss/1000)
                if not CFG.debug:
                    wandb.log({"epoch": epoch,"running_loss": running_loss/1000,"running_lossIndex":ds_length*epoch+index})
                running_loss = 0.0

            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(running_loss/(index%1000 + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"lossc":round(lossc.item(),3),"sequence_len": len(seq[0])})
            # Log metrics
            if not CFG.debug:
                wandb.log({"epoch": epoch, "loss": loss.item(),"lossc":lossc.item(),"lossd":lossd.item(),"lossg":lossg.item(), "sequence_len": len(seq[0]),
                           "Exd":Exd.item(),"Eku":Eku.item(),"Ekf":Ekf.item(),"Eju": Eju.item(), "Ejf":Ejf.item(), "Ecd":Ecd.item(),
                           "step": ds_length*epoch+index,"Ejf_grad":Ejf_grad.item(), "Exdu":Exdu.item()})
            
        print(f"skipped {n_skips}")
        save_checkpoint(epoch, model, optimizer, loss,0,CFG.model_path+str(epoch)+"_final_model.pt")
        # evaluate the model
        val_loss, val_lossd,val_lossg,valid_lossc = validation(model, valid_loader,CFG.device,epoch, CFG.N, optimizer , val_type = 'robust')
         # update wandb metrics
        if not CFG.debug:
            wandb.log({"epoch" : epoch ,"validation loss": val_loss, "learning rate": optimizer.param_groups[0]["lr"], "validation lossd": val_lossd, "validation lossg": val_lossg, "validation lossc": valid_lossc})
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
    # setup half precision training
    scaler = torch.cuda.amp.GradScaler()
    for epoch in (range(EPOCH,CFG.num_epochs+EPOCH)):  # loop over the dataset multiple times

        
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        model,epoch_train_loss,valid_loss = train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader,valid_loss, scheduler,scaler)
        
        
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

def gradient_penalty(X_native, E_native):
    """Implementing the lossg equation:
        The gradient of a wild type structure should be close to zero.
        Therefore we will add it to the loss as lossg"""
    partial_dx_native = torch.autograd.grad(outputs=E_native, inputs=X_native,
                                            grad_outputs=torch.ones_like(E_native),
                                            create_graph=True, retain_graph=True)[0]
    # part_dx_native_norm = 0.5*torch.norm(partial_dx_native,p=2)**2
    # lossg = torch.log(part_dx_native_norm+1)
    # Compute the gradient penalty
    gradients = partial_dx_native#.view(partial_dx_native.size(0))
    gradient_penalty = (gradients.norm(2, dim=0) ** 2).mean()
    lossg = torch.log(gradient_penalty+1)
    return lossg

def criterion(Ejf, Ekf, Eju, Eku, Exd, X_native, Ecd, Exdu, Eh1, Eh2, with_grad = True ):
    """
    The loss function for the model corresponds to 3 main losses:
    1. lossg: the partial derivative of the energy with respect to the native structure
    2. lossd: the energy of the native structure divided by the decoy energy
    3. lossc: the energy softplus function for the native and mutant structure(unfolded and folded)
    Args:
        Ejf (tensor): The energy of the folded native structure
        Ekf (tensor): The energy of the folded mutant structure
        Eju (tensor): The energy of the unfolded native structure
        Eku (tensor): The energy of the unfolded mutant structure
        Exd (tensor): The energy of the decoy sequence
        Ecd (tensor): The energy of the decoy structure
        Exdu (tensor): The energy of the decoy structure unfolded
        Eh1 (tensor): The energy of the decoy structure with a random step added
        Eh2 (tensor): The energy of the decoy structure with a random step decreased
    output:
        loss (tensor): The loss of the model
        lossd (tensor): The loss of the model due to the energy of the native structure divided by the decoy energy
        lossg (tensor): The loss of the model due to the partial derivative of the energy with respect to the native structure
        lossc (tensor): The loss of the model due to the energy softplus function for the native and mutant structure(unfolded and folded)
    """
    lossg = numerical_LG(Ejf, Eh1, Eh2) if with_grad else torch.tensor(0.0).to(Ejf.device)
    lossd = lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju)
    lossc = energy_softplus(Ejf, Ekf, Eju, Eku)
    
    return lossd+lossg+lossc , lossd, lossg, lossc
  
def lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju):
    """Decoy loss:
    - the energy of a decoy sequece is greater than the energy of the wild-type structure (Ejf<Exd)
    - the energy of a decoy structure is greater than the energy of the wild-type structure (Ejf<Ecd)
    - the energy of a folded decoy is greater than the energy of an unfolded decoy (Exdu<Exd)
    - the energy of a decoy structure is greater than the energy of the unfolded native structure (Eju<Ecd)
    """
    # loss_decoy = lambda x,y: torch.log((x+1) / (y+1) +1)
    loss_decoy = lambda x,y: x - y
    
    return loss_decoy(Ejf, Exd)+loss_decoy(Ejf, Ecd)+loss_decoy(Exdu, Exd)+loss_decoy(Eju, Ecd)

def numerical_LG(Ejf, Eh1, Eh2, h = CFG.h):
    """Numerical LG:
    Calculate the lossg in respect to a numerical gradient
    """
    # first order numerical gradient
    g1 = ((Eh1-Eh2)/(2*h))**2 # the first order numerical gradient should be close to zero
    # second order numerical gradient
    # g2 = torch.sigmoid(-1 * (Eh1- 2*Ejf + Eh2)/(h**2)) # the second order numerical gradient should be positive
    # g2 = (Eh1- 2*Ejf + Eh2)/(h**2)
    g2 = torch.tensor(0.0).to(Eh1.device)
    # add sigmoid to the gradient
    lossg = g1-g2 
    return lossg

def energy_softplus(Ejf, Ekf, Eju, Eku, beta = 1):
    """Energy softplus,
    As we know the energy diffrence between an unfolded protein and folded protein is positive.
    Therefore we will add it to the loss as lossc"""
    
    folded_threshold = 2 # the ratio between the energy of the folded and unfolded protein should be greater than 2
    softplus = lambda x: torch.log(torch.exp(beta*x)+1)/beta
    if (Ekf * folded_threshold < Eku and Eku > Ekf):
        lossc1 = softplus(-Ekf) 
    else:
        lossc1 = softplus(Ekf-Eku)
        
    if (Ejf * folded_threshold < Eju and Eju > Ejf):
        lossc2 = softplus(-Ejf) 
    else:
        lossc2 = softplus(Ejf-Eju)
    # lossd2 = torch.where(lossd2 < 0.05, torch.tensor(0.0).to(lossd2.device), torch.where(lossd2 > 10.0, lossd2/2.0, lossd2))
    return lossc1+lossc2


def relu_energy(Ejf, Ekf, Eju, Eku, offset = 5):
    """calculate the energy diffrence between an unfolded protein and folded protein is positive.
    log(min(max(diff+offset,0),6)+1)
    """
    htanh = torch.nn.Hardtanh(min_val=0, max_val=10)
    lossd = lambda x: torch.log(htanh(x+offset)+1)
    return lossd(Ekf-Eku)+lossd(Ejf-Eju)

    
def trainAndTest(model,train_loader,valid_loader,test_loader,optimizer,device,N,epoch,scheduler):
    "train and test the model"
    valid_loss = 100
    if epoch > 0:
        model,optimizer,epoch,loss,valid_loss = load_checkpoint(CFG.model_path+f"{epoch-1}_final_model.pt", model, optimizer)
        epoch += 1
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
    model = PEM(layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef).to(CFG.device)
    model.name = "PEM-With LLM embedding"
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
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
    if not CFG.debug:    
        CFG.model_path = "./res/trianed_models-numerical_LG/" # nomerical lossg 
        CFG.results_path = './res/results-numerical_LG/'
    main()
