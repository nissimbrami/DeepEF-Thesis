from model.data_loader import PEFDataset,fetch_dataloader
from model.data_loader import params as data_params
from model.model_cfg import CFG
# from model.net import ProteinEnergyNet
from model.hydro_net import PEM
from model.net import params as model_params
from train_utils import save_checkpoint,load_checkpoint,validation_plots
import torch
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc
import time
import sys

# define validation function
def validation(model, dataloader, device,epoch,N,optimizer,type = 'robust'):
    """
    Validation function for the model.
    """
    valid_loss = 0
    Exd_list = []
    Exn_list = []
    seq_len = []
    # model.eval() # cant use eval because of the loss function calculation
    #with tqdm(dataloader, unit="batch") as tepoch:
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
        if type == 'robust':
            seq_decoy = torch.swapaxes(seq_decoy,1,2)
        else: 
            seq_decoy = torch.clone(seq_one_hot).to(device)
            seq_decoy[:,torch.randperm(seq_decoy.shape[1])[:1],:] = seq_decoy[:,torch.randperm(seq_decoy.shape[1])[:1],:]
        #emb = torch.cat((esm_embed,seq),dim=2)
        emb = seq_one_hot
        emb_decoy = seq_decoy.to(device)
        
        Xd = Xd.squeeze()
        Xn = Xn.squeeze()
        # Xd = Xd.reshape(Xd.shape[0],-1)
        emb_decoy = emb_decoy.squeeze()
        emb = emb.squeeze()
        
        # Xd_features = torch.cat((Xd,emb_decoy),dim=1)
            # create edge_index
        edge_index = torch.tensor([],dtype=torch.long)
        # forward + backward + optimize
        for i in range(Xd.shape[0]):
            for j in range(i,Xd.shape[0]):
                if i == j:
                    continue
                else:
                    edge_index = torch.cat((edge_index,torch.tensor([[i,j]],dtype=torch.long)),dim=0) 
        edge_index = edge_index.to(device)
        outputs = model(Xd,emb_decoy,Xn,emb,edge_index.t().contiguous())
        
        loss ,lossd, lossg,Exn,Exd = criterion(outputs,Xd,Xn,model,N,CFG.h)

        valid_loss += loss.item() 
        torch.cuda.empty_cache()
        gc.collect()
        # update the progress bar
        if index % 100 == 99:
            print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}")
            validation_plots(Exd_list,Exn_list,seq_len,type)
        #tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(valid_loss/(index + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"Exn":round(Exn.item(),3),"Exd":round(Exd.item(),3)})
        
        Exd_list.append(Exd.item())
        Exn_list.append(Exn.item())
        seq_len.append(Xd.shape[0])
    
    validation_plots(Exd_list,Exn_list,seq_len,type)
    
    print(f"Finished Validation {type}")
            
    return valid_loss/len(dataloader)

def train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader):
    """
    Training function for the model.
    
    """
    epoch_train_loss = []
    ephoch_val_loss = []
    model.train()
    running_loss = 0.0
    with tqdm(dataloader, unit="batch") as tepoch:
        for index, data in enumerate(tepoch):
            # set progress bar description
            tepoch.set_description(f"Epoch {epoch}")
            # Clean the GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            # get the inputs; data is a list of [inputs, labels]   
            seq_one_hot,seq_decoy ,id, Xd,Xn, mask, nativemask, esm_embed = data
            Xd = Xd.to(device)
            Xn = Xn.to(device)
            esm_embed = esm_embed.to(device)
            seq_one_hot = seq_one_hot.to(device) # [batch_size,20,seq_len]
            seq_one_hot = torch.swapaxes(seq_one_hot,1,2) # swap the axes to [batch_size,seq_len,20]
            
            seq_decoy = torch.swapaxes(seq_decoy,1,2)
            
            #emb = torch.cat((esm_embed,seq),dim=2)
            emb = seq_one_hot
            emb_decoy = seq_decoy.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            Xd = Xd.squeeze()
            Xn = Xn.squeeze()
            # Xd = Xd.reshape(Xd.shape[0],-1)
            emb_decoy = emb_decoy.squeeze()
            emb = emb.squeeze()
            
            # Xd_features = torch.cat((Xd,emb_decoy),dim=1)
                # create edge_index
            edge_index = torch.tensor([],dtype=torch.long)
            # forward + backward + optimize
            for i in range(Xd.shape[0]):
                for j in range(i,Xd.shape[0]):
                    if i == j:
                        continue
                    else:
                        edge_index = torch.cat((edge_index,torch.tensor([[i,j]],dtype=torch.long)),dim=0) 
            edge_index = edge_index.to(device)
            
            outputs = model(Xd,emb_decoy,Xn,emb,edge_index.t().contiguous())
            
            loss ,lossd, lossg,Exn,Exd = criterion(outputs,Xd,Xn,model,N,CFG.h)
            
            loss.backward()
            # print_par(model) # print the parameters of the model
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if index % 1000 == 999 or CFG.debug:    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 1000:.3f}')
                save_checkpoint(epoch, model, optimizer, running_loss/1000,0,CFG.model_path+str(epoch)+str(index+1)+"train_model.pt")
               
                epoch_train_loss.append(running_loss/1000)
                running_loss = 0.0

            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(running_loss/(index%1000 + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"Exn":round(Exn.item(),3),"Exd":round(Exd.item(),3)})
            
        save_checkpoint(epoch, model, optimizer, loss,0,CFG.model_path+str(epoch)+"_final_model.pt")
        # # evaluate the model
        # with torch.no_grad():
        #     current_valid_loss = validation(model, valid_loader, device,epoch,N)
        #     epoch_val_loss.append(current_valid_loss)
        #     print(f"loss: {round(loss.item(),3)} current_valid_loss:{round(current_valid_loss,3)}")
        #     valid_loss = current_valid_loss
        #     print('saving model with valid loss: ',valid_loss)
        #     save_checkpoint(epoch, model, optimizer, loss,valid_loss,CFG.model_path+str(epoch)+"_model.pt")
        
        
                
    return model, epoch_train_loss,ephoch_val_loss

# define one epoch train
def training (model, optimizer, dataloader,valid_loader, device,N):
    """
    Training function for the model.
    Args:
        model (torch.model): model to train
        optimizer (torch.optim): optimizer to use
        dataloader (torch.utils.data.DataLoader): dataloader for the training set
        valid_loader (torch.utils.data.DataLoader): dataloader for the validation set
        device (torch.device): device to use ('cpu' or 'cuda' or 'mps')
        N (int): The number of iterations for the iterative optimization
    """
    model.train()
    
    for epoch in range(CFG.num_epochs):  # loop over the dataset multiple times

        
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        model,epoch_train_loss,ephoch_val_loss = train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader)
       
        
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

def criterion(E,X_native,X_decoy,model,N,h):
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
    # partial_dx_decoy = torch.autograd.grad(E[:,0].sum(),model.parameters(),create_graph=True,allow_unused=True)
    partial_dx_native = torch.autograd.grad(E[1].sum(),model.parameters(),create_graph=True,allow_unused=True)
    # print('***End derivative calc function***')
    
    part_dx_native = [1 if part_dx is None else torch.norm(part_dx,p=2) for part_dx in partial_dx_native]
    lossg = torch.log(torch.prod(torch.FloatTensor(part_dx_native),dim=0) +1)
    
    lossd = (torch.log((E[1]+1) / (E[0]+1) +1)).mean()
    
    # lossc = preform_energy_optimization(X_decoy,partial_dx_decoy)
    # print(f"loss g: {round(lossg.item(),4)} loss d: {round(lossd.item(),4)}")
    return lossd+lossg , lossd, lossg,E[1],E[0]

def main():
    print('***Start main function***')
    print('***load the data with dataloader***')
    d_params = data_params(num_workers =CFG.num_workers, batch_size=CFG.batch_size,cuda=CFG.cuda,constraint=CFG.constraint,debug=CFG.debug)
    train_loader, valid_loader,test_loader = fetch_dataloader(data_dir=CFG.data_path, params=d_params)
    
    # Build the model
    print('***Build the model***')
    m_params = model_params(embedding_size = CFG.embedding_size,filters = CFG.filters, layers = CFG.num_layers,
                             cord_size = CFG.coords_emb,h = CFG.h,device=CFG.device)
    model = PEM(dim_in=36,dim_h=64,dim_out=36,layers=3,model_type='GAT',gaussian_coef=CFG.gaussian_coef).to(CFG.device)
    
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    # Run training
    # print('***Start training***')
    # training(model, optimizer, train_loader,valid_loader, CFG.device,CFG.N)
    load_checkpoint(CFG.model_path+"3_final_model.pt", model, optimizer,CFG.device)
    validation(model, valid_loader,CFG.device,3 , CFG.N, optimizer , type = 'robust')
    # validation(model, valid_loader,CFG.device,3 , CFG.N, optimizer, type = 'soft')
    
    return 1

    
def print_par(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
   
if __name__ == '__main__':
    if len(sys.argv)>1:
        CFG.model_path = sys.argv[1]
        CFG.constraint = int(sys.argv[2])
    main()