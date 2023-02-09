from model.data_loader import PEFDataset,fetch_dataloader
from model.data_loader import params as data_params
from model.model_cfg import CFG
from model.net import ProteinEnergyNet
from model.net import params as model_params
from train_utils import save_checkpoint
import torch
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc
import time


# define one epoch train
def training (model, optimizer, dataloader, device,N):
    """
    Training function for the model.
    Args:
        model (torch.model): model to train
        optimizer (torch.optim): optimizer to use
        dataloader (torch.utils.data.DataLoader): dataloader for the training set
        device (torch.device): device to use ('cpu' or 'cuda' or 'mps')
        N (int): The number of iterations for the iterative optimization
    """
    model.train()
    
    for epoch in range(CFG.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        torch.cuda.empty_cache()
        gc.collect()
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                # set progress bar description
                tepoch.set_description(f"Epoch {epoch}")
                
                # get the inputs; data is a list of [inputs, labels]   
                seq, id, Xd,Xn, mask, nativemask, esm_embed = data
                Xd = Xd.to(device)
                Xn = Xn.to(device)
                esm_embed = esm_embed.to(device)
                Xd.requires_grad = True
                Xn.requires_grad = True
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(Xd,Xn,esm_embed)
                loss = criterion(outputs,Xd,Xn,N,CFG.h)
                print(loss.item())
                loss.backward()
                # print_par(model) # print the parameters of the model
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
                
                torch.cuda.empty_cache()
                gc.collect()
                # update the progress bar
                tepoch.set_postfix(loss=round(loss.item(),3))
                # save the model
                save_checkpoint(epoch, model, optimizer,loss,CFG.model_path)
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

def criterion(E,X_native,X_decoy,N,h):
    """
    The loss function for the model coressponds to 3 main losses:
    1. lossg: the partial derivateve of the energy with respect to the native structure
    2. lossd: the energy of the native structure divided by the decoy energy
    3. lossc: After preforming an iterative optimization on the decoy structure, by using the energy partial derivative on the decoy structure, 
              we calculate the dRMSD of the end and the start of the optimization.

    Args:
        E (tensor): A tensor containing the energy of the native and the decoy structure Exd,Exn [batch_size*3,2]
        X_native (tensor): A tensor containing the native structure [batch_size,seq_len,4,3]
        X_decoy (tensor): A tensor containing the decoy structure [batch_size,seq_len,4,3]
        N (int): The number of iterations for the iterative optimization
        h (float): The step size for the numerical derivative
    output:
        loss (tensor): The loss of the model
    """
    # print('***Start criterion function***')
    # partial_dx_decoy = torch.autograd.grad(E[:,0].sum(),X_decoy,create_graph=True)[0]
    # partial_dx_native = torch.autograd.grad(E[:,1].sum(),X_native,create_graph=True)[0]
    # print('***End derivative calc function***')
    batch_size3,_ = E.shape
    batch_size = int(batch_size3/3)
    partial_dx_native = (E[batch_size:2*batch_size,0] - E[2*batch_size:3*batch_size,0])/(2*h)
    lossg = torch.norm(partial_dx_native,p=2)
    
    lossd = (E[:,1] / E[:,0]).mean()
    
    # lossc = preform_energy_optimization(X_decoy,partial_dx_decoy)
    
    return (lossd+lossg)

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
    training(model, optimizer, train_loader, CFG.device,CFG.N)
    
    return 1

def test(optimizer,model):
    x_test = torch.randn((2,10,4,3),requires_grad=True).to(CFG.device)
    x_test_native = torch.randn((2,10,4,3),requires_grad=True).to(CFG.device)
    x_test_embed = torch.randn(2,10,480).to(CFG.device)
    
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    y_pred = model(x_test,x_test_native,x_test_embed)
    y_pred.sum().backward(retain_graph=True)
    loss = criterion(y_pred,x_test_native,x_test)
    loss.backward()
    
    optimizer.step()
    
def print_par(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
   
if __name__ == '__main__':
    main()