import torch
import torch.nn.functional as F
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from model.model_cfg import CFG

def save_checkpoint(epoch, model, optimizer,loss,val_loss,path):
    """
    Save the model check point
    inputs:
        epoch (int): number of epoch
        model(torch.model): model
        optimizer(torch.optim): torch optimizer
        loss(tensor) : loss function value
        val_loss(tensor) : validation loss function value
        path (str) : path to save the model
    """
    os.makedirs(str(Path(path).parent), exist_ok=True)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'valid_loss': val_loss,
            }, path)
   
def load_checkpoint(path,model,optimizer,device):
    """
    Load the model check point
    inputs:
        path (str) : path to load the model
        model(torch.model): model
        optimizer(torch.optim): torch optimizer
        device (str) : device to load the model
    """ 
    
    dict = torch.load(path,map_location=device)
    print(f"Loaded model from {path}")
    # print(f"Epoch: {dict['epoch']},loss: {dict['loss']},valid_loss: {dict['valid_loss']}")
    model.load_state_dict(dict['model_state_dict'])
    optimizer.load_state_dict(dict['optimizer_state_dict'])
    return model,optimizer,dict['epoch'],dict['loss'],dict['valid_loss']
    
def validation_plots(Exd,Exn,seq_len,type,epoch):
    """
    Plot the validation data
    inputs:
        Exd (tensor) : validation data
        Exn (tensor) : validation data
        seq_len (int) : sequence length
        type (str) : type of the plot
    """
    # create the directory if not exist
    os.makedirs(CFG.results_path+'plots', exist_ok=True)
    plot_dir = CFG.results_path+'plots'
    # Plot the validation data
    fig, ax = plt.subplots()
    ax.set_title(f'Validation data for {type},number of sequences: {len(Exd)}')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Energy')
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.plot(seq_len, Exd, marker='o', linestyle='', ms=3, label='decoy')
    ax.plot(seq_len, Exn, marker='o', linestyle='', ms=3, label='native')
    ax.legend()

    #plt.show()
    plt.savefig(f'{plot_dir}/E_len_{type}.png')
    
    plt.close()
    
    # plot validation energy delta
    delta_E = np.array([Exd[i]-Exn[i] for i in range(len(seq_len))])
    seq_len = np.array(seq_len)
    fig, ax = plt.subplots()
    ax.set_title(f'Validation data for {type},number of sequences: {len(Exd)}')
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Energy delta log')
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    ax.plot(seq_len[delta_E>=0], np.log(delta_E[delta_E>=0]+1), marker='o', linestyle='', ms=3, label='positive')
    ax.plot(seq_len[delta_E<0], -1*np.log(-1*delta_E[delta_E<0] +1), marker='o', linestyle='', ms=3, label='negative')
    ax.legend()

    #plt.show()
    plt.savefig(f'{plot_dir}/epoch-{epoch}-Edelta_len_{type}.png')
    
    plt.close()

def mix_A_acid(seq_one_hot,emb,mask,val_type,device):
    """ mix the amino acid sequence"""
    if val_type == 'robust' or val_type == 'train':
        mix_index = torch.randperm(seq_one_hot.shape[1])
        seq_decoy = torch.clone(seq_one_hot[:,mix_index,:]).to(device)
        mask_decoy = torch.clone(mask[:,mix_index]).to(device)
        emb_decoy = torch.clone(emb[:,mix_index,:]).to(device)
    else: 
        mix_index = torch.randperm(seq_one_hot.shape[1])[:2]
        mask_decoy = torch.clone(mask).to(device)
        seq_decoy = torch.clone(seq_one_hot).to(device)
        emb_decoy = torch.clone(emb).to(device)
        seq_decoy[:,mix_index[0],:], seq_decoy[:,[1],:] = seq_decoy[:,mix_index[1],:], seq_decoy[:,[0],:]
        mask_decoy[:,mix_index[0]] ,mask_decoy[:,[1]] = mask_decoy[:,mix_index[1]], mask_decoy[:,[0]]
        emb_decoy[:,mix_index[0],:], emb_decoy[:,[1],:] = emb_decoy[:,mix_index[1],:], emb_decoy[:,[0],:]
    return seq_decoy, mask_decoy, emb_decoy
        
def pad_image(image,desired_size = (500,23)):
    """ pad the image to desired size"""
    # Pad the image to 500x500
    desired_height, desired_width = desired_size
    pad_height = desired_height - image.size(0)
    pad_width = desired_width - image.size(1)

    # Compute the amount of padding on each side
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    # Apply padding using torch.nn.functional.pad
    padded_image = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
    
    return padded_image

def interpolate_image(Xn):
    """ interpolate the image"""
    # Interpolate the image to 500x500
    Xn = F.interpolate(Xn.unsqueeze(0), size=500)
    Xn = Xn.squeeze(0)
    return Xn

def diff_data(model, optimizer, dataloader, device,epoch,N,valid_loader):
    all_Xn = torch.tensor([])
    all_Xn_int = torch.tensor([])
    n = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for index, data in enumerate(tepoch):
            # set progress bar description
            tepoch.set_description(f"Epoch {epoch}")
            # Clean the GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            # get the inputs; data is a list of [inputs, labels]   
            id, Xn, mask, seq_one_hot, seq,ang_backbone, ang, dist_matrix = data
            
            mask_flag = torch.where(mask==1,0,1).sum()==0
            if((seq_one_hot.shape[1] <= 500) & mask_flag):
                Xn = Xn.squeeze()
                seq_one_hot = seq_one_hot.squeeze() 
                Xn = Xn[:,1,:] # take only the first 128 residues
                mean, std, var = torch.mean(Xn), torch.std(Xn), torch.var(Xn) 
                Xn  = (Xn-mean)/std
                Xn = torch.cat((Xn,seq_one_hot),dim=1)
                Xn_pad = pad_image(Xn)
                Xn_interpolat = interpolate_image(Xn.swapaxes(0,1)).swapaxes(0,1)
                all_Xn_int = torch.cat((all_Xn_int,Xn_interpolat.unsqueeze(dim=0)),dim=0)
                all_Xn = torch.cat((all_Xn,Xn_pad.unsqueeze(dim=0)),dim=0)
                n = n+1
                # torch.save(all_Xn,"./all_Xn.pt")
            tepoch.set_postfix({"n":n})
        
        torch.save(all_Xn_int,"./all_Xn_int.pt")
        torch.save(all_Xn,"./all_Xn_padded.pt")

def get_graph(x, one_hot, emb, mask):
    """Get graph representation of protein"""
    D = get_dist_matrix(x) # N,N,16
    D = torch.relu(torch.exp( CFG.gaussian_coef*D**2))
    # remove masks values
    mask_index = torch.where(mask == 0)
    D[mask_index[0],:,:] = 0
    D[:,mask_index[0],:] = 0
    # get bonded features
    Fb = get_bonded_features(D) # N,32
    # sum over the atoms
    D = D.sum(dim=1) #N,16
    D = F.normalize(D,p=2,dim=0)
    Fh = torch.cat([D,Fb,emb,one_hot],dim=1) #N,16+32+emb_size
    
    return Fh

def get_bonded_features(D):
    """Get bonded features from distance matrix"""
    n_range = torch.arange(D.shape[0])
    # get above and below diagonal
    f1, f2 = D[n_range[:-1], n_range[1:]], D[n_range[1:], n_range[:-1]]
    # pad with zeros
    zero_row = torch.zeros((1, D.shape[-1])).to(D.device)
    f1 = torch.cat([f1, zero_row], dim=0)
    f2 = torch.cat([zero_row, f2], dim=0)
    # concat features
    Fb = torch.cat([f1, f2], dim=-1)
    return Fb # N,32   
    
      
def get_dist_matrix(Xd):
    """
    Return the node distence matrix
    Args:
        Xd (tensor):X embeded [n_nodes ,num_atoms=4,new_cords_size]
    Returns:
        tensor : [n_nodes,n_nodes ,atom_dist=16] tensor
    """
    N_residu, N_atoms, coords_size = Xd.shape
    Xd = Xd.reshape(N_residu*N_atoms, coords_size)
    D = torch.cdist(Xd, Xd, p=2)
    D = D.reshape(N_residu, N_atoms, N_residu, N_atoms)
    D = torch.swapaxes(D,1,2)
    D = D.reshape(N_residu, N_residu, N_atoms*N_atoms)
    return D

def add_gaussian_noise(X_native,sigma):
    """ add gaussian noise to the native structure"""
    device = X_native.device
    noise = (torch.randn(X_native.shape)*sigma).clone().to(device)
    X_decoy = X_native + noise
    return X_decoy