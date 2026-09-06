import torch
import torch.nn.functional as F
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from model.model_cfg import CFG
# W6 (item 19): the amino-acid descriptor block. Depends ONLY on the committed CSV --
# never on rdkit, which must never enter esm2_env_py38.
from aa_descriptors import (aa_descriptor_mode, desc_or_none, descriptor_dim,
                            keeps_one_hot)

# amino acid one hot map
AA_MAP = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}

def save_checkpoint(epoch, model, optimizer, loss, val_loss, path, rank=0):
    """
    Save the model check point
    inputs:
        epoch (int): number of epoch
        model(torch.model): model
        optimizer(torch.optim): torch optimizer
        loss(tensor) : loss function value
        val_loss(tensor) : validation loss function value
        path (str) : path to save the model
        rank (int): process rank; only rank 0 writes
    """
    if rank != 0:
        return
    os.makedirs(str(Path(path).parent), exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'valid_loss': val_loss,
            }, path)
   
def load_checkpoint(path,model,optimizer=None,device=CFG.device):
    """
    Load the model check point
    inputs:
        path (str) : path to load the model
        model(torch.model): model
        optimizer(torch.optim): torch optimizer
        device (str) : device to load the model
    """ 
    
    model_dict = torch.load(path,map_location=device,weights_only=False)
    print(f"Loaded model from {path}")
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(model_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
    return model,optimizer,model_dict['epoch'],model_dict['loss'],model_dict['valid_loss']
    
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

# --- W5: solvation and burial -------------------------------------------------
# The dominant force in protein folding had no representation in this model at all.
# DeepDDG (5700 curated mutations, beating eleven methods) found the SASA of the
# mutated residue to be its single most important input.
#
# Kyte-Doolittle hydropathy, index order VERIFIED against AA_MAP above
# (alphabetical ACDEFGHIKLMNPQRSTVWY), not assumed.
_KD = torch.tensor([1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3])
_KD_NORM = (_KD + 4.5) / 9.0          # -> [0,1]
_BURIAL_CAP = 30.0                    # a CONSTANT, never N -- see below
_BURIAL_R = 10.0


def compute_burial(x, mask, radius=_BURIAL_R, cap=_BURIAL_CAP):
    """Cbeta neighbour-count burial -> [N,1] in [0,1].

    x is [N,4,3] with atom order (N, CA, C, CB); glycine has no CB and the loader
    stores CA there, which is the standard substitute.

    Normalised by a CONSTANT, never by N. Burial is a LOCAL quantity -- neighbours
    within 10 A do not scale with chain length -- and dividing by N would inject a
    per-protein length confound into the one feature meant to fix a per-protein
    problem. An earlier implementation did exactly that.
    """
    cb = x[:, 3, :]
    d = torch.cdist(cb, cb)
    v = (mask > 0).float()
    w = (d < radius).float() * v.unsqueeze(0) * v.unsqueeze(1)
    w = w - torch.diag_embed(torch.diagonal(w))        # exclude self
    return (w.sum(1, keepdim=True) / cap).clamp(0.0, 1.0) * v.unsqueeze(1)


def compute_hse(x, mask, radius=_BURIAL_R, cap=_BURIAL_CAP):
    """Half-sphere exposure: neighbours in the CA->CB hemisphere only.

    HSE is the standard neighbour-count burial measure and is DIRECTION-AWARE,
    where a raw count is not. It needs only CA and CB, which is exactly what this
    backbone-only dataset has. Run as an arm against compute_burial.
    """
    ca, cb = x[:, 1, :], x[:, 3, :]
    up = cb - ca
    up = up / up.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    diff = cb.unsqueeze(0) - cb.unsqueeze(1)           # [N,N,3] j - i
    d = diff.norm(dim=-1)
    v = (mask > 0).float()
    near = (d < radius).float() * v.unsqueeze(0) * v.unsqueeze(1)
    near = near - torch.diag_embed(torch.diagonal(near))
    same_side = ((diff * up.unsqueeze(1)).sum(-1) > 0).float()
    return ((near * same_side).sum(1, keepdim=True) / cap).clamp(0.0, 1.0) * v.unsqueeze(1)


def solvation_features(x, one_hot, mask, folded=True):
    """[N,3] = burial, hydrophobicity, burial*hydrophobicity.

    Burial is ZERO in the unfolded state. An extended chain buries nothing, and
    that DIFFERENCE is the hydrophobic driving force. A previous implementation
    used the same coordinates in both states, so the column was bit-identical and
    cancelled exactly in E_u - E_f -- the feature could not express the thing it
    was named for.

    The product term matters: a linear layer cannot construct a product from its
    factors, and buried x hydrophobic is what carries the driving force.
    """
    hyd = one_hot @ _KD_NORM.to(one_hot.device).to(one_hot.dtype).unsqueeze(1)
    if folded:
        fn = compute_hse if getattr(CFG, 'burial_mode', 'count') == 'hse' else compute_burial
        bur = fn(x, mask).to(one_hot.dtype)
    else:
        bur = torch.zeros_like(hyd)
    return torch.cat([bur, hyd, bur * hyd], dim=1)


def _solv_or_none(x, one_hot, mask, folded):
    """Returns the [N,3] block, or None when the lever is off (bit-identical)."""
    if not getattr(CFG, 'burial_features', False):
        return None
    return solvation_features(x, one_hot, mask, folded=folded)


def _desc_or_none(one_hot):
    """W6: the [N,K] descriptor block, or None when --aa_descriptors none (default)."""
    return desc_or_none(one_hot, CFG)


def _onehot_block(one_hot):
    """The trailing 20 columns. Under pca16_only the alphabet is removed by ZEROING
    rather than deleting: hydro_net slices emb and one-hot RIGHT-ANCHORED, so deleting
    the columns would silently re-point x[:, -20:] into the ProtT5 embedding and slide
    the 1024-wide llm window left into Fb. Every shape check would still pass and every
    number would be wrong. Zeroing carries the same information -- a constant column is
    a bias the following Linear already has."""
    if keeps_one_hot(aa_descriptor_mode(CFG)):
        return one_hot
    return torch.zeros_like(one_hot)


def get_graph(x, one_hot, emb, mask, gaussian_coef=CFG.gaussian_coef):
    """Get graph representation of protein 
    Args:
        x (torch.Tensor): Input tensor representing the coordinates of atoms in the protein structure. Shape: [N, 4,3].
        one_hot (torch.Tensor): One-hot encoded tensor representing additional categorical features for each atom. Shape: [N, 20].
        emb (torch.Tensor): Tensor representing learned embedding features for each atom. Shape: [N, E].
        mask (torch.Tensor): Binary mask tensor indicating which atoms should be considered in the graph. 0 indicates masked out, 1 indicates active. Shape: [N].
        gaussian_coef (float, optional): Coefficient used in the Gaussian kernel to compute distances between atoms. Default value is taken from configuration (CFG.gaussian_coef).
    Returns:
        torch.Tensor: Concatenated tensor containing distance matrix features, bonded features, embedding features, and one-hot encoded features. Shape: [N, 16 + 32 + emb_size].
    """
    D = get_dist_matrix(x) # N,N,16
    D = torch.relu(torch.exp(gaussian_coef*D**2))
    # remove masks values
    mask_index = torch.where(mask == 0)
    D[mask_index[0],:,:] = 0
    D[:,mask_index[0],:] = 0
    # get bonded features
    Fb = get_bonded_features(D) # N,32
    # sum over the atoms
    D = D.sum(dim=1) #N,16
    D = F.normalize(D,p=2,dim=0)
    emb = F.normalize(emb,p=2,dim=0)
    _S = _solv_or_none(x, one_hot, mask, folded=True)
    _Dsc = _desc_or_none(one_hot)                     # W6: [N,K] or None
    _oh = _onehot_block(one_hot)                      # W6: zeroed under pca16_only
    _blocks = [D, Fb] + ([] if _S is None else [_S]) +               ([] if _Dsc is None else [_Dsc]) + [emb, _oh]
    Fh = torch.cat(_blocks, dim=1)  # N,16+32[+3][+K]+emb_size+20
    
    return Fh


def rbf_expand(d, n=16, lo=0.0, hi=20.0):
    """W7: expand a distance into a bank of n radial basis functions.

    CONCATENATE the bank. NEVER sum it. A previous implementation summed the M
    responses back to the original width and a 5 A contact and a 15 A non-contact
    both returned 4.649 -- the kernel went flat past 5 A and the model was blind to
    distance. The assertion k(2A) > k(8A) > k(15A) catches that completely, and is
    run by scripts/gate_w7.py before any training.
    """
    centers = torch.linspace(lo, hi, n, device=d.device, dtype=d.dtype)
    width = (hi - lo) / n
    return torch.exp(-((d.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))

def _unfolded_emb(emb):
    """U2 -- make the unfolded reference state fold-blind.

    An unfolded chain has no fold; it should not know which family it came from.
    Fold-family memorisation is the diagnosed failure (WT dG correlation 0.86
    in-distribution, ~0.07 out-of-distribution).

    Measured by the W0 channel ablation on 28 test proteins, calib_ctrl_repro2 e14:
    zeroing this block in the unfolded pass drops var(E_u) across proteins to 0.331
    of baseline and corr(E_u, wt_err) from 0.420 to 0.119. The Flory coil, by
    contrast, RAISES var(E_u) to 1.175 of baseline -- it touches D and Fb, which
    were not the cause. Hence factor D of the calibration factorial is this flag.

    Modes:
      full  current behaviour, returned unchanged (default; bit-identical)
      zero  the block is all zeros
      mean  per-column mean over residues, broadcast back: keeps whatever global
            scale ProtT5 contributes while removing per-residue identity. If zero
            hurts but mean helps, the embedding was carrying scale, not identity.

    Applied BEFORE F.normalize(emb, p=2, dim=0), so that 'mean' is normalised the
    same way every other input is. For 'zero' the order is immaterial: F.normalize
    of a zero tensor is zero (it clamps the denominator by eps).
    """
    mode = getattr(CFG, 'unfolded_emb', 'full')
    if mode == 'full':
        return emb
    if mode == 'zero':
        return torch.zeros_like(emb)
    if mode == 'mean':
        return emb.mean(dim=0, keepdim=True).expand_as(emb)
    raise ValueError("CFG.unfolded_emb must be one of full|zero|mean; got %r" % (mode,))


def get_unfolded_graph(x, one_hot, emb, mask, gaussian_coef=CFG.gaussian_coef):
    """Get graph representation of a unfolded protein.

    Baseline (CFG.flory_unfolded=False): keep only the tridiagonal (chain-local) contacts via
    zero_except_udiagonal — the historical reference state (no long-range 3D structure).

    Offset-attack Lever D (CFG.flory_unfolded=True): replace the stark tridiagonal mask with an
    analytic Flory random-coil reference where the expected inter-residue distance scales as
    d(i,j) = b*|i-j|^nu (polymer physics). Value-only: the output tensor SHAPE is identical to
    the baseline. Default OFF reproduces the tridiagonal baseline bit-for-bit.
    """
    if getattr(CFG, 'flory_unfolded', False):
        return _flory_unfolded_graph(x, one_hot, emb, mask, gaussian_coef)
    D = get_dist_matrix(x) # N,N,16
    D = torch.relu(torch.exp(gaussian_coef*D**2))
    # remove masks values
    mask_index = torch.where(mask == 0)
    D[mask_index[0],:,:] = 0
    D[:,mask_index[0],:] = 0
    # remove all values except the diagonal
    D = zero_except_udiagonal(D)
    # get bonded features
    Fb = get_bonded_features(D) # N,32
    # sum over the atoms
    D = D.sum(dim=1) #N,16
    D = F.normalize(D,p=2,dim=0)
    emb = F.normalize(_unfolded_emb(emb),p=2,dim=0)   # U2: unfolded pass only
    _S = _solv_or_none(x, one_hot, mask, folded=False)   # W5: burial is ZERO unfolded
    # W6: the descriptor block is IDENTICAL in both states, by design -- valine is
    # valine in the coil. It therefore cancels in E_u - E_f for a linear readout and
    # contributes only through the nonlinearity and message passing. That is honest
    # physics, NOT the W5 cancellation bug: burial was supposed to differ and did not;
    # chemistry is supposed not to differ, and does not.
    _Dsc = _desc_or_none(one_hot)
    _oh = _onehot_block(one_hot)
    _blocks = [D, Fb] + ([] if _S is None else [_S]) +               ([] if _Dsc is None else [_Dsc]) + [emb, _oh]
    Fh = torch.cat(_blocks, dim=1)

    return Fh


def _flory_unfolded_graph(x, one_hot, emb, mask, gaussian_coef):
    """Lever D — analytic Flory random-coil unfolded reference.

    Instead of zeroing off-tridiagonal contacts, model the unfolded chain as an ideal random
    coil: the expected distance between residues i and j scales as d(i,j) = b*|i-j|^nu. We build
    a per-atom-pair distance block matching get_dist_matrix's width (16), apply the SAME Gaussian
    kernel and masking as the baseline path, then reduce identically. The output tensor shape is
    byte-identical to the baseline unfolded graph (value-only lever, no new parameters).

    Reference: DeepPEF_v5/training/train_utils.py:flory_reference (simplified to match Shahar's
    get_unfolded_graph exactly — no burial / RBF bank / AFRC, none of which exist in this tree).
    """
    nu = float(getattr(CFG, 'flory_nu', 0.5))
    # Fail-fast: nu outside (0,1] is unphysical for a polymer coil scaling exponent.
    if not (0.0 < nu <= 1.0):
        raise ValueError(f"[Lever D] flory_nu must be in (0, 1]; got {nu}. "
                         f"(0.5 = ideal chain, ~0.588 = self-avoiding walk.)")
    N, N_atoms, _ = x.shape
    n_atom_dist = N_atoms * N_atoms  # 16, matches get_dist_matrix last-dim width
    dev = x.device
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # |i-j|, [N,N]
    # Effective bond length b: mean CA-CA neighbor distance so the coil is scaled to THIS protein
    # rather than an arbitrary constant. CA is atom index 1 (N=0, CA=1, C=2, CB=3).
    ca = x[:, 1, :]
    if N > 1:
        b = torch.linalg.norm(ca[1:] - ca[:-1], dim=-1).mean()
        b = torch.clamp(b, min=1e-3)
    else:
        b = torch.tensor(3.8, device=dev)
    d_coil = b * torch.pow(sep + 1e-6, nu)  # [N,N] analytic expected coil distance
    # Broadcast the scalar coil distance across all atom-atom channels (coil model is residue-level).
    D = d_coil.unsqueeze(-1).expand(N, N, n_atom_dist).contiguous()  # [N,N,16] RAW distances
    # SAME kernel as folded/baseline unfolded path (train_utils.py get_graph line 208).
    D = torch.relu(torch.exp(gaussian_coef * D ** 2))
    # remove masks values (identical to baseline)
    mask_index = torch.where(mask == 0)
    D[mask_index[0], :, :] = 0
    D[:, mask_index[0], :] = 0
    # NOTE: no zero_except_udiagonal — the coil IS the full unfolded reference now.
    Fb = get_bonded_features(D)  # N,32
    D = D.sum(dim=1)  # N,16
    D = F.normalize(D, p=2, dim=0)
    emb = F.normalize(_unfolded_emb(emb), p=2, dim=0)   # U2: unfolded pass only
    _S = _solv_or_none(x, one_hot, mask, folded=False)   # W5: burial is ZERO unfolded
    _Dsc = _desc_or_none(one_hot)                        # W6: state-independent
    _oh = _onehot_block(one_hot)
    _blocks = [D, Fb] + ([] if _S is None else [_S]) +               ([] if _Dsc is None else [_Dsc]) + [emb, _oh]
    Fh = torch.cat(_blocks, dim=1)

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

def wandb_config(wandb, model, optimizer, scheduler, dataloader,
                 model_path = CFG.model_path,reg_alpha = CFG.reg_alpha,gaussian_coef = CFG.gaussian_coef,
                 lr = CFG.lr,num_layers = CFG.num_layers,dropout_rate = CFG.dropout_rate,precision = CFG.precision):
    """wandb_config """
    if not CFG.debug:
        wandb.config.learning_rate = optimizer.param_groups[0]['lr']
        wandb.config.batch_size = CFG.batch_size
        wandb.config.epochs = CFG.num_epochs
        wandb.config.optimizer = type(optimizer).__name__
        wandb.config.scheduler = type(scheduler).__name__
        wandb.config.model = type(model).__name__
        wandb.config.dataset = type(dataloader.dataset).__name__
        wandb.config.wd = CFG.wd
        wandb.config.model_path = model_path
        wandb.config.reg_alpha = reg_alpha
        wandb.config.gaussian_coef = gaussian_coef
        wandb.config.lr = lr
        wandb.config.num_layers = num_layers
        wandb.config.dropout_rate = dropout_rate
        wandb.config.precision = precision
        

def zero_except_udiagonal(D):
    """Zero all values except the diagonal and its neighbors"""
    n_range = torch.arange(D.shape[0])
    f1, f2 = D[n_range[:-1], n_range[1:]], D[n_range[1:], n_range[:-1]]
    diag = D[n_range, n_range]
    D[:, :, :] = 0
    D[n_range[:-1], n_range[1:]] = f1
    D[n_range[1:], n_range[:-1]] = f2
    D[n_range, n_range] = diag
    return D

def get_one_hot(seq):
  """get one hot from sequence"""
  seq_one_hot = torch.zeros((len(seq),20))
  for i,a in enumerate(seq):
    if a in AA_MAP:
        seq_one_hot[i][AA_MAP[a]] = 1
  return seq_one_hot

def add_cb(crd_coords):
        """
        Add the Cbeta atom to the coordinates
        Args:
            crd_coords (tensor): tensor of shape [n_residues,3,3]

        Returns:
            crd_coords: tensor shape [n_residues,4,3]
        """
        # Get the coordinates of the backbone atoms
        N, CA, C = crd_coords[:, 0], crd_coords[:, 1], crd_coords[:, 2]
        # CB = CA + c1*(N-CA) + c2*(C-CA) + c3* (N-CA)x(C-CA)
        CAmN = N - CA
        # CAmN = CAmN / torch.sqrt(CAmN ** 2).sum(dim=2, keepdim=True)
        CAmC = C - CA
        # CAmC = CAmC / torch.sqrt(CAmC ** 2).sum(dim=2, keepdim=True)
        ANxAC = torch.cross(CAmN, CAmC, dim=1)

        A = torch.cat((CAmN.reshape(-1, 1), CAmC.reshape(-1, 1), ANxAC.reshape(-1, 1)), dim=1)
        c = torch.tensor([0.5507, 0.5354, -0.5691]) / 100  # torch.tensor([1.1930, 1.2106, -2.7906]) #
        b = (A @ c).reshape(-1,3)
        CB = CA - b
      
        # Add Cbeta coordinates to existing coordinates array
        crd_coords = torch.cat((crd_coords, CB.unsqueeze(1)), dim=1)
        return crd_coords
    
def Add_random_step(X, h = CFG.h):
    """Add random step to the coordinates
    Args:
        X (tensor): tensor of coordinats of shape [n_residues,4,3]
        corrds_index (int): number of coordinates to add
        h (float): step size
    """
    # Add random step to the coordinates
    v = torch.randn(X.shape).to(X.device)
    X1 = X + h * v
    X2 = X - h * v
    return X1, X2

def print_max_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            max_grad = param.grad.data.abs().max().item()
            print(f"Max gradient for {name}: {max_grad}")
            
            
def get_noised_proteins(data,device, config = CFG):
    """
    Returns a noised version of the protein data.
    """
    id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy,proT5_cycle1, proT5_cycle2 = data
            
    # wild type and mutant type
    Xjf = crd_backbone.to(device) # wilde type structure folded
    Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
    Xcd = torch.clone(crd_decoy).to(device) # decoy structure
    Xcy1 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy2 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    
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
    
    if Xjf.shape[1] > config.seq_len:  # if the protein is too long, skip it (GPU memory limitation)
        return None,None,None,None,None,None,None
    #emb = torch.cat((esm_embed,seq),dim=2)
    emb = seq_one_hot.to(device)
    emb_decoy = seq_decoy.to(device)
    # get the cycle permutation
    cycle_emb1 = get_one_hot(seq[0][-1] + seq[0][:-1]).to(device)
    cycle_emb2 = get_one_hot(seq[0][1:] + seq[0][0]).to(device)
    
    # move proT5_emb to device
    proT5_emb_decoy, proT5_emb, proT5_cycle1, proT5_cycle2 = proT5_emb_decoy.to(device), proT5_emb.to(device), proT5_cycle1.to(device), proT5_cycle2.to(device)
    
    # squeeze the data
    Xd, Xjf, Xju, Xcd, Xdu, Xcy1, Xcy2 = Xd.squeeze(), Xjf.squeeze(), Xju.squeeze(), Xcd.squeeze(), Xdu.squeeze(), Xcy1.squeeze(), Xcy2.squeeze()
    emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
    mask_decoy, mask, mask_crd_decoy= mask_decoy.squeeze(), mask.squeeze(), mask_crd_decoy.squeeze()
    proT5_emb_decoy, proT5_emb, proT5_cycle1, proT5_cycle2 = proT5_emb_decoy.squeeze(), proT5_emb.squeeze(), proT5_cycle1.squeeze(), proT5_cycle2.squeeze()
    
    # get folded graph  
    Xjf = get_graph(Xjf, emb, proT5_emb, mask, gaussian_coef=config.gaussian_coef)
    # get unfolded graph
    Xju = get_unfolded_graph(Xju, emb, proT5_emb, mask, gaussian_coef=config.gaussian_coef)
    # get decoy graph
    Xd, Xcd, Xdu = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy, gaussian_coef=config.gaussian_coef), get_graph(Xcd, emb, proT5_emb, mask_crd_decoy, gaussian_coef=config.gaussian_coef), get_unfolded_graph(Xdu, emb_decoy, proT5_emb_decoy, mask_decoy, gaussian_coef=config.gaussian_coef)
    # Add cycle permutation
    Xcy1 = get_graph(Xcy1, cycle_emb1, proT5_cycle1, mask, gaussian_coef=config.gaussian_coef)
    Xcy2 = get_graph(Xcy2, cycle_emb2, proT5_cycle2, mask, gaussian_coef=config.gaussian_coef)
    # Xjf.requires_grad = True
    # create a batch of Xjf,Xkf,Xju,Xku,x_decoy
    Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2 = Xjf.unsqueeze(0),Xju.unsqueeze(0),Xd.unsqueeze(0), Xcd.unsqueeze(0), Xdu.unsqueeze(0),Xcy1.unsqueeze(0),Xcy2.unsqueeze(0)

    return Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2