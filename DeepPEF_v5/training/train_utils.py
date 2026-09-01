import torch
import torch.nn.functional as F
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
from model.model_cfg_v5 import CFG

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

def get_graph(x, one_hot, emb, mask, gaussian_coef=CFG.gaussian_coef, emb_delta=None):
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
    # apply distance kernel: single Gaussian (baseline) or RBF bank summed back to width 16
    # (Lever B, dimension-preserving; rbf_centers==0 -> exact baseline exp(coef*D^2)).
    D = apply_distance_kernel(D, gaussian_coef)
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
    Fh = maybe_concat_burial(x, D, Fb, emb, one_hot, mask, emb_delta=emb_delta) #N, [16+32(+burial)+emb(+delta)+20]

    return Fh

def get_unfolded_graph(x, one_hot, emb, mask, gaussian_coef=CFG.gaussian_coef, emb_delta=None):
    """Get graph representation of a unfolded protein.

    Baseline (CFG.flory_unfolded=False): keep only the tridiagonal (chain-local) contacts —
    the unfolded state has no long-range 3D structure. This is the historical reference state.

    Lever D (CFG.flory_unfolded=True): replace the chain-local reference with a Flory
    random-coil reference where the inter-residue distance scales as d_ij ~ b*|i-j|^nu (polymer
    physics). This is value-only: the output tensor SHAPE is identical to the baseline.
    """
    if getattr(CFG, 'flory_unfolded', False):
        return flory_reference(x, one_hot, emb, mask, gaussian_coef, emb_delta=emb_delta)
    D = get_dist_matrix(x) # N,N,16
    D = apply_distance_kernel(D, gaussian_coef)   # Lever B kernel (baseline single Gaussian)
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
    emb = F.normalize(emb,p=2,dim=0)
    Fh = maybe_concat_burial(x, D, Fb, emb, one_hot, mask, emb_delta=emb_delta) #N, [16+32(+burial)+emb(+delta)+20]

    return Fh


def flory_reference(x, one_hot, emb, mask, gaussian_coef, emb_delta=None):
    """Lever D — Flory random-coil unfolded reference.

    Instead of zeroing off-tridiagonal contacts, model the unfolded chain as an ideal random
    coil: the expected distance between residues i and j scales as b*|i-j|^nu. We build a
    per-atom-pair distance block of width 16 (matching get_dist_matrix's layout) from this
    polymer model, apply the SAME Gaussian/RBF kernel and masking as the folded path, then
    reduce identically. Output shape == baseline unfolded graph.

    Lever D+ (CFG.flory_afrc=True): replace the analytic b*|i-j|^nu with the AFRC
    (Analytical Flory Random Coil) SEQUENCE-SPECIFIC ensemble-average distance map — the same
    random-coil-distogram reference IFUM (Lee et al., Nat. Commun. 2026) used. Falls back to
    the analytic coil if the `afrc` package is not installed.
    """
    N = x.shape[0]
    nu = float(getattr(CFG, 'flory_nu', 0.5))
    # Lever D fail-fast: nu outside (0, 1] is unphysical for a polymer coil scaling exponent.
    if not (0.0 < nu <= 1.0):
        raise ValueError(f"[Lever D] flory_nu must be in (0, 1]; got {nu}. "
                         f"(0.5 = ideal chain, ~0.588 = self-avoiding walk.)")
    n_atom_dist = CFG.dist_dim  # 16
    dev = x.device
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # |i-j|, [N,N]
    # Effective bond length b: use the mean CA-CA neighbor distance so the reference is scaled
    # to this protein rather than an arbitrary constant. CA is atom index 1.
    ca = x[:, 1, :]
    if N > 1:
        b = torch.linalg.norm(ca[1:] - ca[:-1], dim=-1).mean()
        b = torch.clamp(b, min=1e-3)
    else:
        b = torch.tensor(3.8, device=dev)
    d_coil = b * torch.pow(sep + 1e-6, nu)  # [N,N] analytic expected coil distance
    # Lever D+: sequence-specific AFRC distance map (falls back to analytic coil on any failure).
    if getattr(CFG, 'flory_afrc', False):
        d_afrc = afrc_distance_map(one_hot, N, dev)
        if d_afrc is not None:
            d_coil = d_afrc
    # Broadcast the same scalar distance across all 16 atom-atom channels (the coil model is
    # residue-level; per-atom detail is not defined in the unfolded ensemble).
    D = d_coil.unsqueeze(-1).expand(N, N, n_atom_dist).contiguous()  # [N,N,16] RAW distances
    D = apply_distance_kernel(D, gaussian_coef)   # same kernel as folded (Lever B aware)
    mask_index = torch.where(mask == 0)
    D[mask_index[0], :, :] = 0
    D[:, mask_index[0], :] = 0
    Fb = get_bonded_features(D)  # N,32
    D = D.sum(dim=1)  # N,16
    D = F.normalize(D, p=2, dim=0)
    emb = F.normalize(emb, p=2, dim=0)
    Fh = maybe_concat_burial(x, D, Fb, emb, one_hot, mask, emb_delta=emb_delta)
    return Fh


# AFRC distance maps are deterministic per sequence; cache them so we call the polymer model
# once per unique sequence instead of once per mutation/mini-batch pass.
_AFRC_CACHE = {}
_AFRC_AVAILABLE = None  # tri-state: None=untested, True/False after first probe


def _seq_from_one_hot(one_hot, N):
    """Reconstruct the 1-letter amino-acid string from the one-hot block."""
    inv = {v: k for k, v in AA_MAP.items()}
    idx = one_hot.argmax(dim=-1).tolist()
    return "".join(inv.get(int(i), "A") for i in idx[:N])


def afrc_distance_map(one_hot, N, dev):
    """Lever D+ — AFRC ensemble-average inter-residue distance map for this sequence.

    Uses the Analytical Flory Random Coil model (pip install afrc; idptools/afrc), which returns
    a sequence-specific mean distance map with NO simulation and only the sequence as input.
    Returns an [N,N] float tensor of expected unfolded distances, or None if afrc is unavailable
    or errors (caller then falls back to the analytic b*|i-j|^nu coil). Cached per sequence.
    """
    global _AFRC_AVAILABLE
    if _AFRC_AVAILABLE is False:
        return None
    seq = _seq_from_one_hot(one_hot, N)
    cached = _AFRC_CACHE.get(seq)
    if cached is not None:
        return cached.to(dev)
    try:
        from afrc import AnalyticalFRC  # optional dependency
        _AFRC_AVAILABLE = True
        P = AnalyticalFRC(seq)
        dmap = P.get_distance_map()  # numpy [L,L] ensemble-average distances
        dmap = np.asarray(dmap, dtype=np.float32)
        if dmap.shape[0] < N:  # guard: pad/trim to N if AFRC dropped terminal residues
            pad = np.zeros((N, N), dtype=np.float32)
            k = dmap.shape[0]
            pad[:k, :k] = dmap
            dmap = pad
        t = torch.from_numpy(dmap[:N, :N])
        _AFRC_CACHE[seq] = t
        return t.to(dev)
    except ImportError:
        _AFRC_AVAILABLE = False  # never probe again this run
        return None
    except Exception:
        # Any AFRC-internal failure (odd residue, length limits): fall back silently to analytic.
        return None

def apply_distance_kernel(D, gaussian_coef):
    """Turn raw pairwise distances into contact features (Lever B).

    Baseline (CFG.rbf_centers==0): a single Gaussian contact kernel relu(exp(coef*D^2)) —
    byte-for-byte the original behavior.

    Lever B (CFG.rbf_centers==M>0): a bank of M radial basis functions centered on a grid in
    [rbf_min, rbf_max]. Each RBF is relu(exp(coef*(D-center)^2)); the M responses are SUMMED,
    so the output keeps the SAME last-dim width as the input (dimension-preserving) — no
    downstream dim change. It is a richer, soft-histogram encoding of contact distances.
    """
    M = int(getattr(CFG, 'rbf_centers', 0))
    if M <= 0:
        return torch.relu(torch.exp(gaussian_coef * D ** 2))
    lo = float(getattr(CFG, 'rbf_min', 0.0))
    hi = float(getattr(CFG, 'rbf_max', 20.0))
    # Lever B fail-fast: an inverted/degenerate range silently collapses the RBF bank.
    if not hi > lo:
        raise ValueError(f"[Lever B] rbf_max ({hi}) must be > rbf_min ({lo}); "
                         f"got M={M} centers over an empty range.")
    centers = torch.linspace(lo, hi, M, device=D.device, dtype=D.dtype)
    acc = torch.zeros_like(D)
    for c in centers:
        acc = acc + torch.relu(torch.exp(gaussian_coef * (D - c) ** 2))
    return acc


def compute_burial(x, mask, radius):
    """Lever C — per-residue burial: count CB neighbors within `radius` Angstrom.

    Buried residues (many neighbors) contribute differently to stability than surface ones.
    Returns a [N, 1] tensor scaled by 1/N so it is bounded.
    """
    N = x.shape[0]
    cb = x[:, 3, :]  # CB is atom index 3 (N=0, CA=1, C=2, CB=3)
    d = torch.cdist(cb, cb)  # [N, N]
    within = (d < radius).float()
    within = within - torch.eye(N, device=x.device, dtype=within.dtype)  # exclude self
    valid = mask.float().unsqueeze(0)  # [1, N] zero out masked neighbors
    counts = (within * valid).sum(dim=1, keepdim=True)  # [N, 1]
    return counts / max(N, 1)


def maybe_concat_burial(x, D, Fb, emb, one_hot, mask, emb_delta=None):
    """Assemble the node feature vector, optionally inserting the burial scalar BEFORE emb
    and/or the mutation-delta block AFTER emb.

    Baseline layout:            [D(16) | Fb(32) | emb(E) | one_hot(20)]
    Lever C (CFG.use_burial):   [D(16) | Fb(32) | burial(b) | emb(E) | one_hot(20)]
    Lever G (CFG.mutation_delta): [D(16) | Fb(32) | (burial) | emb(E) | delta(md) | one_hot(20)]
    one_hot ALWAYS stays last; when Lever G is on, a width-md (emb_mut - emb_wt) block is placed
    directly before one_hot so the model reads emb / delta / one_hot from-the-right indices.
    """
    parts_pre = [D, Fb]
    if getattr(CFG, 'use_burial', False):
        b = compute_burial(x, mask, float(getattr(CFG, 'burial_radius', 10.0)))
        # Lever C fail-fast: the burial width MUST equal CFG.burial_dim or the model's slice
        # indices (burial_start .. burial_start+burial_dim) will read the wrong columns.
        expected = int(getattr(CFG, 'burial_dim', 0))
        if b.shape[1] != expected:
            raise ValueError(
                f"[Lever C] burial width {b.shape[1]} != CFG.burial_dim {expected}. "
                f"The graph builder and model disagree on the node layout — check "
                f"CFG.burial_dim was set to 1 when --use_burial is on.")
        parts_pre.append(b)
    parts_pre.append(emb)
    # Lever G: mutation-delta block goes after emb, before one_hot.
    if getattr(CFG, 'mutation_delta', False):
        # Fail-fast (matches Levers B/C/D/F): if the lever is on but the caller did not supply the
        # delta, the node vector would be md columns too NARROW and blow up in a downstream GNN
        # matmul with a misleading error. Name the lever now.
        if emb_delta is None:
            raise ValueError(
                "[Lever G] mutation_delta is ON but emb_delta was not passed by the graph-builder "
                "caller. The (emb_mut - emb_wt) block is required; the node vector would be "
                f"{int(getattr(CFG, 'mutation_delta_dim', 0))} columns too narrow otherwise.")
        expected_md = int(getattr(CFG, 'mutation_delta_dim', 0))
        if emb_delta.shape[1] != expected_md:
            raise ValueError(
                f"[Lever G] mutation-delta width {emb_delta.shape[1]} != "
                f"CFG.mutation_delta_dim {expected_md}. Graph builder / model layout mismatch — "
                f"delta block must equal the embedding width.")
        parts_pre.append(emb_delta)
    parts_pre.append(one_hot)
    return torch.cat(parts_pre, dim=1)


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