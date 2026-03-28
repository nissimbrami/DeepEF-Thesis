from model.data_loader import fetch_dataloader,fetch_inference_loader
from model.data_loader import params as data_params
from model.model_cfg import CFG
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
import os
import pandas as pd
import wandb

# Set the default data type to float32
torch.set_default_dtype(CFG.torch_default_dtype)

# Device-aware helpers for mixed precision and cache clearing
def _empty_cache():
    if CFG.device.type == "cuda":
        torch.cuda.empty_cache()
    elif CFG.device.type == "mps":
        torch.mps.empty_cache()

def _autocast():
    """Return autocast context manager for the active device."""
    if CFG.device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=CFG.precision)
    # MPS/CPU: no-op context manager (autocast not well supported)
    return torch.amp.autocast(device_type="cpu", enabled=False)

def _make_scaler():
    """GradScaler only works on CUDA. Return a dummy on other devices."""
    if CFG.device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    # On MPS/CPU: return a scaler that passes through (scale=1, no-op)
    return torch.amp.GradScaler(enabled=False)
# torch.autograd.set_detect_anomaly(True)
# CFG.debug = True
# CFG.clip_grad_norm = True 
# wandb and debug mode are configured in main() after parsing args



def get_noised_proteins(data,device):
    """
    Returns a noised version of the protein data.
    """
    id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy,proT5_cycle1,\
                proT5_cycle2, proT5_cycle3, proT5_cycle4, proT5_cycle5, proT5_cycle6 = data
            
    # wild type and mutant type
    Xjf = crd_backbone.to(device) # wilde type structure folded
    Xju = torch.clone(Xjf).to(device) # wilde type structure unfolded
    Xcd = torch.clone(crd_decoy).to(device) # decoy structure
    Xcy1 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy2 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy3 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy4 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy5 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    Xcy6 = torch.clone(crd_backbone).to(device) # Cycle permutation structure
    
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
        return None,None,None,None,None,None,None,None,None,None,None,None,None
    #emb = torch.cat((esm_embed,seq),dim=2)
    emb = seq_one_hot.to(device)
    emb_decoy = seq_decoy.to(device)
    # get the cycle permutation
    cycle_emb1 = get_one_hot(seq[0][-1] + seq[0][:-1]).to(device)
    cycle_emb2 = get_one_hot(seq[0][1:] + seq[0][0]).to(device)
    cycle_emb3 = get_one_hot(seq[0][-2:] + seq[0][:-2]).to(device)
    cycle_emb4 = get_one_hot(seq[0][-5:] + seq[0][:-5]).to(device)
    cycle_emb5 = get_one_hot(seq[0][2:] + seq[0][:2]).to(device)
    cycle_emb6 = get_one_hot(seq[0][5:] + seq[0][:5]).to(device)
    
    # move proT5_emb to device
    proT5_emb_decoy, proT5_emb, proT5_cycle1, proT5_cycle2, proT5_cycle3, proT5_cycle4, proT5_cycle5, proT5_cycle6  = proT5_emb_decoy.to(device), proT5_emb.to(device), proT5_cycle1.to(device), proT5_cycle2.to(device), proT5_cycle3.to(device),proT5_cycle4.to(device), proT5_cycle5.to(device), proT5_cycle6.to(device)
    
    # squeeze the data
    Xd, Xjf, Xju, Xcd, Xdu, Xcy1, Xcy2, Xcy3, Xcy4, Xcy5, Xcy6 = Xd.squeeze(), Xjf.squeeze(), Xju.squeeze(), Xcd.squeeze(), Xdu.squeeze(), Xcy1.squeeze(), Xcy2.squeeze(), Xcy3.squeeze(), Xcy4.squeeze(), Xcy5.squeeze(), Xcy6.squeeze()
    emb_decoy, emb = emb_decoy.squeeze(), emb.squeeze()
    mask_decoy, mask, mask_crd_decoy= mask_decoy.squeeze(), mask.squeeze(), mask_crd_decoy.squeeze()
    proT5_emb_decoy, proT5_emb = proT5_emb_decoy.squeeze(), proT5_emb.squeeze()
    proT5_cycle1, proT5_cycle2, proT5_cycle3, proT5_cycle4, proT5_cycle5, proT5_cycle6 = proT5_cycle1.squeeze(), proT5_cycle2.squeeze(), proT5_cycle3.squeeze(), proT5_cycle4.squeeze(), proT5_cycle5.squeeze(), proT5_cycle6.squeeze()
    # Extract CA coordinates (atom index 1) before get_graph consumes 3D coords
    ca_native = Xjf[:, 1, :].clone()   # [N, 3]
    ca_decoy = Xcd[:, 1, :].clone()    # [N, 3]
    # get folded graph
    Xjf = get_graph(Xjf, emb, proT5_emb, mask)
    # get unfolded graph
    Xju = get_unfolded_graph(Xju, emb, proT5_emb, mask)
    # get decoy graph
    Xd, Xcd, Xdu = get_graph(Xd, emb_decoy, proT5_emb_decoy, mask_decoy), get_graph(Xcd, emb, proT5_emb, mask_crd_decoy), get_unfolded_graph(Xdu, emb_decoy, proT5_emb_decoy, mask_decoy)
    # Add cycle permutation
    Xcy1 = get_graph(Xcy1, cycle_emb1, proT5_cycle1, mask)
    Xcy2 = get_graph(Xcy2, cycle_emb2, proT5_cycle2, mask)
    Xcy3 = get_graph(Xcy3, cycle_emb3, proT5_cycle3, mask)
    Xcy4 = get_graph(Xcy4, cycle_emb4, proT5_cycle4, mask)
    Xcy5 = get_graph(Xcy5, cycle_emb5, proT5_cycle5, mask)
    Xcy6 = get_graph(Xcy6, cycle_emb6, proT5_cycle6, mask)
    # Xjf.requires_grad = True
    # create a batch of Xjf,Xkf,Xju,Xku,x_decoy
    Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4,Xcy5,Xcy6 = Xjf.unsqueeze(0),Xju.unsqueeze(0),Xd.unsqueeze(0), Xcd.unsqueeze(0), Xdu.unsqueeze(0),Xcy1.unsqueeze(0),Xcy2.unsqueeze(0),Xcy3.unsqueeze(0),Xcy4.unsqueeze(0),Xcy5.unsqueeze(0),Xcy6.unsqueeze(0)

    # Also return native coords + metadata for DSM (needs to noise distance matrix)
    native_info = (crd_backbone.squeeze().to(device), emb, proT5_emb, mask)
    # Build CA coords batch [9, N, 3] matching X = cat(Xjf,Xju,Xd,Xcd,Xdu,Xcy1..4)
    ca_n = ca_native.unsqueeze(0)  # [1, N, 3]
    ca_d = ca_decoy.unsqueeze(0)   # [1, N, 3]
    ca_coords = torch.cat([ca_n, ca_n, ca_n, ca_d, ca_n,
                            ca_n, ca_n, ca_n, ca_n], dim=0)  # [9, N, 3]
    return Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4,Xcy5,Xcy6, native_info, ca_coords
    
# define validation function
def compute_metrics(Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4):
    """Compute ranking and energy gap metrics from energy values."""
    ejf = Ejf.item()
    eju, exd, ecd = Eju.item(), Exd.item(), Ecd.item()
    exdu = Exdu.item()
    ecy1, ecy2, ecy3, ecy4 = Ecy1.item(), Ecy2.item(), Ecy3.item(), Ecy4.item()
    all_energies = [ejf, eju, exd, ecd, exdu, ecy1, ecy2, ecy3, ecy4]
    return {
        # Ranking (1 if correct, 0 if wrong)
        "rank_Ejf_lt_Exd": 1.0 if ejf < exd else 0.0,
        "rank_Ejf_lt_Ecd": 1.0 if ejf < ecd else 0.0,
        "rank_Eju_lt_Ecd": 1.0 if eju < ecd else 0.0,
        "rank_Ejf_lowest": 1.0 if ejf == min(all_energies) else 0.0,
        # Energy gaps (positive = correct ranking)
        "gap_Exd_Ejf": exd - ejf,
        "gap_Ecd_Ejf": ecd - ejf,
        "gap_Ecd_Eju": ecd - eju,
        # Stability
        "energy_mean": sum(all_energies) / len(all_energies),
        "energy_std": float(torch.std(torch.tensor(all_energies)).item()),
    }

def validation(model, dataloader, device,epoch,N,optimizer,val_type = 'robust'):
    """
    Validation function for the model.
    """
    valid_loss = 0
    valid_lossd = 0
    valid_lossg = 0
    valid_lossc = 0
    n_skips = 0
    val_metrics_accum = {}
    model.eval()
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Validation: Epoch {epoch}")
        for index, data in (enumerate(tepoch)):
            # Clean the GPU cache
            if(device.type == "cuda" or device.type == "mps"):
                _empty_cache()
            gc.collect()
            # zero the parameter gradients
            optimizer.zero_grad()
            Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4,Xcy5,Xcy6, native_info, ca_coords = get_noised_proteins(data,device)
            if Xjf is None:
                n_skips += 1
                continue
            X = torch.cat((Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4),dim=0)

            with torch.no_grad():
                # half precision validation
                with _autocast():
                    # calculate the energy for the folded unfolded and decoy structure
                    E = model(X, ca_coords=ca_coords)
                    Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4 = E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7], E[8]
                    # calculate the loss
                    loss ,lossd, lossg,lossc = criterion(Ejf, Eju, Exd, Xjf, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4, with_grad = False)

            # Denoising score matching (replaces gradient penalty)
            if CFG.gradient_penalty:
                # zero the parameter gradients
                optimizer.zero_grad()
                _empty_cache()
                gc.collect()
                with _autocast():
                    lossg, _ = denoising_score_matching(model, Xjf, native_info, sigma=CFG.sigma)

                loss += lossg

            valid_loss += loss.item()
            valid_lossd += lossd.item()
            valid_lossg += lossg.item()
            valid_lossc += lossc.item()

            # Accumulate ranking/gap metrics
            metrics = compute_metrics(Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4)
            for k, v in metrics.items():
                val_metrics_accum.setdefault(k, []).append(v)

            _empty_cache()
            gc.collect()
            # update the progress bar
            if index % 1000 == 999:
                print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}, n_skips: {n_skips}")

    n = len(dataloader) - n_skips if len(dataloader) > n_skips else len(dataloader)
    # Average validation metrics
    val_metrics_avg = {f"val_{k}": sum(v) / len(v) for k, v in val_metrics_accum.items()}
    return valid_loss/n, valid_lossd/n, valid_lossg/n, valid_lossc/n, val_metrics_avg

def train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader,best_val=1000,scheduler=None,scaler=None):
    """
    Training function for the model.
    
    """
    model.train()
    epoch_train_loss = []
    running_loss = 0.0
    n_skips = 0
    ds_length = len(dataloader)
    # Epoch-level accumulators
    epoch_loss_sum, epoch_lossd_sum, epoch_lossg_sum = 0.0, 0.0, 0.0
    epoch_metrics_accum = {}
    epoch_n_samples = 0
    denoising_score_matching._logged_this_epoch = False  # reset per-epoch FD-DSM diagnostic
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Epoch {epoch}")
        for index, data in enumerate(tepoch):
            # Clean the GPU cache
            _empty_cache()
            gc.collect()
             # zero the parameter gradients
            optimizer.zero_grad()
            Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4,Xcy5,Xcy6, native_info, ca_coords = get_noised_proteins(data,device)
            if Xjf is None:
                n_skips += 1
                continue
            X = torch.cat((Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4),dim=0)

            # half precision training
            with _autocast():
                # calculate the energy for the folded unfolded and decoy structure
                E = model(X, ca_coords=ca_coords)
                Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4 = E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7], E[8]
                # calculate the loss
                loss ,lossd, lossg,lossc = criterion(Ejf, Eju, Exd, Xjf, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4, with_grad = False)

            # Denoising score matching — combined with InfoNCE in single backward
            dsm_raw_val = torch.tensor(0.0)
            if CFG.gradient_penalty:
                with _autocast():
                    lossg, dsm_raw_val = denoising_score_matching(model, Xjf, native_info, sigma=CFG.sigma)
                # Clip DSM loss to prevent Hessian explosions (observed spikes to 2000+)
                lossg = torch.clamp(lossg, max=20.0)
                loss = lossd + lossg

            # Single backward pass — both losses contribute to Adam state together
            scaler.scale(loss).backward()

            # Clip gradients to prevent exploding gradients
            if CFG.clip_grad_norm:
                clip_grad_norm(model.parameters(), CFG.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            # Compute ranking/gap metrics
            metrics = compute_metrics(Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4)

            # Accumulate epoch-level stats
            epoch_loss_sum += loss.item()
            epoch_lossd_sum += lossd.item()
            epoch_lossg_sum += lossg.item()
            for k, v in metrics.items():
                epoch_metrics_accum.setdefault(k, []).append(v)
            epoch_n_samples += 1

            # Gradient norm for monitoring stability
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            # print statistics
            running_loss += loss.item()
            if index % 1000 == 999 :    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 1000:.3f}')
                print(f"skipped {n_skips}")
                epoch_train_loss.append(running_loss/1000)
                wandb.log({"epoch": epoch,"running_loss": running_loss/1000,"running_lossIndex":ds_length*epoch+index})
                running_loss = 0.0

            _empty_cache()
            gc.collect()
            # update the progress bar
            tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(running_loss/(index%1000 + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"sequence_len": Xjf.shape[1]})
            # Log metrics
            step = ds_length*epoch+index
            wandb.log({
                    "epoch": epoch, "step": step, "sequence_len": Xjf.shape[1],
                    # Losses
                    "loss": loss.item(), "lossd": lossd.item(), "lossg": lossg.item(),
                    # Energies
                    "Ejf": Ejf.item(), "Eju": Eju.item(), "Exd": Exd.item(),
                    "Ecd": Ecd.item(), "Exdu": Exdu.item(),
                    "Ecy1": Ecy1.item(), "Ecy2": Ecy2.item(),
                    "Ecy3": Ecy3.item(), "Ecy4": Ecy4.item(),
                    # Ranking (per-sample)
                    **{f"train_{k}": v for k, v in metrics.items()},
                    # Stability
                    "grad_norm": grad_norm,
                    "dsm_raw": dsm_raw_val.item(),
                })
            
        print(f"skipped {n_skips}")
        # ---- Epoch summary (train) ----
        if epoch_n_samples > 0:
            avg_loss = epoch_loss_sum / epoch_n_samples
            avg_lossd = epoch_lossd_sum / epoch_n_samples
            avg_lossg = epoch_lossg_sum / epoch_n_samples
            avg_metrics = {k: sum(v)/len(v) for k, v in epoch_metrics_accum.items()}
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch} TRAIN SUMMARY  (emb_projection={CFG.emb_projection})")
            print(f"  loss={avg_loss:.4f}  lossd={avg_lossd:.4f}  lossg={avg_lossg:.4f}")
            print(f"  rank_Ejf<Exd={avg_metrics.get('rank_Ejf_lt_Exd',0):.2%}  rank_Ejf<Ecd={avg_metrics.get('rank_Ejf_lt_Ecd',0):.2%}  rank_Ejf_lowest={avg_metrics.get('rank_Ejf_lowest',0):.2%}")
            print(f"  gap(Exd-Ejf)={avg_metrics.get('gap_Exd_Ejf',0):.4f}  gap(Ecd-Ejf)={avg_metrics.get('gap_Ecd_Ejf',0):.4f}")
            print(f"{'='*60}\n")
        save_checkpoint(epoch, model, optimizer, loss,0,CFG.model_path+str(epoch)+"_final_model.pt")
        # evaluate the model
        val_loss, val_lossd, val_lossg, valid_lossc, val_metrics = validation(model, valid_loader,CFG.device,epoch, CFG.N, optimizer , val_type = 'robust')
         # update wandb metrics
        wandb.log({
                "epoch": epoch,
                "learning rate": optimizer.param_groups[0]["lr"],
                # Validation losses
                "validation loss": val_loss,
                "validation lossd": val_lossd,
                "validation lossg": val_lossg,
                # Validation ranking & gaps (averaged over all val samples)
                **val_metrics,
            })
         # Update the learning rate based on the validation loss
        scheduler.step()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} VALIDATION SUMMARY  (emb_projection={CFG.emb_projection})")
        print(f"  val_loss={val_loss:.4f}  val_lossd={val_lossd:.4f}  val_lossg={val_lossg:.4f}")
        vr = val_metrics
        print(f"  rank_Ejf<Exd={vr.get('val_rank_Ejf_lt_Exd',0):.2%}  rank_Ejf<Ecd={vr.get('val_rank_Ejf_lt_Ecd',0):.2%}  rank_Ejf_lowest={vr.get('val_rank_Ejf_lowest',0):.2%}")
        print(f"  gap(Exd-Ejf)={vr.get('val_gap_Exd_Ejf',0):.4f}  gap(Ecd-Ejf)={vr.get('val_gap_Ecd_Ejf',0):.4f}")
        print(f"{'='*60}\n")
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
    scaler = _make_scaler()
    for epoch in (range(EPOCH,CFG.num_epochs+EPOCH)):  # loop over the dataset multiple times

        
        _empty_cache()
        gc.collect()
        model.train()
        model,epoch_train_loss,valid_loss = train_one_epoch(model, optimizer, dataloader, device,epoch,N,valid_loader,valid_loss, scheduler,scaler)

    print('Finished Training')


def gradient_penalty(X_native, E_native):
    """Implementing the lossg equation:
        The gradient of a wild type structure should be close to zero.
        Therefore we will add it to the loss as lossg"""
    partial_dx_native = torch.autograd.grad(outputs=E_native, inputs=X_native,
                                            grad_outputs=torch.ones_like(E_native),
                                            create_graph=True, retain_graph=True)[0]
    # Use mse loss
    lossg = torch.mean(partial_dx_native**2)
    return lossg

def denoising_score_matching(model, X_native, native_info, sigma=CFG.sigma, K=1, epsilon=0.1):
    """FD-DSM on distance features only (D, 16 dims per residue).

    Uses finite-difference to estimate ∂E/∂x_D instead of autograd with
    create_graph=True. This avoids the vanishing Hessian problem:
      - Autograd DSM: needs ∂²E/(∂x_D ∂θ) — vanishes through norm layers → no learning
      - FD-DSM:       needs only ∂E/∂θ       — first-order, healthy gradients

    FD approximation: v·∇E ≈ (E(x+εv) - E(x-εv)) / 2ε
    DSM target:       v·score = v·(-noise/σ²)
    Loss:             mean_K[ (FD_score - target_v)² ]

    Args:
        model: the energy model
        X_native: clean graph features [1, N, features]
        native_info: tuple (coords, emb, proT5, mask)
        sigma: noise std for distance features
        K: number of random FD directions per sample
        epsilon: finite-difference step size
    Returns:
        lossg: scalar FD-DSM loss
        dsm_raw: detached raw value for logging
    """
    D_DIM = 16
    X_clean = X_native.squeeze(0).detach()  # [N, F]
    N = X_clean.shape[0]
    device = X_clean.device

    # Noise only D features (first 16 dims)
    noise_d = torch.randn(N, D_DIM, device=device) * sigma
    X_noisy = X_clean.clone()
    X_noisy[:, :D_DIM] = X_clean[:, :D_DIM] + noise_d

    # Extract CA coords for distance-based GAT edges
    crd_backbone = native_info[0]  # [N, 4, 3]
    ca_single = crd_backbone[:, 1, :].unsqueeze(0)  # [1, N, 3]

    loss = torch.tensor(0.0, device=device)
    fd_scores = []
    for _ in range(K):
        # Random unit direction in D-space only
        v = torch.zeros_like(X_clean)
        v_d = torch.randn(N, D_DIM, device=device)
        v_d = v_d / (v_d.norm() + 1e-8)
        v[:, :D_DIM] = v_d

        # Single batched forward pass: E+ and E- share the same dropout mask
        model.eval()
        X_pair = torch.stack([X_noisy + epsilon * v, X_noisy - epsilon * v], dim=0)  # [2, N, F]
        E = model(X_pair, ca_coords=ca_single.expand(2, -1, -1))
        model.train()
        fd_score = (E[0] - E[1]) / (2 * epsilon)  # scalar: v·∇E

        # Target: v·score = v·(-noise/σ²), only D dims contribute since v=0 elsewhere
        target_v = -torch.sum(v_d * noise_d) / (sigma ** 2)
        loss = loss + (fd_score - target_v) ** 2
        fd_scores.append(fd_score.detach().abs().item())

    lossg = loss / K

    # Diagnostic: print once per epoch
    if not denoising_score_matching._logged_this_epoch:
        denoising_score_matching._logged_this_epoch = True
        target_mag = (noise_d.abs() / sigma**2).mean().item()
        print(f"  FD-DSM diag: |fd_score|={sum(fd_scores)/len(fd_scores):.4f}  |target|≈{target_mag:.4f}  lossg={lossg.item():.4f}")

    return lossg, lossg.detach()


denoising_score_matching._logged_this_epoch = False

def criterion(Ejf, Eju, Exd, X_native, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4, with_grad = True , reg_alpha = CFG.reg_alpha):
    """
    The loss function for the model corresponds to 2 main losses:
    1. lossg: the partial derivative of the energy with respect to the native structure
    2. lossd: the energy of the native structure divided by the decoy energy
    Args:
        Ejf (tensor): The energy of the folded native structure
        Eju (tensor): The energy of the unfolded native structure
        Exd (tensor): The energy of the decoy sequence
        X_native (tensor): The native structure coordinates (for gradient penalty)
        Ecd (tensor): The energy of the decoy structure
        Exdu (tensor): The energy of the decoy structure unfolded
        Ecy1-Ecy4 (tensor): The energy of cycle permutation structures
    output:
        loss (tensor): The total loss
        lossd (tensor): The ranking loss
        lossg (tensor): The gradient penalty loss
        lossc (tensor): Always zero (removed, kept for API compatibility)
    """
    lossg = gradient_penalty(X_native, Ejf) if with_grad else torch.tensor(0.0).to(Ejf.device)
    lossd = lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju, Ecy1, Ecy2, Ecy3, Ecy4)
    lossc = torch.tensor(0.0).to(Ejf.device)

    return lossd+lossg , lossd, lossg, lossc
  
def lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju, Ecy1, Ecy2, Ecy3, Ecy4, tau=CFG.tau):
    """Boltzmann contrastive loss (InfoNCE) for energy ranking.

    Two contrastive terms:
    1. Ejf should have lowest energy among: Ejf, Exd, Ecd, Eju, Ecy1..4
       (native folded is the most stable state)
    2. Eju should have lower energy than Ecd
       (native unfolded is more stable than a decoy structure)

    L = -log( exp(-E_positive/τ) / Σ exp(-Eᵢ/τ) )
      = E_positive/τ + log(Σ exp(-Eᵢ/τ))

    When the native is already well below decoys, the loss saturates → 0.
    Temperature τ controls how strict the ranking is.
    """
    # Primary: native folded should be lowest energy
    energies_primary = torch.stack([Ejf, Exd, Ecd, Eju, Ecy1, Ecy2, Ecy3, Ecy4])
    log_probs_primary = -energies_primary / tau
    loss_primary = -log_probs_primary[0] + torch.logsumexp(log_probs_primary, dim=0)

    # Secondary: native unfolded should be lower than decoy structure
    energies_secondary = torch.stack([Eju, Ecd])
    log_probs_secondary = -energies_secondary / tau
    loss_secondary = -log_probs_secondary[0] + torch.logsumexp(log_probs_secondary, dim=0)

    return loss_primary + loss_secondary

    
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
    # Set wandb and debug paths
    if CFG.debug:
        CFG.model_path = f"./res/debug-{CFG.emb_projection}/"
        CFG.results_path = f'./res/results-debug-{CFG.emb_projection}/'
        os.makedirs(CFG.model_path, exist_ok=True)
        os.makedirs(CFG.results_path, exist_ok=True)
        print('**** Debug mode ****')
    proj_tag = f"emb-{CFG.emb_projection}" if CFG.emb_projection != "none" else "no-proj"
    if CFG.emb_projection == "low_rank":
        proj_tag += f"-r{CFG.emb_proj_rank}"
    if CFG.emb_projection == "mlp":
        proj_tag += f"-d{CFG.emb_proj_dim}"
    wandb.init(project="DeepEF-InfoNCE-DSM",
               name=f'InfoNCE+DSM light attention GCN ({proj_tag})' if not CFG.debug else f'debug-{CFG.debug_size}prot-{CFG.num_epochs}ep-{proj_tag}')
    print('***Start main function***')
    print('***load the data with dataloader***')
    d_params = data_params(num_workers =CFG.num_workers, batch_size=CFG.batch_size,cuda=CFG.cuda,constraint=CFG.constraint, 
                           debug=CFG.debug,dataset='scn',LLM_EMB=True)
    train_loader, valid_loader,test_loader = fetch_dataloader(data_dir=CFG.data_path, params=d_params)
    # Build the model
    print('***Build the model***')
    model = PEM(layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True,
                emb_projection=CFG.emb_projection,
                gat_cutoff=CFG.gat_cutoff).to(CFG.device)
    model.name = "PEM-With LLM embedding"
    model.energy_epsilon = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    # Define the learning rate scheduler based on loss
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # configurate wandb
    wandb_config(wandb, model, optimizer, scheduler, train_loader,
                CFG.model_path,CFG.reg_alpha,CFG.gaussian_coef,CFG.lr,
                CFG.num_layers,CFG.dropout_rate,CFG.precision)
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
    # Parse --emb_projection flag
    for i, arg in enumerate(sys.argv):
        if arg == '--emb_projection' and i + 1 < len(sys.argv):
            CFG.emb_projection = sys.argv[i + 1]
        if arg == '--emb_proj_rank' and i + 1 < len(sys.argv):
            CFG.emb_proj_rank = int(sys.argv[i + 1])
        if arg == '--emb_proj_dim' and i + 1 < len(sys.argv):
            CFG.emb_proj_dim = int(sys.argv[i + 1])
        if arg == '--emb_proj_hidden' and i + 1 < len(sys.argv):
            CFG.emb_proj_hidden = int(sys.argv[i + 1])
    # Parse --debug flag for quick local testing on MPS/CPU
    if '--debug' in sys.argv:
        CFG.debug = True
        CFG.debug_size = 50
        CFG.num_epochs = 10
        CFG.num_workers = 0
        CFG.seq_len = 350  # limit protein size for MPS memory
        print(f'**** Debug mode: {CFG.debug_size} proteins, {CFG.num_epochs} epochs, seq_len<={CFG.seq_len}, emb_projection={CFG.emb_projection}, device={CFG.device} ****')
    if not CFG.debug:
        CFG.model_path = './res/trianed_models-light_attention_newGCN/'
        CFG.results_path = './res/results-emb/'
    CFG.dropout_rate = 0.3
    CFG.gaussian_coef = -0.08
    CFG.reg_alpha = 0.1
    main()
