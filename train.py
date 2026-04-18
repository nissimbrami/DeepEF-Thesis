from model.data_loader import fetch_dataloader,fetch_inference_loader
from model.data_loader import params as data_params
from model.model_cfg import CFG
from model.hydro_net import PEM
from model.net import params as model_params
from train_utils import *
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import gc
import time
import sys
import os
import csv
import datetime
import pathlib
import pandas as pd
import wandb


# ---------------------------------------------------------------------------
# DDP helpers — all no-ops when running single-GPU (no torchrun / LOCAL_RANK)
# ---------------------------------------------------------------------------
def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _is_dist() else 0

def _world_size():
    return dist.get_world_size() if _is_dist() else 1

def _is_main():
    return _rank() == 0

def setup_ddp():
    """Initialize NCCL process group when launched via torchrun."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return  # single-GPU: nothing to do
    torch.cuda.set_device(local_rank)
    CFG.device = torch.device(f"cuda:{local_rank}")
    CFG.cuda = True
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    if _is_dist():
        dist.destroy_process_group()


def _make_loader(dataset, shuffle=True):
    """Build a DataLoader with DistributedSampler when running DDP."""
    if _is_dist():
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=CFG.seed)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=CFG.num_workers,
        pin_memory=CFG.cuda,
        persistent_workers=CFG.persistent_workers and CFG.num_workers > 0,
        prefetch_factor=CFG.prefetch_factor if CFG.num_workers > 0 else None,
    ), sampler


# ---------------------------------------------------------------------------
# Run logger: writes per-step + per-epoch CSVs locally; wandb gets only the
# curated per-epoch summary so trends are clear.
# ---------------------------------------------------------------------------
class RunLogger:
    def __init__(self, root="logs", run_name=None):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = run_name or ts
        self.dir = pathlib.Path(root) / f"{ts}_{run_name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.dir / "steps.csv"
        self.epochs_path = self.dir / "epochs.csv"
        self._steps_fh = None
        self._steps_writer = None
        self._epochs_fh = None
        self._epochs_writer = None
        print(f"[RunLogger] writing local logs to {self.dir}")

    def _ensure_writer(self, attr_fh, attr_writer, path, fieldnames):
        fh = getattr(self, attr_fh)
        if fh is None:
            fh = open(path, "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            setattr(self, attr_fh, fh)
            setattr(self, attr_writer, writer)
            return writer
        return getattr(self, attr_writer)

    def log_step(self, row: dict):
        writer = self._ensure_writer("_steps_fh", "_steps_writer",
                                     self.steps_path, list(row.keys()))
        writer.writerow(row)
        self._steps_fh.flush()

    def log_epoch(self, row: dict):
        writer = self._ensure_writer("_epochs_fh", "_epochs_writer",
                                     self.epochs_path, list(row.keys()))
        writer.writerow(row)
        self._epochs_fh.flush()

    def close(self):
        for fh in (self._steps_fh, self._epochs_fh):
            if fh is not None:
                fh.close()


# Module-level logger; instantiated in main()
RUN_LOGGER: "RunLogger | None" = None

# Set the default data type to float32
torch.set_default_dtype(CFG.torch_default_dtype)

# Enable TF32 on Ampere/Ada GPUs — free ~1.5x matmul speedup, same API
if CFG.device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Device-aware helpers for mixed precision and cache clearing
def _empty_cache():
    if CFG.device.type == "cuda":
        torch.cuda.empty_cache()
    elif CFG.device.type == "mps":
        torch.mps.empty_cache()

def _autocast():
    """Return autocast context manager for the active device."""
    if CFG.device.type == "cuda":
        try:
            return torch.amp.autocast(device_type="cuda", dtype=CFG.precision)
        except AttributeError:
            return torch.cuda.amp.autocast()
    try:
        return torch.amp.autocast(device_type="cpu", enabled=False)
    except AttributeError:
        import contextlib
        return contextlib.nullcontext()

def _make_scaler():
    """GradScaler only works on CUDA. Return a dummy on other devices."""
    if CFG.device.type == "cuda":
        try:
            return torch.amp.GradScaler("cuda")
        except AttributeError:
            return torch.cuda.amp.GradScaler()
    try:
        return torch.amp.GradScaler(enabled=False)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=False)
# torch.autograd.set_detect_anomaly(True)
# CFG.debug = True
# CFG.clip_grad_norm = True 
# wandb and debug mode are configured in main() after parsing args



def _build_graph_features(D_base, mask, proT5_emb, one_hot, unfolded=False, gaussian_coef=CFG.gaussian_coef):
    """Build [N, F] graph feature tensor from a pre-computed raw distance matrix.

    Output layout: [D_sum(16), Fb(32), proT5_emb_normalized(1024), one_hot(20)] = [N, 1092]
    Matches the layout assumed by PEM: llm_index=-1044, one_hot_index=-20.
    """
    D = torch.relu(torch.exp(gaussian_coef * D_base ** 2))
    mi = torch.where(mask == 0)
    if mi[0].numel() > 0:
        D[mi[0], :, :] = 0
        D[:, mi[0], :] = 0
    if unfolded:
        D = zero_except_udiagonal(D)
    Fb = get_bonded_features(D)
    D_sum = F.normalize(D.sum(dim=1), p=2, dim=0)
    emb_n = F.normalize(proT5_emb, p=2, dim=0)
    return torch.cat([D_sum, Fb, emb_n, one_hot], dim=1)


def get_noised_proteins(data,device):
    """
    Returns 9 graph representations of a protein for InfoNCE + DSM training.
    Computes the expensive O(N²) distance matrix only twice (native + decoy coords)
    instead of once per representation.
    """
    id, crd_backbone, mask, seq_one_hot, seq,ang_backbone, ang,\
                proT5_emb, proT5_mut,seq_mut, crd_decoy, mask_crd_decoy, seq_crd_decoy,proT5_cycle1,\
                proT5_cycle2, proT5_cycle3, proT5_cycle4, proT5_cycle5, proT5_cycle6 = data

    # Load native and decoy coordinates to device
    Xjf = crd_backbone.to(device)    # native folded coords
    Xcd = crd_decoy.to(device)       # decoy structure coords

    # Align decoy length to native length
    if Xcd.shape[1] > Xjf.shape[1]:
        Xcd = Xcd[:,:Xjf.shape[1],:,:]
        mask_crd_decoy = mask_crd_decoy[:,:Xjf.shape[1]]
    elif Xcd.shape[1] < Xjf.shape[1]:
        Xcd = torch.cat((Xcd, torch.zeros(Xjf.shape[0], Xjf.shape[1] - Xcd.shape[1], *Xcd.shape[2:]).to(device)), dim=1)
        mask_crd_decoy = torch.cat((mask_crd_decoy, torch.zeros(mask.shape[0], mask.shape[1] - mask_crd_decoy.shape[1])), dim=1)

    seq_one_hot = seq_one_hot.to(device)

    # create decoy sequence (shuffled native)
    seq_decoy, mask_decoy, proT5_emb_decoy = mix_A_acid(seq_one_hot=seq_one_hot, emb=proT5_emb, mask=mask, val_type='train', device=device)

    if Xjf.shape[1] > CFG.seq_len:  # if the protein is too long, skip it (GPU memory limitation)
        return None,None,None,None,None,None,None,None,None,None,None,None,None

    # Squeeze batch dim (batch_size=1)
    Xjf_sq = Xjf.squeeze()        # [N, 4, 3]
    Xcd_sq = Xcd.squeeze()        # [N, 4, 3]
    emb      = seq_one_hot.squeeze()       # [N, 20]
    emb_decoy = seq_decoy.squeeze()        # [N, 20]
    mask_sq  = mask.squeeze()              # [N]
    mask_decoy = mask_decoy.squeeze()      # [N]
    mask_crd_decoy = mask_crd_decoy.squeeze()  # [N]

    proT5_emb   = proT5_emb.to(device).squeeze()          # [N, 1024]
    proT5_emb_decoy = proT5_emb_decoy.squeeze()           # [N, 1024]
    proT5_cycle1 = proT5_cycle1.to(device).squeeze()
    proT5_cycle2 = proT5_cycle2.to(device).squeeze()
    proT5_cycle3 = proT5_cycle3.to(device).squeeze()
    proT5_cycle4 = proT5_cycle4.to(device).squeeze()
    proT5_cycle5 = proT5_cycle5.to(device).squeeze()
    proT5_cycle6 = proT5_cycle6.to(device).squeeze()

    # Cycle permutation one-hot embeddings
    cycle_emb1 = get_one_hot(seq[0][-1]  + seq[0][:-1]).to(device)
    cycle_emb2 = get_one_hot(seq[0][1:]  + seq[0][0]).to(device)
    cycle_emb3 = get_one_hot(seq[0][-2:] + seq[0][:-2]).to(device)
    cycle_emb4 = get_one_hot(seq[0][-5:] + seq[0][:-5]).to(device)
    cycle_emb5 = get_one_hot(seq[0][2:]  + seq[0][:2]).to(device)
    cycle_emb6 = get_one_hot(seq[0][5:]  + seq[0][:5]).to(device)

    # Extract CA coords before computing distance matrices
    ca_native = Xjf_sq[:, 1, :].clone()  # [N, 3]
    ca_decoy  = Xcd_sq[:, 1, :].clone()  # [N, 3]

    # ── Phase 4a: compute O(N²) distance matrices ONCE per coordinate set ──
    # Native coords are shared by 10/11 representations; decoy coords only by Xcd.
    D_native_raw = get_dist_matrix(Xjf_sq)  # [N, N, 16]
    D_decoy_raw  = get_dist_matrix(Xcd_sq)  # [N, N, 16]

    # Build all 11 graph representations from the two pre-computed matrices.
    # Arg order: (D_base, mask, proT5_emb, one_hot) — proT5 is normalized into features,
    # one_hot is appended raw. Matches PEM's llm_index=-1044, one_hot_index=-20.
    Xjf  = _build_graph_features(D_native_raw, mask_sq,        proT5_emb,       emb,        unfolded=False)
    Xju  = _build_graph_features(D_native_raw, mask_sq,        proT5_emb,       emb,        unfolded=True)
    Xd   = _build_graph_features(D_native_raw, mask_decoy,     proT5_emb_decoy, emb_decoy,  unfolded=False)
    Xcd  = _build_graph_features(D_decoy_raw,  mask_crd_decoy, proT5_emb,       emb,        unfolded=False)
    Xdu  = _build_graph_features(D_native_raw, mask_decoy,     proT5_emb_decoy, emb_decoy,  unfolded=True)
    Xcy1 = _build_graph_features(D_native_raw, mask_sq,        proT5_cycle1,    cycle_emb1, unfolded=False)
    Xcy2 = _build_graph_features(D_native_raw, mask_sq,        proT5_cycle2,    cycle_emb2, unfolded=False)
    Xcy3 = _build_graph_features(D_native_raw, mask_sq,        proT5_cycle3,    cycle_emb3, unfolded=False)
    Xcy4 = _build_graph_features(D_native_raw, mask_sq,        proT5_cycle4,    cycle_emb4, unfolded=False)
    Xcy5 = _build_graph_features(D_native_raw, mask_sq,        proT5_cycle5,    cycle_emb5, unfolded=False)
    Xcy6 = _build_graph_features(D_native_raw, mask_sq,        proT5_cycle6,    cycle_emb6, unfolded=False)

    # Add batch dimension
    Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4,Xcy5,Xcy6 = (
        Xjf.unsqueeze(0), Xju.unsqueeze(0), Xd.unsqueeze(0), Xcd.unsqueeze(0),
        Xdu.unsqueeze(0), Xcy1.unsqueeze(0), Xcy2.unsqueeze(0), Xcy3.unsqueeze(0),
        Xcy4.unsqueeze(0), Xcy5.unsqueeze(0), Xcy6.unsqueeze(0)
    )

    # Native coords + metadata for DSM score matching loss
    native_info = (crd_backbone.squeeze().to(device), emb, proT5_emb, mask_sq)
    # CA coords batch [9, N, 3] matching X = cat(Xjf,Xju,Xd,Xcd,Xdu,Xcy1..4)
    ca_n = ca_native.unsqueeze(0)
    ca_d = ca_decoy.unsqueeze(0)
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
    metric_accum = {
        "rank_Ejf_lt_Exd": 0.0, "rank_Ejf_lt_Ecd": 0.0,
        "rank_Eju_lt_Ecd": 0.0, "rank_Ejf_lowest": 0.0,
        "gap_Exd_Ejf": 0.0, "gap_Ecd_Ejf": 0.0, "gap_Ecd_Eju": 0.0,
    }
    energy_accum = {"Ejf": 0.0, "Eju": 0.0, "Exd": 0.0, "Ecd": 0.0,
                    "Exdu": 0.0, "Ecy1": 0.0, "Ecy2": 0.0, "Ecy3": 0.0, "Ecy4": 0.0}
    n_metric = 0
    model.eval()
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Validation: Epoch {epoch}")
        for index, data in (enumerate(tepoch)):
            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)
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
                with torch.no_grad(), _autocast():
                    lossg, _, _fd = denoising_score_matching(model, Xjf, native_info, sigma=CFG.sigma)
                loss += lossg

            valid_loss += loss.item()
            valid_lossd += lossd.item()
            valid_lossg += lossg.item()
            valid_lossc += lossc.item()

            # Per-sample EBM ranking + energies
            m = compute_metrics(Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4)
            for k in metric_accum:
                metric_accum[k] += m[k]
            for k, v in zip(energy_accum.keys(),
                            [Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4]):
                energy_accum[k] += v.item()
            n_metric += 1

            # update the progress bar
            if index % 1000 == 999:
                print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}, n_skips: {n_skips}")

    n = len(dataloader) - n_skips if len(dataloader) > n_skips else len(dataloader)
    val_metrics = {f"val_{k}": v / max(n_metric, 1) for k, v in metric_accum.items()}
    val_metrics.update({f"val_mean_{k}": v / max(n_metric, 1) for k, v in energy_accum.items()})
    return valid_loss/n, valid_lossd/n, valid_lossg/n, valid_lossc/n, val_metrics

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
    epoch_loss_sum, epoch_lossd_sum, epoch_lossg_sum, epoch_dsm_alpha_sum, epoch_fd_score_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    epoch_energy_accum = {"Ejf": [], "Eju": [], "Exd": [], "Ecd": [], "Exdu": [], "Ecy1": [], "Ecy2": [], "Ecy3": [], "Ecy4": []}
    epoch_n_samples = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Epoch {epoch}")
        for index, data in enumerate(tepoch):
             # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)
            Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4,Xcy5,Xcy6, native_info, ca_coords = get_noised_proteins(data,device)
            if Xjf is None:
                n_skips += 1
                continue
            X = torch.cat((Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2,Xcy3,Xcy4),dim=0)

            # ── Shape & feature-layout sanity check (first step only) ──
            if index == 0 and epoch == 0:
                print(f"\n[DEBUG shapes] Xjf={tuple(Xjf.shape)}, X={tuple(X.shape)}, ca_coords={tuple(ca_coords.shape)}")
                print(f"[DEBUG shapes] native_info: crd={tuple(native_info[0].shape)}, emb={tuple(native_info[1].shape)}, proT5={tuple(native_info[2].shape)}, mask={tuple(native_info[3].shape) if hasattr(native_info[3], 'shape') else native_info[3]}")
                last20 = Xjf[0, 0, -20:].tolist()
                pt_sample = Xjf[0, 0, -1044:-1040].tolist()
                is_binary = all(v in (0.0, 1.0) for v in last20)
                print(f"[DEBUG layout] last-20 one-hot binary={is_binary}: {last20}")
                print(f"[DEBUG layout] proT5 sample (floats expected): {pt_sample}")
                assert is_binary, "FAIL: last-20 dims should be one-hot (0/1)"
                assert X.shape[-1] == 1092, f"FAIL: expected feature dim 1092, got {X.shape[-1]}"
                print("[DEBUG] Shape and layout checks PASSED")

            # half precision training
            with _autocast():
                # calculate the energy for the folded unfolded and decoy structure
                E = model(X, ca_coords=ca_coords)
                if index == 0 and epoch == 0:
                    print(f"[DEBUG shapes] E={tuple(E.shape)} (expected [9])")
                    assert E.shape == (9,), f"FAIL: E shape {E.shape}, expected (9,)"
                Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4 = E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7], E[8]
                # calculate the loss
                loss ,lossd, lossg,lossc = criterion(Ejf, Eju, Exd, Xjf, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4, with_grad = False)

            # Denoising score matching with gradient-norm scaling
            dsm_raw_val = torch.tensor(0.0)
            dsm_alpha = torch.tensor(1.0)
            grad_norm_d = torch.tensor(0.0)
            grad_norm_g = torch.tensor(0.0)
            fd_score_val = 0.0
            if CFG.gradient_penalty:
                model.eval()
                with _autocast():
                    lossg, dsm_raw_val, fd_score_val = denoising_score_matching(model, Xjf, native_info, sigma=CFG.sigma)
                model.train()
                lossg = torch.clamp(lossg, max=20.0)

                # Pass 1: backward on lossd — accumulates ∂lossd/∂θ
                scaler.scale(lossd).backward(retain_graph=True)
                grad_norm_d = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5
                # Snapshot lossd grads before zeroing
                grads_d = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                optimizer.zero_grad(set_to_none=True)

                # Pass 2: backward on lossg — accumulates ∂lossg/∂θ
                scaler.scale(lossg).backward()
                grad_norm_g = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5

                # Alpha scales DSM so its gradient norm matches InfoNCE.
                # Clamp to [0.01, 10] — prevents explosion when grad_norm_g ≈ 0
                dsm_alpha = torch.clamp(grad_norm_d / (grad_norm_g + 1e-8), min=0.01, max=10.0).detach()

                # Manually combine: final grad = grad_d + alpha * grad_g (no third backward)
                for n, p in model.named_parameters():
                    if p.grad is not None and n in grads_d:
                        p.grad.mul_(dsm_alpha).add_(grads_d[n])
                    elif n in grads_d:
                        p.grad = grads_d[n]
                loss = lossd + dsm_alpha * lossg  # for logging only
            else:
                # No DSM: single backward on InfoNCE loss
                scaler.scale(loss).backward()

            # Unscale first so clip threshold applies to true (not scaled) gradients
            scaler.unscale_(optimizer)
            if CFG.clip_grad_norm:
                clip_grad_norm(model.parameters(), CFG.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            # Accumulate epoch-level stats
            epoch_loss_sum += loss.item()
            epoch_lossd_sum += lossd.item()
            epoch_lossg_sum += lossg.item()
            epoch_dsm_alpha_sum += dsm_alpha.item()
            epoch_fd_score_sum += fd_score_val
            for key, val in zip(["Ejf","Eju","Exd","Ecd","Exdu","Ecy1","Ecy2","Ecy3","Ecy4"],
                                 [Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2, Ecy3, Ecy4]):
                epoch_energy_accum[key].append(val.item())
            epoch_n_samples += 1

            # Gradient norm for monitoring stability
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            # print statistics
            running_loss += loss.item()

            # update the progress bar
            tepoch.set_postfix({"loss":round(loss.item(),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"dsm_α":round(dsm_alpha.item(),3),"seq_len": Xjf.shape[1]})

            # Local per-step CSV (no wandb spam) — rank 0 only
            if _is_main() and RUN_LOGGER is not None:
                step = ds_length*epoch+index
                RUN_LOGGER.log_step({
                    "epoch": epoch, "step": step, "sequence_len": int(Xjf.shape[1]),
                    "loss": loss.item(), "lossd": lossd.item(), "lossg": lossg.item(),
                    "dsm_alpha": dsm_alpha.item(), "fd_score": fd_score_val,
                    "grad_norm": grad_norm,
                    "grad_norm_d": float(grad_norm_d) if torch.is_tensor(grad_norm_d) else grad_norm_d,
                    "grad_norm_g": float(grad_norm_g) if torch.is_tensor(grad_norm_g) else grad_norm_g,
                })
            
        print(f"skipped {n_skips}")
        # ---- Epoch summary (train) ----
        if epoch_n_samples > 0:
            avg_loss = epoch_loss_sum / epoch_n_samples
            avg_lossd = epoch_lossd_sum / epoch_n_samples
            avg_lossg = epoch_lossg_sum / epoch_n_samples
            avg_dsm_alpha = epoch_dsm_alpha_sum / epoch_n_samples
            avg_fd_score = epoch_fd_score_sum / epoch_n_samples
            mean_energies = {f"train_mean_{k}": sum(v)/len(v) for k, v in epoch_energy_accum.items() if v}
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch} TRAIN SUMMARY  (emb_projection={CFG.emb_projection})")
            print(f"  loss={avg_loss:.4f}  lossd={avg_lossd:.4f}  lossg={avg_lossg:.4f}  dsm_alpha={avg_dsm_alpha:.4f}  fd_score={avg_fd_score:.4f}")
            print(f"{'='*60}\n")
        save_checkpoint(epoch, model, optimizer, loss, 0, CFG.model_path+str(epoch)+"_final_model.pt", rank=_rank())
        # evaluate the model
        val_loss, val_lossd, val_lossg, valid_lossc, val_metrics = validation(model, valid_loader,CFG.device,epoch, CFG.N, optimizer , val_type = 'robust')

        # Curated per-epoch summary — single wandb.log call per epoch keeps trends clean
        epoch_summary = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            # Train averages
            "train_loss": avg_loss,
            "train_lossd": avg_lossd,
            "train_lossg": avg_lossg,
            "train_dsm_alpha": avg_dsm_alpha,
            "train_fd_score": avg_fd_score,
            **mean_energies,
            # Validation losses
            "val_loss": val_loss,
            "val_lossd": val_lossd,
            "val_lossg": val_lossg,
            # Validation EBM ranking + energies
            **val_metrics,
        }
        if _is_main():
            wandb.log(epoch_summary)
            if RUN_LOGGER is not None:
                RUN_LOGGER.log_epoch(epoch_summary)
        # Update the learning rate based on the validation loss
        scheduler.step()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} VALIDATION SUMMARY  (emb_projection={CFG.emb_projection})")
        print(f"  val_loss={val_loss:.4f}  val_lossd={val_lossd:.4f}  val_lossg={val_lossg:.4f}")
        print(f"{'='*60}\n")
        if val_loss<best_val:
            print('saving model with valid loss: ',val_loss)
            save_checkpoint(epoch, model, optimizer, loss, val_loss, CFG.model_path+"best_model.pt", rank=_rank())
            best_val = val_loss
       
        
                
    return model, epoch_train_loss,val_loss

# define one epoch train
def training(model, optimizer, dataloader, valid_loader, device, N, EPOCH, valid_loss, scheduler, train_sampler=None):
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
        train_sampler: DistributedSampler for DDP (None in single-GPU mode)
    """
    # setup half precision training
    scaler = _make_scaler()
    for epoch in (range(EPOCH,CFG.num_epochs+EPOCH)):  # loop over the dataset multiple times
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # ensures different shuffling per epoch on each rank

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

        # Single batched forward pass: E+ and E- share the same dropout mask.
        # Caller is responsible for model.eval() — don't toggle state here.
        X_pair = torch.stack([X_noisy + epsilon * v, X_noisy - epsilon * v], dim=0)  # [2, N, F]
        E = model(X_pair, ca_coords=ca_single.expand(2, -1, -1))
        fd_score = (E[0] - E[1]) / (2 * epsilon)  # scalar: v·∇E

        # Target: v·score = v·(-noise/σ²), only D dims contribute since v=0 elsewhere
        target_v = -torch.sum(v_d * noise_d) / (sigma ** 2)
        loss = loss + (fd_score - target_v) ** 2
        fd_scores.append(fd_score.detach().abs().item())

    lossg = loss / K
    avg_fd_score = sum(fd_scores) / len(fd_scores)

    return lossg, lossg.detach(), avg_fd_score


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

    
def trainAndTest(model, train_loader, valid_loader, test_loader, optimizer, device, N, epoch, scheduler, train_sampler=None):
    "train and test the model"
    valid_loss = 100
    if epoch > 0:
        model,optimizer,epoch,loss,valid_loss = load_checkpoint(CFG.model_path+f"{epoch-1}_final_model.pt", model, optimizer)
        epoch += 1
    training(model, optimizer, train_loader, valid_loader, CFG.device, CFG.N, epoch, valid_loss, scheduler, train_sampler)
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
    setup_ddp()  # no-op in single-GPU; sets CFG.device per rank in DDP

    # Set wandb and debug paths
    if CFG.debug:
        CFG.model_path = f"./res/debug-{CFG.emb_projection}/"
        CFG.results_path = f'./res/results-debug-{CFG.emb_projection}/'
        if _is_main():
            os.makedirs(CFG.model_path, exist_ok=True)
            os.makedirs(CFG.results_path, exist_ok=True)
        print('**** Debug mode ****')
    proj_tag = f"emb-{CFG.emb_projection}" if CFG.emb_projection != "none" else "no-proj"
    if CFG.emb_projection == "low_rank":
        proj_tag += f"-r{CFG.emb_proj_rank}"
    if CFG.emb_projection == "mlp":
        proj_tag += f"-d{CFG.emb_proj_dim}"
    run_name = f'InfoNCE+DSM light attention GCN ({proj_tag})' if not CFG.debug else f'debug-{CFG.debug_size}prot-{CFG.num_epochs}ep-{proj_tag}'

    if _is_main():
        wandb.init(project="DeepEF-InfoNCE-DSM", name=run_name)
        global RUN_LOGGER
        RUN_LOGGER = RunLogger(root="logs", run_name=run_name.replace(" ", "_").replace("(", "").replace(")", ""))

    print('***Start main function***')
    print('***load the data with dataloader***')
    # Build datasets (all ranks share the same cache; init workers=0 to avoid fork issues before DDP)
    d_params = data_params(num_workers=0, batch_size=CFG.batch_size, cuda=False, constraint=CFG.constraint,
                           debug=CFG.debug, dataset='scn', LLM_EMB=True,
                           persistent_workers=False, prefetch_factor=None)
    raw_train, raw_valid, raw_test = fetch_dataloader(data_dir=CFG.data_path, params=d_params)
    train_loader, train_sampler = _make_loader(raw_train.dataset, shuffle=True)
    valid_loader, _             = _make_loader(raw_valid.dataset, shuffle=False)
    test_loader,  _             = _make_loader(raw_test.dataset,  shuffle=False)

    # Build the model
    print('***Build the model***')
    model = PEM(layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True,
                emb_projection=CFG.emb_projection,
                gat_cutoff=CFG.gat_cutoff).to(CFG.device)
    model.name = "PEM-With LLM embedding"
    model.energy_epsilon = 1e-6
    if CFG.compile_model and hasattr(torch, "compile") and CFG.device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
            print("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    # Wrap with DDP after compile (compile-then-DDP is the recommended order)
    if _is_dist():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[_rank()], output_device=_rank()
        )
        if _is_main():
            print(f"DDP enabled: {_world_size()} GPUs")

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    # Define the learning rate scheduler based on loss
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # configurate wandb (rank-0 only)
    if _is_main():
        wandb_config(wandb, model, optimizer, scheduler, train_loader,
                    CFG.model_path,CFG.reg_alpha,CFG.gaussian_coef,CFG.lr,
                    CFG.num_layers,CFG.dropout_rate,CFG.precision)
    # Run training
    print('***Start training***')
    epoch = 0
    trainAndTest(model, train_loader, valid_loader, test_loader, optimizer, CFG.device, CFG.N, epoch, scheduler, train_sampler)
    cleanup_ddp()
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
        CFG.num_epochs = 20
        CFG.num_workers = 0
        CFG.seq_len = 350  # limit protein size for MPS memory
        CFG.compile_model = False  # skip compile overhead in debug
        print(f'**** Debug mode: {CFG.debug_size} proteins, {CFG.num_epochs} epochs, seq_len<={CFG.seq_len}, emb_projection={CFG.emb_projection}, device={CFG.device} ****')
    if not CFG.debug:
        CFG.model_path = './res/trianed_models-light_attention_newGCN/'
        CFG.results_path = './res/results-emb/'
    CFG.dropout_rate = 0.3
    CFG.gaussian_coef = -0.08
    CFG.reg_alpha = 0.1
    main()
