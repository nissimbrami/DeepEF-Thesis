# fine tuning the existing model with mega-scale data
# Path: Megascale-fineTuning/train.py

import os
import sys
sys.path.append('./')
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import get_graph, get_unfolded_graph, load_checkpoint
import wandb
from tqdm import tqdm
from sklearn.model_selection import KFold
import gc
import random
import pandas as pd

# Constants
COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'prott5_embeddings'
VAL_RATIO = 0.2
RANDOM_SEED = 42
NANO_TO_ANGSTROM = 0.1
DEBUG  = False
EPOCHS = 50 if not DEBUG else 1
FREEZE_LAYERS = True
CRITERION = "L1"
MODEL_PATH = './Megascale-fineTuning/models'
MINI_BATCH_SIZE = 64
DEVICE = 'cuda'# if torch.cuda.is_available() else 'cpu'
# TRAINED_MODEL_PATH = "./res/trianed_models-cycle_per_norm_SM/13_final_model.pt"
TRAINED_MODEL_PATH = "./res/trianed_models-light_attention/20_final_model.pt"
# TRAINED_MODEL_PATH = "./res/trianed_models-2cycle_drop/25_final_model.pt"
BASE_MODEL_NAME = TRAINED_MODEL_PATH.split('/')[-2]
MODEL_NAME = 'PEM_fine_tuned-'+BASE_MODEL_NAME if FREEZE_LAYERS else 'PEM_full_trained-'+BASE_MODEL_NAME
MODEL_NAME += 'kf' # pnas data clearning
# WS1: fine-tuning objective ablation (dg = current behavior, ddg = direct mutation-aware
# ddG = dG_mut - dG_wt, joint = dg + w*ddg). parse_known_args keeps SLURM/other args intact.
import argparse as _argparse
_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument('--loss_mode', default='dg', choices=['dg', 'ddg', 'joint', 'ddg_head'])
_p.add_argument('--seed', type=int, default=42, help='global RNG seed for python/numpy/torch and KFold/train_test_split random_state (reproducibility / seed-ensemble)')
_p.add_argument('--ddg_weight', type=float, default=1.0)
_p.add_argument('--epochs', type=int, default=None, help='override EPOCHS (freeze-phase) for a cheap pilot')
_p.add_argument('--max_folds', type=int, default=None, help='run only the first N folds (pilot)')
_p.add_argument('--run_tag', default='', help='suffix appended to MODEL_NAME to isolate runs')
_p.add_argument('--full_data', action='store_true', help='train on ALL train proteins (no kfold split); 28 test proteins stay held out')
_p.add_argument('--no_pretrain', action='store_true', help='ablation: random init instead of loading the zero-shot core (tests whether zero-shot pretraining helps)')
_p.add_argument('--fold', type=int, default=None, help='run ONLY this kfold index (enables parallel 5-fold CV across GPUs)')
_p.add_argument('--unfreeze_from', default=None, help='Phase-2: load this freeze-phase checkpoint, unfreeze ALL layers, train whole net at low LR')
_p.add_argument('--unfreeze_lr', type=float, default=1e-5, help='LR for the unfreeze/full-training phase')
_p.add_argument('--unfreeze_epochs', type=int, default=6, help='epochs for the unfreeze phase')
_p.add_argument('--no_freeze', action='store_true', help='train the WHOLE network from the start (no freeze phase) — correct baseline for random init')
_p.add_argument('--readout', default='sum', choices=['sum', 'attention', 'gated'], help='Experiment 1: per-residue energy aggregation')
# Cross-protein calibration levers (each independent; all OFF reproduces the baseline).
_p.add_argument('--val_frac', type=float, default=0.1, help='lever 0: carve a held-out val split from TRAIN proteins for epoch selection (FULL_DATA only); 0 disables (falls back to old behavior)')
_p.add_argument('--pooled_corr_weight', type=float, default=0.0, help='lever 1: weight of the pooled cross-protein Pearson loss (0 = off; enables window accumulation)')
_p.add_argument('--pooled_window', type=int, default=8, help='lever 1: # proteins accumulated before an optimizer step + pooled-correlation loss')
_p.add_argument('--pooled_cap', type=int, default=12, help='lever 1: max variants per protein in the pooled buffer (bounds memory; cross-protein corr wants many proteins, few variants each)')
_p.add_argument('--dg_length_norm', default='none', choices=['none', 'n', 'sqrtn'], help='lever 2: divide predicted dG by valid-residue count (n) or its sqrt (sqrtn) to remove the length-driven cross-protein offset')
_p.add_argument('--affine_calib', action='store_true', help='lever 3: after training, fit a global affine (a,b) on TRAIN predictions and write a sidecar for evaluate.py (fixes RMSE; does not change PCC)')
_p.add_argument('--designed_weight', type=float, default=1.0, help='EXP-19: oversample DESIGNED-fold minis (HHH/HEEH/EEHEE/EHEE/TrROS/v2_) by this factor via WeightedRandomSampler (1.0=off=uniform). Tests whether more gradient on designed folds fixes the EXP-14 slope-collapse.')
_p.add_argument('--wt_anchor_weight', type=float, default=0.0, help='WS-1 Arm A: weight of the WT/absolute-dG anchor loss L1(pred_dG(WT), exp_dG(WT)). In ddg mode the absolute scale is free (only differences are supervised) so pred WT abs-dG is at chance (EXP-20/22); this term pins the per-protein baseline = attacks the calibration offset. 0 = off.')
_p.add_argument('--slope_weight', type=float, default=0.0, help='Agent-E SLOPE term: weight of abs(std(pred_ddg) - std(true_ddg)) computed WITHIN the current protein/minibatch (per-protein ddg = output - wt_dg vs delta_g - delta_g_wt; WT = row 0). Penalises the model under-reacting/compressing the spread of ddG within a protein (per-protein slope a_p). 0 = off = bit-identical to baseline.')
_p.add_argument('--flory_unfolded', action='store_true', help="Lever D (coil): replace the tridiagonal-mask unfolded reference with an analytic Flory random-coil, d(i,j)=b*|i-j|^nu, b = protein mean CA-CA bond length. Value-only, shape-identical, no new parameters. Default OFF reproduces the tridiagonal baseline bit-for-bit.")
_p.add_argument('--flory_nu', type=float, default=0.5, help='Lever D coil scaling exponent, must be in (0,1]. 0.5 = ideal chain; ~0.588 = self-avoiding walk. Only read when --flory_unfolded is set.')
_a, _ = _p.parse_known_args()
READOUT = _a.readout
WT_ANCHOR_WEIGHT = _a.wt_anchor_weight
SLOPE_WEIGHT = _a.slope_weight
VAL_FRAC = _a.val_frac
POOLED_CORR_WEIGHT = _a.pooled_corr_weight
POOLED_WINDOW = _a.pooled_window
POOLED_CAP = _a.pooled_cap
DG_LENGTH_NORM = _a.dg_length_norm
AFFINE_CALIB = _a.affine_calib
DESIGNED_WEIGHT = _a.designed_weight
# Lever D (coil): train_utils.get_unfolded_graph reads these off CFG at call time, so setting
# them here is what makes the flag reachable from the command line.
CFG.flory_unfolded = _a.flory_unfolded
CFG.flory_nu = _a.flory_nu
if _a.flory_unfolded and not (0.0 < _a.flory_nu <= 1.0):
    raise ValueError('--flory_nu must be in (0, 1]; got %s' % _a.flory_nu)

# --seed: single knob for full reproducibility + seed-ensemble. Overrides the hardcoded
# RANDOM_SEED so KFold/train_test_split random_state below all follow --seed too.
RANDOM_SEED = _a.seed
def set_seed(seed):
    """Seed python, numpy and torch (CPU+all CUDA devices) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(RANDOM_SEED)

import re as _re
def is_designed(name):
    """EXP-19: Rocklin/Tsuboyama de-novo DESIGNED mini-proteins by name (HHH/HEEH/EEHEE/EHEE topologies,
    trRosetta-hallucinated *_TrROS_*, and v2_* NMR-validated designs)."""
    return bool(_re.match(r'^(HHH|HEEH|EEHEE|EHEE|EHHE|HHHH)', name) or '_TrROS_' in name or name.startswith('v2'))
FULL_DATA = _a.full_data
NO_PRETRAIN = _a.no_pretrain
NO_FREEZE = _a.no_freeze
ONLY_FOLD = _a.fold
UNFREEZE_FROM = _a.unfreeze_from
UNFREEZE_LR = _a.unfreeze_lr
UNFREEZE_EPOCHS = _a.unfreeze_epochs
LOSS_MODE = _a.loss_mode
DDG_WEIGHT = _a.ddg_weight
PILOT_EPOCHS = _a.epochs
if NO_FREEZE:
    FREEZE_LAYERS = False  # train the whole network from the start (proper random-init baseline)
MAX_FOLDS = _a.max_folds
if _a.run_tag:
    MODEL_NAME += f'_{_a.run_tag}'   # keep ablation/pilot runs in separate dirs
PRETRAINED = not NO_PRETRAIN  # --no_pretrain => random init (zero-shot ablation)
TM_PATH = "./data/ThermoMPNN/mega_test.csv"
LR = 1e-4
DROP_OUT = 0.2
REG_LAMBDA = 0
E_REG_LAMBDA = 0.001
UNSTABLE_MUT = True
LIGHT_ATTENTION = True

# config wandb
config = {
    'coords': COORDS,
    'delta_g': DELTA_G,
    'masks': MASKS,
    'one_hot': ONE_HOT,
    'prott5_embeddings': PROTT5_EMBEDDINGS,
    'val_ratio': VAL_RATIO,
    'random_seed': RANDOM_SEED,
    'nano_to_angstrom': NANO_TO_ANGSTROM,
    'debug': DEBUG,
    'epochs': EPOCHS,
    'freeze_layers': FREEZE_LAYERS,
    'model_path': MODEL_PATH,
    'model_name': MODEL_NAME,
    'mini_batch_size': MINI_BATCH_SIZE,
    'device': DEVICE,
    'trained_model_path': TRAINED_MODEL_PATH,
    'pretrained': PRETRAINED,
    'lr': LR,
    'dropout': DROP_OUT,
    'reg_lambda': REG_LAMBDA,
    'e_reg_lambda': E_REG_LAMBDA,
    'unstable_mut': UNSTABLE_MUT,
    'light_attention': LIGHT_ATTENTION,
    'loss_mode': LOSS_MODE,
    'wt_anchor_weight': WT_ANCHOR_WEIGHT,
    'flory_unfolded': _a.flory_unfolded,
    'flory_nu': _a.flory_nu,
    'designed_weight': DESIGNED_WEIGHT,
    'pooled_corr_weight': POOLED_CORR_WEIGHT,
    'val_frac': VAL_FRAC,
    'affine_calib': AFFINE_CALIB
}

if not os.path.exists(os.path.join(MODEL_PATH, MODEL_NAME)):
    os.makedirs(os.path.join(MODEL_PATH, MODEL_NAME))


def wandb_log(log_dict,run = None):
    if not DEBUG:
        if run is not None:
            run.log(log_dict)
        else:
            wandb.log(log_dict)

def normalize_batch(batch, LLM_EMB = True):
    batch['one_hot'] = batch['one_hot'][:, :, :, :-1]
    batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM
    if not LLM_EMB: # zero prot5 embedding
         batch['prott5'] = torch.zeros_like(batch['prott5'])
    return batch

class DDGHead(nn.Module):
    """Direct-ddG readout (loss_mode='ddg_head'): replaces the energy-difference
    ddG = sum e_i(mut) - sum e_i(WT) with a LEARNED mutation-aware head over the
    per-residue latent features h (PEM f_type='features', 128-d). Per residue it sees
    [h_wt, h_mut, h_mut - h_wt] and emits a scalar ddG contribution; summed over residues
    (extensive, matching the energy form). Tests (a) whether dropping the energy-difference
    constraint helps ddG accuracy and (b) whether the zero-shot core helps under this head.
    NOTE: this model is NO LONGER an energy function (no absolute dG / native-vs-decoy)."""
    def __init__(self, feat_dim=128, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim * 3, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, h_wt, h_mut):
        # h_wt: [1,N,F]  h_mut: [n,N,F]
        h_wt = h_wt.expand_as(h_mut)
        feat = torch.cat([h_wt, h_mut, h_mut - h_wt], dim=-1)  # [n,N,3F]
        per_res = self.net(feat).squeeze(-1)                    # [n,N]
        return per_res.sum(dim=1)                               # [n]  ddG per variant


def pearson_loss(pred, true):
    """Lever 1: differentiable (1 - Pearson) over a buffer pooling ddG across proteins.
    Optimizing this enforces cross-protein scale consistency (attacks overall PCC), unlike the
    within-protein MAE which is blind to it."""
    pred = pred - pred.mean()
    true = true - true.mean()
    denom = (pred.norm() * true.norm()).clamp(min=1e-8)
    return 1.0 - (pred * true).sum() / denom

class AllProteinValidationDataset(Dataset):

    def __init__(self, tensor_root_dir, mutations_root_dir, train  = True, one_mut = True ):
        self.tensor_root_dir = tensor_root_dir
        self.train = train
        self.mutations_root_dir = mutations_root_dir
        self.protein_dirs = [protein for i, protein in enumerate(os.listdir(self.tensor_root_dir))]
        self.one_mut = one_mut # remove the mutations with more than one mutation
        self.unstable_mut = UNSTABLE_MUT
        # remove TM proteins 
        tm_proteins = pd.read_csv(TM_PATH)
        tm_proteins = tm_proteins['name'].apply(lambda x: x.split(".")[0]).unique().tolist()
        self.test_protein = [protein for protein in self.protein_dirs if protein in tm_proteins]
        self.protein_dirs = [protein for protein in self.protein_dirs if protein not in tm_proteins]
        if DEBUG:
            self.protein_dirs = self.protein_dirs[:5]
        # # Train test split
        # self.training_protein, self.val_proteins = train_test_split(self.protein_dirs, test_size=VAL_RATIO, random_state=RANDOM_SEED)
        # if train:
        #     self.protein_dirs = self.training_protein
        # else:
        #     self.protein_dirs = self.val_proteins

    def __len__(self):
        if self.train:
            return len(self.protein_dirs)
        else:
            return len(self.test_protein)

    def __getitem__(self, idx):
        if self.train:
            return self.load_protein_data(idx)
        else:
            return self.load_test_protein_data(idx)
        
    def load_test_protein_data(self, idx):
        protein_dir = os.path.join(self.tensor_root_dir, self.test_protein[idx])
        mutations_path = os.path.join(self.mutations_root_dir, f'{self.test_protein[idx]}.csv')
        mutations = pd.read_csv(mutations_path)
        mutations = mutations[~mutations['mut_type'].str.contains('ins|del')].reset_index(drop=True)
        # Load and preprocess the data for each protein
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS),weights_only=True)
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G),weights_only=True)
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS),weights_only=True)
        one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT),weights_only=True)
        embedding_tensor = self.load_embedding_tensor(os.path.join(protein_dir, PROTT5_EMBEDDINGS))
        
        indexes = set(mutations.index)
        # remove unstable mut
        if not self.unstable_mut:
            indexes -= set(mutations[mutations['ddG_ML'] == '-'].index)
                 
        # remove the mutations with more than one mutation
        if self.one_mut:
            indexes -= set(mutations[mutations['mut_type'].str.contains(':')].index)
       
        indexes = list(indexes)
        mutations = mutations.loc[indexes]
        delta_g_tensor = delta_g_tensor[indexes]
        one_hot_tensor = one_hot_tensor[indexes]
        embedding_tensor = embedding_tensor[indexes]
        
        mutations_data = {
            'name': self.test_protein[idx],
            'mutations': mutations['mut_type'].to_list(),
            'prott5': embedding_tensor,
            'coords': coords_tensor,
            'one_hot': one_hot_tensor,
            'delta_g': delta_g_tensor,
            'masks': mask_tensor
        }
        
        return mutations_data
    
    def load_protein_data(self, idx):
        
        protein_dir = os.path.join(self.tensor_root_dir, self.protein_dirs[idx])
        mutations_path = os.path.join(self.mutations_root_dir, f'{self.protein_dirs[idx]}.csv')
        mutations = pd.read_csv(mutations_path)
        mutations = mutations[~mutations['mut_type'].str.contains('ins|del')].reset_index(drop=True)
        # Load and preprocess the data for each protein
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS),weights_only=True)
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G),weights_only=True)
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS),weights_only=True)
        one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT),weights_only=True)
        embedding_tensor = self.load_embedding_tensor(os.path.join(protein_dir, PROTT5_EMBEDDINGS))
        
        indexes = set(mutations.index)
        # remove unstable mut
        if not self.unstable_mut:
            indexes -= set(mutations[mutations['ddG_ML'] == '-'].index)
                 
        # remove the mutations with more than one mutation
        if self.one_mut:
            indexes -= set(mutations[mutations['mut_type'].str.contains(':')].index)
       
        indexes = list(indexes)
        mutations = mutations.loc[indexes]
        delta_g_tensor = delta_g_tensor[indexes]
        one_hot_tensor = one_hot_tensor[indexes]
        embedding_tensor = embedding_tensor[indexes]
            
        mutations_data = {
            'name': self.protein_dirs[idx],
            'mutations': mutations['mut_type'].to_list(),
            'prott5': embedding_tensor,
            'coords': coords_tensor,
            'one_hot': one_hot_tensor,
            'delta_g': delta_g_tensor,
            'masks': mask_tensor
        }

        return mutations_data

    def load_embedding_tensor(self, embeddings_dir):
        embeddings = []
        all_embedding_files = sorted(glob.glob(os.path.join(embeddings_dir, 'prott5_embedding_*.pt')),
                                     key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        for filename in all_embedding_files:
            if filename.endswith('.pt'):
                embedding_tensor = torch.load(filename,weights_only=True).to('cpu') # load the tensor to cpu memory
                embeddings.append(embedding_tensor)
        return torch.vstack(embeddings)


# Trainer class

class Trainer():
    def __init__(self, model, train_ds, val_ds, device = DEVICE):
        self.model = model.to(device)
        # ddg_head experiment: attach a learned mutation-aware ddG head; register on the model
        # so model.parameters() (and the freeze logic below) include it.
        self.ddg_head = None
        if LOSS_MODE == 'ddg_head':
            self.ddg_head = DDGHead().to(device)
            self.model.ddg_head = self.ddg_head
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = device
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        self.model.to(self.device)
        self.mini_batch_size = MINI_BATCH_SIZE
        self.model_name = 'PEM_fine_tuned' if FREEZE_LAYERS else 'PEM_full_trained'

    def train(self, epochs = 10, kf = 0):
        """
        Train the model
        args:
        epochs: int, number of epochs
        kf: int, kfold number
        """
        run = None
        if not DEBUG:
            run = wandb.init(project='KF-1MUT-MegaScaleFineTuning', config=config, name=MODEL_NAME + f'Kfold{kf}')
        
        # Freeze the layers and only train the last layer
        if FREEZE_LAYERS:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc2.parameters():
                param.requires_grad = True
            for param in self.model.fc1.parameters():
                param.requires_grad = True
            if LIGHT_ATTENTION:
                for param in self.model.LA.parameters():
                    param.requires_grad = True
        running_loss = 0
        wandb_step = 0
        for epoch in range(epochs):
            self.model.train()
            # lever 1: window of recent proteins (re-forwarded at the boundary; only when ON)
            window_batches = []
            for i, batch in enumerate(tqdm(self.train_ds, desc=f'Training Epoch: {epoch}')):
                batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    self.optimizer.zero_grad()
                    # ddg_head experiment: learned mutation-aware head predicts ddG directly
                    # (no energy / no dG-difference). Bypasses the energy path entirely.
                    if LOSS_MODE == 'ddg_head':
                        ddg_pred = self.get_ddg_head(batch, j)
                        delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                        ddg_true = delta_g - batch['delta_g'][0, 0].to(self.device)
                        loss = self.criterion(ddg_pred, ddg_true)
                        loss.backward(); self.optimizer.step()
                        tpc = torch.corrcoef(torch.stack([ddg_pred.detach(), ddg_true]))[0,1] if ddg_pred.numel() > 1 and ddg_pred.std() > 0 else torch.tensor(float('nan'))
                        batch_loss += loss.item(); wandb_step += 1
                        wandb_log({'loss': loss.item(), 'epoch': epoch, 'batch': i,
                                   'ddg_head_loss': loss.item(), 'wandb_step': wandb_step,
                                   'train_pc_corr': tpc}, run)
                        continue
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                    l1_loss = self.criterion(output, delta_g)
                    # WS1: direct mutation-aware ddG term (wild-type = row 0 of the protein tensors).
                    wt_anchor_loss = torch.zeros((), device=self.device)
                    if LOSS_MODE in ('ddg', 'joint') or WT_ANCHOR_WEIGHT > 0:
                        wt_dg, _, _ = self.get_wt_deltaG(batch)
                        delta_g_wt = batch['delta_g'][0, 0].to(self.device)
                        if LOSS_MODE in ('ddg', 'joint'):
                            ddg_pred = output - wt_dg
                            ddg_loss = self.criterion(ddg_pred, delta_g - delta_g_wt)
                        else:
                            ddg_loss = torch.zeros((), device=self.device)
                        # WS-1 Arm A: pin pred_dG(WT) to the experimental WT abs-dG. Only the WT row
                        # (not all variants, which diluted under 'joint'); supplies the absolute scale
                        # the ddG objective leaves free, so the per-protein baseline/offset is calibrated.
                        if WT_ANCHOR_WEIGHT > 0:
                            wt_anchor_loss = self.criterion(wt_dg.squeeze(), delta_g_wt.squeeze())
                    else:
                        ddg_loss = torch.zeros((), device=self.device)
                    if FREEZE_LAYERS:
                        reg_loss = REG_LAMBDA * (F.mse_loss(self.model.fc1.weight,torch.zeros_like(self.model.fc1.weight)) + F.mse_loss(self.model.fc2.weight,torch.zeros_like(self.model.fc2.weight)))
                    else:
                        reg_loss = REG_LAMBDA * sum([F.mse_loss(param,torch.zeros_like(param)) for param in self.model.parameters()])
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    # Agent-E SLOPE term (thesis contribution): penalise the mismatch between the
                    # spread of predicted vs true ddG WITHIN this protein/minibatch. ddG uses the
                    # exact WS-1 convention (WT = row 0; ddg = output - wt_dg vs delta_g - delta_g_wt).
                    # Guarded: when SLOPE_WEIGHT == 0 NOTHING is built, so the autograd graph + loss
                    # value are bit-identical to baseline. unbiased=False for determinism; skip <2 vars.
                    if SLOPE_WEIGHT > 0 and output.numel() >= 2:
                        wt_dg_slope, _, _ = self.get_wt_deltaG(batch)
                        delta_g_wt_slope = batch['delta_g'][0, 0].to(self.device)
                        slope_pred_ddg = output - wt_dg_slope
                        slope_true_ddg = delta_g - delta_g_wt_slope
                        slope_loss = torch.abs(slope_pred_ddg.std(unbiased=False) - slope_true_ddg.std(unbiased=False))
                    data_loss = l1_loss if LOSS_MODE == 'dg' else (ddg_loss if LOSS_MODE == 'ddg' else l1_loss + DDG_WEIGHT * ddg_loss)
                    loss = data_loss + reg_loss + energy_reg + WT_ANCHOR_WEIGHT * wt_anchor_loss
                    if SLOPE_WEIGHT > 0 and output.numel() >= 2:
                        loss = loss + SLOPE_WEIGHT * slope_loss
                    loss.backward()
                    self.optimizer.step()
                    train_pc_corr = torch.corrcoef(torch.cat((output[None,:],delta_g[None,:])))[0, 1]
                    batch_loss += loss.item()
                    wandb_step += 1
                    wandb_log({'loss': loss.item(), 'epoch': epoch, 'batch': i,'l1_loss': l1_loss.item(), 'reg_loss': reg_loss.item(),
                               'energy_reg': energy_reg.item(), 'wandb_step': wandb_step,
                               'wt_anchor_loss': float(wt_anchor_loss), 'ddg_loss': float(ddg_loss),
                               'train_pc_corr': train_pc_corr},run)
                batch_loss /= batch_idx
                running_loss += batch_loss
                if (i+1) % 100 == 0:
                    wandb_log({'epoch': epoch, 'running_loss': running_loss/100},run)
                    running_loss = 0
                # lever 1: stash this (already-normalized) protein; every POOLED_WINDOW proteins,
                # re-forward the window with CURRENT params and take one pooled-correlation step.
                # (Re-forwarding avoids retaining graphs across the per-mini-batch optimizer steps,
                # which would otherwise depend on parameters mutated in-place between collection and backward.)
                if POOLED_CORR_WEIGHT > 0:
                    window_batches.append(batch)
                    if len(window_batches) >= POOLED_WINDOW:
                        wandb_step = self._pooled_step(window_batches, epoch, wandb_step, run)
                        window_batches = []

            # lever 1: flush the partial window at epoch end
            if POOLED_CORR_WEIGHT > 0 and window_batches:
                wandb_step = self._pooled_step(window_batches, epoch, wandb_step, run)
                window_batches = []

            # save the model
            if not DEBUG:
                torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME, f'kf_{kf}_epoch_{epoch}.pt'))
            
            pc_corr, val_loss = self.validate(epoch,run)
            self.model.train()
            # update the learning rate
            self.scheduler.step(pc_corr)
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb_log({'epoch': epoch, 'lr': current_lr}, run)
        return self.model, pc_corr

    def validate(self, epoch, run = None, test = False):
        """
        Validate the model
        args:
        epoch: int, the current epoch
        run: wandb run object
        returns:
        pc_corr: float, pearson correlation
        """
        self.model.eval()
        val_loss = 0
        val_dg = torch.tensor([],device=self.device)
        val_dg_pred = torch.tensor([],device=self.device)
        ddg_exp_all, ddg_pred_all, perprot_pcc = [], [], []   # benchmark ddG metric
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_ds,desc=f'Validation Epoch: {epoch}')):
                batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                prot_dg, prot_pred = [], []   # this protein's variants (for per-protein ddG)
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    batch_idx += 1
                    delta_g = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                    if LOSS_MODE == 'ddg_head':
                        # head outputs ddG directly; centering below (ppred-ppred[0]) sets WT=0
                        output = self.get_ddg_head(batch, j)
                        loss = self.criterion(output, delta_g - batch['delta_g'][0, 0].to(self.device))
                    else:
                        output,u_energy,f_energy = self.get_deltaG(batch, j)
                        loss = self.criterion(output,delta_g)
                        energys = torch.cat((u_energy,f_energy),dim=0)
                        energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                        loss += energy_reg
                    batch_loss += loss.item()
                    val_dg = torch.cat((val_dg, delta_g), dim=0)
                    val_dg_pred = torch.cat((val_dg_pred, output), dim=0)
                    prot_dg.append(delta_g); prot_pred.append(output)
                    # clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
                batch_loss /= batch_idx
                # ddG = variant - wild-type (row 0 of this protein); the benchmark quantity
                pdg = torch.cat(prot_dg); ppred = torch.cat(prot_pred)
                ddg_e = (pdg - pdg[0]); ddg_p = (ppred - ppred[0])
                ddg_exp_all.append(ddg_e); ddg_pred_all.append(ddg_p)
                if ddg_e.numel() >= 3 and torch.std(ddg_p) > 0:
                    perprot_pcc.append(torch.corrcoef(torch.stack([ddg_e, ddg_p]))[0,1].item())
            val_loss += batch_loss
        val_loss /= len(self.val_ds)
        print(f'Validation Loss: {val_loss}')
        pc_corr = torch.corrcoef(torch.cat((val_dg[None,:],val_dg_pred[None,:])))[0, 1]
        # Benchmark ddG metrics (pooled + per-protein mean) — comparable to thesis Table 3.6
        ddg_e_all = torch.cat(ddg_exp_all); ddg_p_all = torch.cat(ddg_pred_all)
        ddg_pcc = torch.corrcoef(torch.stack([ddg_e_all, ddg_p_all]))[0,1].item()
        ddg_rmse = float(torch.sqrt(torch.mean((ddg_e_all - ddg_p_all)**2)))
        ddg_pcc_pp = float(sum(perprot_pcc)/len(perprot_pcc)) if perprot_pcc else float('nan')
        prefix = 'test' if test else 'val'
        wandb_log({f'{prefix}_loss': val_loss, 'epoch': epoch, f'{prefix}_pc_corr': pc_corr,
                   f'{prefix}_ddg_pcc': ddg_pcc, f'{prefix}_ddg_pcc_pp': ddg_pcc_pp,
                   f'{prefix}_ddg_rmse': ddg_rmse}, run)
        print(f'  ddG PCC={ddg_pcc:.3f}  ddG PCC-PP={ddg_pcc_pp:.3f}  ddG RMSE={ddg_rmse:.3f}')
        if LOSS_MODE == 'ddg_head':
            pc_corr = torch.tensor(ddg_pcc)   # scheduler + epoch-selection on ddG (no dG predicted)
        return pc_corr, val_loss
        
    def fit_affine(self, loader):
        """Lever 3: least-squares affine (a,b) mapping pred_dG -> true_dG, fit on a TRAIN-derived
        (held-out val) loader. Affects RMSE only — PCC/SCC are scale/shift invariant."""
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                batch = normalize_batch(batch, True)
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    output, _, _ = self.get_deltaG(batch, j)
                    dg = batch['delta_g'][0, j: j + self.mini_batch_size].to(self.device)
                    preds.append(output.detach().float().cpu())
                    trues.append(dg.detach().float().cpu())
        p = torch.cat(preds).numpy(); t = torch.cat(trues).numpy()
        a, b = np.polyfit(p, t, 1)
        return float(a), float(b)

    def get_deltaG(self, batch, i, n=None):
        n = self.mini_batch_size if n is None else n
        # move all to the same device
        one_hot_minibatch = batch['one_hot'][0, i: i + n].to(self.device)
        prott5_embedding_minibatch = batch['prott5'][0, i: i + n].to(self.device)
        batch['coords'] = batch['coords'].to(self.device)
        batch['masks'] = batch['masks'].to(self.device)
        # get the graph
        folded_graph_minibatch = torch.stack(
            [get_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(), prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])
        unfolded_graph_minibatch = torch.stack(
            [get_unfolded_graph(batch['coords'].squeeze(), one_hot_minibatch[i].squeeze(), prott5_embedding_minibatch[i].squeeze(), batch['masks'].squeeze()) for i in
            range(prott5_embedding_minibatch.size(0))])

        all_graph_minibatch = torch.cat([folded_graph_minibatch, unfolded_graph_minibatch], dim=0)

        minibatch_energy = self.model(all_graph_minibatch)
        folded_energy = minibatch_energy[:minibatch_energy.size(0) // 2]
        unfolded_energy = minibatch_energy[minibatch_energy.size(0) // 2:]

        # lever 2: length-normalize dG (raw energies kept for the energy regularizer)
        return (unfolded_energy - folded_energy) / self._dg_norm(batch), unfolded_energy, folded_energy

    def get_ddg_head(self, batch, i, n=None):
        """ddg_head experiment: ddG via the learned head over per-residue features of the
        FOLDED wild-type (row 0) and folded mutant structures (shared backbone). No unfolded
        state, no energy. Returns ddG per variant [n]."""
        n = self.mini_batch_size if n is None else n
        one_hot_mb = batch['one_hot'][0, i: i + n].to(self.device)
        prott5_mb = batch['prott5'][0, i: i + n].to(self.device)
        batch['coords'] = batch['coords'].to(self.device)
        batch['masks'] = batch['masks'].to(self.device)
        coords = batch['coords'].squeeze(); mask = batch['masks'].squeeze()
        folded_mut = torch.stack([get_graph(coords, one_hot_mb[k].squeeze(), prott5_mb[k].squeeze(), mask)
                                  for k in range(prott5_mb.size(0))])
        h_mut = self.model(folded_mut, f_type='features')          # [n,N,128]
        one_hot_wt = batch['one_hot'][0, 0:1].to(self.device)
        prott5_wt = batch['prott5'][0, 0:1].to(self.device)
        folded_wt = get_graph(coords, one_hot_wt[0].squeeze(), prott5_wt[0].squeeze(), mask).unsqueeze(0)
        h_wt = self.model(folded_wt, f_type='features')            # [1,N,128]
        return self.ddg_head(h_wt, h_mut)                          # [n]

    def get_wt_deltaG(self, batch):
        """Predicted dG of the wild-type (row 0), recomputed in-graph for the ddG loss."""
        one_hot_wt = batch['one_hot'][0, 0:1].to(self.device)
        prott5_wt = batch['prott5'][0, 0:1].to(self.device)
        batch['coords'] = batch['coords'].to(self.device)
        batch['masks'] = batch['masks'].to(self.device)
        folded = get_graph(batch['coords'].squeeze(), one_hot_wt[0].squeeze(), prott5_wt[0].squeeze(), batch['masks'].squeeze()).unsqueeze(0)
        unfolded = get_unfolded_graph(batch['coords'].squeeze(), one_hot_wt[0].squeeze(), prott5_wt[0].squeeze(), batch['masks'].squeeze()).unsqueeze(0)
        energy = self.model(torch.cat([folded, unfolded], dim=0))
        folded_energy = energy[:1]
        unfolded_energy = energy[1:]
        return (unfolded_energy - folded_energy) / self._dg_norm(batch), unfolded_energy, folded_energy

    def _dg_norm(self, batch):
        """Lever 2: divisor that length-normalizes the predicted dG.
        Raw dG = E_u - E_f is extensive (E = sum of per-residue energies, so ~proportional to N).
        'sqrtn' imposes a per-protein reshape the shared-weight network cannot fully re-absorb,
        reducing the length-driven cross-protein scale inconsistency. 'none' = current behavior."""
        if DG_LENGTH_NORM == 'none':
            return 1.0
        nv = batch['masks'].squeeze().float().sum().clamp(min=1.0)
        return nv if DG_LENGTH_NORM == 'n' else torch.sqrt(nv)

    def _pooled_sample(self, batch):
        """Lever 1: this protein's ddG (pred & true) over the first mini_batch_size variants,
        kept WITH grad so it can join the cross-protein pooled-correlation loss. Capped to bound
        the memory held across a window of proteins."""
        output, _, _ = self.get_deltaG(batch, 0, n=POOLED_CAP)
        wt_dg, _, _ = self.get_wt_deltaG(batch)
        ddg_pred = output - wt_dg
        dg = batch['delta_g'][0, 0:output.size(0)].to(self.device)
        ddg_true = dg - batch['delta_g'][0, 0].to(self.device)
        return ddg_pred, ddg_true

    def _pooled_step(self, window_batches, epoch, wandb_step, run):
        """Lever 1: one optimizer step on POOLED_CORR_WEIGHT * (1 - Pearson) over ddG pooled across
        the window of proteins. The window is RE-FORWARDED here with current params (single graph,
        single backward) — separate from the per-mini-batch within-protein steps."""
        pred_parts, true_parts = [], []
        for b in window_batches:
            ddg_pred_c, ddg_true_c = self._pooled_sample(b)
            pred_parts.append(ddg_pred_c); true_parts.append(ddg_true_c)
        pp = torch.cat(pred_parts); tt = torch.cat(true_parts)
        if pp.numel() >= 3 and torch.std(pp) > 0:
            self.optimizer.zero_grad()
            ploss = POOLED_CORR_WEIGHT * pearson_loss(pp, tt)
            ploss.backward()
            self.optimizer.step()
            wandb_step += 1
            wandb_log({'pooled_corr_loss': ploss.item(), 'epoch': epoch,
                       'pooled_n_variants': int(pp.numel()), 'wandb_step': wandb_step}, run)
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return wandb_step


def train_fold(fold, model = None):
    """"Train the model for a single fold"""
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    
    prot_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                            mutations_root_dir=mutations_root_dir, train=True)
    
    train_ids, test_ids = list(kfold.split(prot_ds))[fold]
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = Subset(prot_ds, train_ids)
    test_subsampler = Subset(prot_ds, test_ids)

    # Create the dataloaders
    train_ds = DataLoader(train_subsampler, batch_size=1, shuffle=False)
    val_ds = DataLoader(test_subsampler, batch_size=1, shuffle=False)
      
    print(f'FOLD {fold}')
    print('--------------------------------')
    if model is None:
        # Create the model
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,dropout_rate = CFG.dropout_rate,
                    light_attention=LIGHT_ATTENTION, readout=READOUT).to(DEVICE)
        if PRETRAINED: 
            try:
                model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
            except:
                model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    
    # Train the model
    trainer = Trainer(model, train_ds, val_ds)
    model, pc_corr = trainer.train(epochs = EPOCHS, kf=fold)
    
    return model, pc_corr

def test_fold(fold, model):
    """Test the model for a single fold"""
    prot_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                            mutations_root_dir=mutations_root_dir, train=False)
    # Create the dataloaders
    test_dl = DataLoader(prot_ds, batch_size=1, shuffle=False)
      
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Test the model
    trainer = Trainer(model, None, test_dl)
    pc_corr, val_loss = trainer.validate(0, test=True)
    wandb_log({'test_pc_corr': pc_corr, 'test_loss': val_loss})
    print(f'Pearson Correlation: {pc_corr} , Test Loss: {val_loss}')
    wandb.finish()
    
    

def run_training():
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)

    results = {}

    prot_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                          mutations_root_dir=mutations_root_dir, train=True)
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(prot_ds)):
        if ONLY_FOLD is not None and fold != ONLY_FOLD:
            continue
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = Subset(prot_ds, train_ids)
        test_subsampler = Subset(prot_ds, test_ids)
    
        # Create the dataloaders
        train_ds = DataLoader(train_subsampler, batch_size=1, shuffle=False)
        val_ds = DataLoader(test_subsampler, batch_size=1, shuffle=False)
    
        # Create the model
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,dropout_rate = CFG.dropout_rate,
                    light_attention=LIGHT_ATTENTION, readout=READOUT).to(DEVICE)
        if PRETRAINED: 
            try:
                model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
            except:
                model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
        
        # Train the model
        trainer = Trainer(model, train_ds, val_ds)
        model, pc_corr = trainer.train(epochs = (PILOT_EPOCHS or EPOCHS), kf=fold)
        results[fold] = pc_corr
        wandb.finish()
        if MAX_FOLDS is not None and (fold + 1) >= MAX_FOLDS:
            print(f'Pilot: stopping after {fold + 1} fold(s)')
            break

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        
def get_valid_proteins(val_ds):
    # create dataframe and append the name of the protein and the mutations
    df = pd.DataFrame(columns=['name', 'mutations'])
    for i, batch in enumerate(val_ds):
        df = df.append({'name': batch['name'][0]}, ignore_index=True)
    
    df.to_csv('validation_proteins_mutations.csv', index=False)
    
    return df
if __name__ == '__main__':
    tensor_root_dir = r'./data/Processed_K50_dG_datasets/training_data'
    mutations_root_dir = r'./data/Processed_K50_dG_datasets/mutation_datasets'
    CFG.dropout_rate = DROP_OUT
    if UNFREEZE_FROM:
        # Phase 2: full-training (unfreeze ALL layers) from a freeze-phase checkpoint, low LR.
        # Trains on ALL train proteins, validates on the 28-protein test each epoch.
        FREEZE_LAYERS = False
        LR = UNFREEZE_LR
        prot_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                              mutations_root_dir=mutations_root_dir, train=True)
        test_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                              mutations_root_dir=mutations_root_dir, train=False)
        train_dl = DataLoader(prot_ds, batch_size=1, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate,
                    light_attention=LIGHT_ATTENTION, readout=READOUT).to(DEVICE)
        sd = torch.load(UNFREEZE_FROM, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd)
        print(f'UNFREEZE phase from {UNFREEZE_FROM}: all layers trainable, lr={LR}, {UNFREEZE_EPOCHS} epochs, {len(prot_ds)} train proteins')
        trainer = Trainer(model, train_dl, test_dl)
        trainer.train(epochs=UNFREEZE_EPOCHS, kf='full')
    elif FULL_DATA:
        # Train on ALL train proteins (the 28 ThermoMPNN test proteins are excluded by
        # AllProteinValidationDataset(train=True)). Lever 0: carve a held-out VAL split from these
        # train proteins for epoch selection/scheduler — the 28-test is NEVER seen during training
        # (it is scored post-hoc via evaluate.py). VAL_FRAC=0 falls back to legacy 28-test validation.
        prot_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                              mutations_root_dir=mutations_root_dir, train=True)
        if VAL_FRAC and VAL_FRAC > 0:
            tr_idx, va_idx = train_test_split(list(range(len(prot_ds))), test_size=VAL_FRAC,
                                              random_state=RANDOM_SEED)
            train_sub = Subset(prot_ds, tr_idx)
            if DESIGNED_WEIGHT != 1.0:
                # EXP-19: oversample designed-fold minis in the TRAIN subset (val untouched)
                names = [prot_ds.protein_dirs[i] for i in tr_idx]
                w = torch.tensor([DESIGNED_WEIGHT if is_designed(n) else 1.0 for n in names], dtype=torch.double)
                ndes = int(sum(is_designed(n) for n in names))
                sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(train_sub), replacement=True)
                train_dl = DataLoader(train_sub, batch_size=1, sampler=sampler)
                print(f'EXP-19 designed-fold reweight: {ndes}/{len(names)} train proteins are designed, '
                      f'weight x{DESIGNED_WEIGHT} (WeightedRandomSampler)')
            else:
                train_dl = DataLoader(train_sub, batch_size=1, shuffle=True)
            val_dl = DataLoader(Subset(prot_ds, va_idx), batch_size=1, shuffle=False)
            print(f'FULL-DATA: {len(tr_idx)} train / {len(va_idx)} held-out val proteins '
                  f'(epoch selection on VAL; 28-test untouched, score with evaluate.py)')
        else:
            test_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                                  mutations_root_dir=mutations_root_dir, train=False)
            train_dl = DataLoader(prot_ds, batch_size=1, shuffle=True)
            val_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
            print(f'FULL-DATA (legacy VAL_FRAC=0): {len(prot_ds)} train, validating on 28-test')
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate,
                    light_attention=LIGHT_ATTENTION, readout=READOUT).to(DEVICE)
        if PRETRAINED:
            try:
                model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
            except Exception:
                model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
        trainer = Trainer(model, train_dl, val_dl)
        trainer.train(epochs=(PILOT_EPOCHS or EPOCHS), kf='all')
        if AFFINE_CALIB:
            # lever 3: fit a global affine (a,b) on the held-out val split (TRAIN-derived, not test);
            # evaluate.py applies it via env DEEPEF_AFFINE. Affects RMSE only.
            a, b = trainer.fit_affine(val_dl)
            import json as _json
            _affine_path = os.path.join(MODEL_PATH, MODEL_NAME, 'affine.json')
            with open(_affine_path, 'w') as _f:
                _json.dump({'a': a, 'b': b, 'fit_on': 'val', 'dg_length_norm': DG_LENGTH_NORM}, _f)
            print(f'AFFINE fit a={a:.4f} b={b:.4f} -> {_affine_path} '
                  f'(apply in eval: DEEPEF_AFFINE={_affine_path})')
    else:
        # Freeze-phase k-fold training (saves kf_{fold}_epoch_{epoch}.pt under MODEL_NAME).
        # Pilot: --max_folds 1 --epochs N; evaluate checkpoints with evaluate.py on 28-protein test.
        run_training()

    # Get validation proteins
    # protein_val = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
    #                                               mutations_root_dir=mutations_root_dir, train=False)
    # # Create the dataloaders
    # val_ds = DataLoader(protein_val, batch_size=1, shuffle=False)
    
    # get_valid_proteins(val_ds)
