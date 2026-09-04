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
from model.hydro_net import build_energy_model
from model.model_cfg import CFG
from train_utils import get_graph, get_unfolded_graph, load_checkpoint
import wandb
from tqdm import tqdm
from sklearn.model_selection import KFold
import gc
import pandas as pd
# parser
import argparse
# import the new dataset
from new_dataset import MSDataset

parser = argparse.ArgumentParser(description='Train the model with the mega-scale data')
parser.add_argument('--debug',default=False, action='store_true', help='Debug mode')
parser.add_argument('--epochs', type=int, default=50, help='Total epochs; split into frozen+unfrozen two-stage (default 1/3 frozen, 2/3 unfrozen) unless --epochs_freeze/--epochs_no_freeze given')
parser.add_argument('--epochs_freeze', type=int, default=-1, help='Override: epochs in the frozen stage (-1 = derive from --epochs)')
parser.add_argument('--epochs_no_freeze', type=int, default=-1, help='Override: epochs in the unfrozen stage (-1 = derive from --epochs)')
parser.add_argument('--model_name', type=str, default='PEM_fine_tuned', help='Model name')
parser.add_argument('--dataset_type', type=str, default='pnas', help='Dataset type')
parser.add_argument('--unstable_mut', action='store_true', help='Save the unstable mutations')
parser.add_argument('--one_mut', action='store_true', help='Remove the multiple mutations when fine-tuning')
parser.add_argument('--freeze_layers',action = 'store_true', help ='Freeze model layers except mlp and LA')
parser.add_argument('--trained_model_path',type=str,default = "./res/trianed_models-light_attention/43_final_model.pt",help='Trained model path')
parser.add_argument('--dg_ml', action='store_true', help='Change deltaG threshold to [-1,5]')
parser.add_argument('--model_arch', type=str, default='pem', choices=['pem', 'graph_transformer'], help='Energy model architecture')
parser.add_argument('--emb_projection', type=str, default='mlp', choices=['none', 'mlp', 'low_rank'], help='Protein embedding projection')
parser.add_argument('--gt_hidden_dim', type=int, default=64, help='Graph Transformer hidden dimension')
parser.add_argument('--gt_heads', type=int, default=4, help='Graph Transformer attention heads')
parser.add_argument('--gt_layers', type=int, default=3, help='Graph Transformer layer count')
parser.add_argument('--gt_edge_cutoff', type=float, default=12.0, help='Graph Transformer CA edge cutoff')
parser.add_argument('--loss_mode', type=str, default='dg', choices=['dg', 'ddg', 'joint'],
                    help="Fine-tuning objective: 'dg' (predict deltaG, current), "
                         "'ddg' (direct mutation-aware ddG = dG_mut - dG_wt), "
                         "'joint' (dG + ddg_weight * ddG)")
parser.add_argument('--ddg_weight', type=float, default=1.0,
                    help='Weight of the ddG term when --loss_mode joint')
# --- offset-attack additions (branch offset-attack-run; defaults preserve baseline behavior) ---
parser.add_argument('--no_pretrained', action='store_true',
                    help='Train from scratch (PRETRAINED=False): do NOT load any checkpoint. '
                         'Matches the from-scratch 0.531/0.655 regime; also required when '
                         'the pretrained core (43_final_model.pt) is absent.')
parser.add_argument('--mini_batch_size', type=int, default=16,
                    help='Mini-batch of variants per forward pass (OOM rule: 16, not the old 64).')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for torch/numpy/random (deterministic run).')
parser.add_argument('--flory_unfolded', action='store_true',
                    help='Lever D: use an analytic Flory random-coil unfolded reference '
                         '(d(i,j)=b*|i-j|^nu) instead of the tridiagonal-mask unfolded state.')
parser.add_argument('--flory_nu', type=float, default=0.5,
                    help='Lever D coil scaling exponent in (0,1]; 0.5=ideal chain, ~0.588=SAW.')
parser.add_argument('--dump_energies', action='store_true',
                    help='During validate/eval, write per-protein mean Eu/Ef/dG to a CSV '
                         '(mechanism diagnostic; does not affect the normal path).')
parser.add_argument('--resume', action='store_true',
                    help='Resume from the latest epoch_N.pt in the model dir. Loads that state '
                         'and restarts at epoch N+1, correctly split across the frozen/unfrozen '
                         'stages. Makes a crash cost one epoch instead of the whole run.')

args, _ = parser.parse_known_args()
CFG.model_arch = args.model_arch
CFG.emb_projection = args.emb_projection
CFG.gt_hidden_dim = args.gt_hidden_dim
CFG.gt_heads = args.gt_heads
CFG.gt_layers = args.gt_layers
CFG.gt_edge_cutoff = args.gt_edge_cutoff
# Lever D (analytic Flory coil unfolded reference) — read by train_utils.get_unfolded_graph
# off the SAME model.model_cfg.CFG object. Default OFF => bit-identical tridiagonal baseline.
CFG.flory_unfolded = args.flory_unfolded
CFG.flory_nu = args.flory_nu


def set_seed(seed):
    """Deterministic seeding for torch / numpy / python-random."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(args.seed)

# Constants
COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'prott5_embeddings'
VAL_RATIO = 0.2
RANDOM_SEED = 42
NANO_TO_ANGSTROM = 0.1
DEBUG  = args.debug
# Two-stage schedule now derives from --epochs (was hardcoded 20+60=80; our prior runs plateau by
# epoch ~12-14, so default --epochs 15 -> 5 frozen + 10 unfrozen). Explicit overrides win.
if args.epochs_freeze >= 0 or args.epochs_no_freeze >= 0:
    EPOCHS_FREEZE = args.epochs_freeze if args.epochs_freeze >= 0 else 0
    EPOCHS_NO_FREEZE = args.epochs_no_freeze if args.epochs_no_freeze >= 0 else 0
else:
    EPOCHS_FREEZE = max(1, args.epochs // 3)
    EPOCHS_NO_FREEZE = args.epochs - EPOCHS_FREEZE
if DEBUG:
    EPOCHS_FREEZE = 1
    EPOCHS_NO_FREEZE = 1
FREEZE_LAYERS = args.freeze_layers
CRITERION = "L1"
MODEL_PATH = './Megascale-fineTuning/models'
MINI_BATCH_SIZE = args.mini_batch_size
DEVICE = 'cuda'# if torch.cuda.is_available() else 'cpu'
TRAINED_MODEL_PATH = args.trained_model_path
BASE_MODEL_NAME = TRAINED_MODEL_PATH.split('/')[-2]
MODEL_NAME = args.model_name
PRETRAINED = not args.no_pretrained
TM_PATH = "./data/ThermoMPNN/mega_test.csv"
PNAS_PROTEINS = "./data/Processed_K50_dG_datasets/Pnas_filtering/train_proteins.csv"
PNAS_MUT = "./data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv"
LR = 1e-4
DROP_OUT = 0.2
REG_LAMBDA = 0
E_REG_LAMBDA = 0.001
UNSTABLE_MUT = args.unstable_mut
DS_TYPE = args.dataset_type
LIGHT_ATTENTION = True
ONE_MUT =  args.one_mut
DG_ML = args.dg_ml
LOSS_MODE = args.loss_mode
DDG_WEIGHT = args.ddg_weight

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
    'epochs': EPOCHS_FREEZE,
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
    'ds_type': DS_TYPE,
    'one_mutation': ONE_MUT,
    'dG_ml': DG_ML,
    'loss_mode': LOSS_MODE,
    'ddg_weight': DDG_WEIGHT,
    'no_pretrained': args.no_pretrained,
    'seed': args.seed,
    'flory_unfolded': CFG.flory_unfolded,
    'flory_nu': CFG.flory_nu,
    'dump_energies': args.dump_energies,
    'model_arch': CFG.model_arch,
    'emb_projection': CFG.emb_projection,
    'gt_hidden_dim': CFG.gt_hidden_dim,
    'gt_heads': CFG.gt_heads,
    'gt_layers': CFG.gt_layers,
    'gt_edge_cutoff': CFG.gt_edge_cutoff
}

if not os.path.exists(os.path.join(MODEL_PATH, MODEL_NAME)):
    os.makedirs(os.path.join(MODEL_PATH, MODEL_NAME))

if not DEBUG and os.path.basename(sys.argv[0]) == 'pnas_train.py':
    wandb.init(project='Megascale-fineTuning', config=config)
    wandb.run.name = MODEL_NAME

def wandb_log(log_dict,run = None):
    if not DEBUG:
        if run is not None:
            run.log(log_dict)
        elif wandb.run is not None:
            wandb.log(log_dict)

def normalize_batch(batch, LLM_EMB = True):
    batch['one_hot'] = batch['one_hot'][:, :, :, :-1]
    batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM
    if not LLM_EMB: # zero prot5 embedding
         batch['prott5'] = torch.zeros_like(batch['prott5'])
    return batch    

class AllProteinValidationDataset(Dataset):

    def __init__(self, tensor_root_dir, mutations_root_dir, train  = True, one_mut = ONE_MUT ):
        self.tensor_root_dir = tensor_root_dir
        self.train = train
        self.mutations_root_dir = mutations_root_dir
        self.protein_dirs = [protein for i, protein in enumerate(os.listdir(self.tensor_root_dir))]
        self.one_mut = one_mut # remove the mutations with more than one mutation
        self.unstable_mut = UNSTABLE_MUT
        self.ds_type = DS_TYPE
        self.dG_ml = DG_ML
        # remove TM proteins 
        tm_proteins = pd.read_csv(TM_PATH)
        tm_proteins = tm_proteins['name'].apply(lambda x: x.split(".")[0]).unique().tolist()
        self.test_protein = [protein for protein in self.protein_dirs if protein in tm_proteins]
        self.protein_dirs = [protein for protein in self.protein_dirs if protein not in tm_proteins]
        self.pnas_mutations = pd.read_csv(PNAS_MUT)
        self.test_mutations = pd.read_csv(TM_PATH)
        # Remove the homologs from the training set
        self.remove_homologs()
        
        if DEBUG:
            self.protein_dirs = self.protein_dirs[:5]
        


    def remove_homologs(self):
        """Remove the homologs from the training set"""
        pnas_proteins = pd.read_csv(PNAS_PROTEINS)['protein_name'].tolist()
        self.protein_dirs = [protein for protein in self.protein_dirs if protein in pnas_proteins]
        
        
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
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS), weights_only=True)
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G), weights_only=True)
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS), weights_only=True)
        one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT), weights_only=True)
        embedding_tensor = self.load_embedding_tensor(os.path.join(protein_dir, PROTT5_EMBEDDINGS))
        
        # If dG_ml is check save the threshold of -1 and 5
        if self.dG_ml:
            threshold = [-1.0, 5.0]
            delta_g_tensor = torch.where(delta_g_tensor > threshold[0], delta_g_tensor, threshold[0])
            delta_g_tensor = torch.where(delta_g_tensor < threshold[1], delta_g_tensor, threshold[1])
        
        indexes = set(mutations.index)
        
        # remove unstable mut
        if not self.unstable_mut:
            indexes -= set(mutations[mutations['ddG_ML'] == '-'].index)
             
        # remove the mutations with more than one mutation
        if self.one_mut:
            indexes -= set(mutations[mutations['mut_type'].str.contains(':')].index)
        
        if self.ds_type in ('pnas', 'deepef1'):
            # get the pnas mutations indexes
            indexes -= set(mutations[~mutations['name'].isin(self.test_mutations['name'])].index)
        
        # Ensure iloc[0] is not removed by adding it as the first index for ddg calc
        indexes.insert(0, 0)
        
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
        
        # Check if deltaG thershold is set and apply it to the mutations dataframe
        if self.dG_ml:
            threshold = [-1,5]
            delta_g_tensor = torch.where(delta_g_tensor > threshold[0], delta_g_tensor, threshold[0])
            delta_g_tensor = torch.where(delta_g_tensor < threshold[1], delta_g_tensor, threshold[1])
        
        indexes = set(mutations.index)
        # remove unstable mut
        if not self.unstable_mut:
            indexes -= set(mutations[mutations['ddG_ML'] == '-'].index)
                 
        # remove the mutations with more than one mutation
        if self.one_mut:
            indexes -= set(mutations[mutations['mut_type'].str.contains(':')].index)
        
        if self.ds_type == 'pnas':
            # get the pnas mutations indexes
            indexes -= set(mutations[~mutations['name'].isin(self.pnas_mutations['name'])].index)
       
        # Ensure iloc[0] is not removed by adding it as the first index for ddg calc
        indexes.insert(0, 0)
        
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
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = device
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5)
        self.model.to(self.device)
        self.mini_batch_size = MINI_BATCH_SIZE
        self.model_name = 'PEM_fine_tuned' if FREEZE_LAYERS else 'PEM_full_trained'

    def handle_freez_layers(self):
        """ Freeze the layers and only train the last layer"""
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
        else:
            for param in self.model.parameters():
                param.requires_grad = True
    
    def train(self, epochs = 10, s_epoch = 0):
        """
        Train the model
        args:
        epochs: int, number of epochs
        """
        run = None
        # freeze the layers
        self.handle_freez_layers()
        
        wandb_step = 0
        for epoch in range(s_epoch, s_epoch + epochs):
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_ds, desc=f'Training Epoch: {epoch}')):
                # batch = normalize_batch(batch, True)
                if batch['delta_g'].size(1) == 1:
                    continue
                batch_loss = 0
                batch_idx = 1
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    self.optimizer.zero_grad()
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0,j: j + self.mini_batch_size].to(self.device)
                    l1_loss = self.criterion(output, delta_g)
                    # WS1: direct mutation-aware ddG objective (ddG = dG_mut - dG_wt).
                    # The wild-type is row 0 of every per-protein tensor (see new_dataset.py).
                    if LOSS_MODE in ('ddg', 'joint'):
                        wt_dg, wt_u, wt_f = self.get_wt_deltaG(batch)
                        ddg_pred = output - wt_dg                       # [mb] - [1] broadcast
                        delta_g_wt = batch['delta_g'][0, 0].to(self.device)
                        ddg_true = delta_g - delta_g_wt
                        ddg_loss = self.criterion(ddg_pred, ddg_true)
                    else:
                        ddg_loss = torch.zeros((), device=self.device)
                    if FREEZE_LAYERS:
                        reg_loss = REG_LAMBDA * (F.mse_loss(self.model.fc1.weight,torch.zeros_like(self.model.fc1.weight)) + F.mse_loss(self.model.fc2.weight,torch.zeros_like(self.model.fc2.weight)))
                    else:
                        reg_loss = REG_LAMBDA * sum([F.mse_loss(param,torch.zeros_like(param)) for param in self.model.parameters()])
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    if LOSS_MODE == 'dg':
                        data_loss = l1_loss
                    elif LOSS_MODE == 'ddg':
                        data_loss = ddg_loss
                    else:  # joint
                        data_loss = l1_loss + DDG_WEIGHT * ddg_loss
                    loss = data_loss + reg_loss + energy_reg
                    loss.backward()
                    self.optimizer.step()
                    train_pc_corr = torch.corrcoef(torch.cat((output[None,:],delta_g[None,:])))[0, 1]
                    batch_loss += loss.item()
                    wandb_step += 1
                    wandb_log({'loss': loss.item(), 'epoch': epoch, 'batch': i,'l1_loss': l1_loss.item(),
                               'ddg_loss': ddg_loss.item(), 'reg_loss': reg_loss.item(),
                               'energy_reg': energy_reg.item(), 'wandb_step': wandb_step,
                               'train_pc_corr': train_pc_corr},run)
                
                    
            # save the model
            if not DEBUG:
                print(f"Saving model in path {os.path.join(MODEL_PATH, MODEL_NAME, f'epoch_{epoch}.pt')}")
                # save the model
                torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME, f'epoch_{epoch}.pt'))
            
            pc_corr, val_loss, _ = self.validate(epoch,run)
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
        val_df = pd.DataFrame([],columns=['protein','deltaG','pred_deltaG'])
        # --dump_energies: accumulate per-protein mean unfolded/folded energy + dG (mechanism split).
        energy_rows = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_ds,desc=f'Validation Epoch: {epoch}')):
                # batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                prot_eu = torch.tensor([], device=self.device)
                prot_ef = torch.tensor([], device=self.device)
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    batch_idx += 1
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0,j: j + self.mini_batch_size].to(self.device)
                    loss = self.criterion(output,delta_g)
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    loss += energy_reg
                    batch_loss += loss.item()
                    val_dg = torch.cat((val_dg, delta_g), dim=0)
                    val_dg_pred = torch.cat((val_dg_pred, output), dim=0)
                    if args.dump_energies:
                        prot_eu = torch.cat((prot_eu, u_energy.detach().flatten()), dim=0)
                        prot_ef = torch.cat((prot_ef, f_energy.detach().flatten()), dim=0)
                    batch_df = pd.DataFrame([],columns=['protein','deltaG','pred_deltaG'])
                    batch_df['deltaG'] = delta_g.cpu().numpy()
                    batch_df['pred_deltaG']  = output.cpu().numpy()
                    batch_df['protein'] = [batch['name'][0] for i in range(len(delta_g))]
                    val_df = pd.concat([val_df,batch_df])
                    # clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
                if args.dump_energies and prot_eu.numel() > 0:
                    mean_eu = float(prot_eu.mean().item())
                    mean_ef = float(prot_ef.mean().item())
                    energy_rows.append({'protein': batch['name'][0],
                                        'L': int(batch['masks'].sum().item()),
                                        'n_variants': int(prot_eu.numel()),
                                        'Eu': mean_eu, 'Ef': mean_ef, 'dG': mean_eu - mean_ef})
                batch_loss /= batch_idx
            val_loss += batch_loss
        val_loss /= len(self.val_ds)
        print(f'Validation Loss: {val_loss}')
        # ddG calculation (difference from wildtype, which is the first value)
        dg_wt = val_dg[0].item() if val_dg.numel() > 0 else 0.0
        dg_pred_wt = val_dg_pred[0].item() if val_dg_pred.numel() > 0 else 0.0
        ddg_true = (val_dg - dg_wt).cpu().numpy()
        ddg_pred = (val_dg_pred - dg_pred_wt).cpu().numpy()
        # Pearson correlation (deltaG)
        pc_corr = torch.corrcoef(torch.cat((val_dg[None,:],val_dg_pred[None,:])))[0, 1].item()
        # Spearman correlation (deltaG)
        try:
            from scipy.stats import spearmanr
            sp_corr, _ = spearmanr(val_dg.cpu().numpy(), val_dg_pred.cpu().numpy())
            ddg_pc_corr, _ = spearmanr(ddg_true, ddg_pred)
        except ImportError:
            sp_corr = float('nan')
            ddg_pc_corr = float('nan')
        # RMSE (deltaG)
        rmse = float(torch.sqrt(F.mse_loss(val_dg_pred, val_dg)).item())
        # ddG metrics
        from sklearn.metrics import mean_squared_error
        ddg_rmse = mean_squared_error(ddg_true, ddg_pred, squared=False) if len(ddg_true) > 1 else float('nan')
        # Pearson for ddG
        try:
            from scipy.stats import pearsonr
            ddg_pearson_corr, _ = pearsonr(ddg_true, ddg_pred)
        except ImportError:
            ddg_pearson_corr = float('nan')
        # Log all metrics
        if not test:
            wandb_log({'val_loss': val_loss,
                       'epoch': epoch,
                       'val_pc_corr': pc_corr,
                       'val_sp_corr': sp_corr,
                       'val_rmse': rmse,
                       'val_ddg_pc_corr': ddg_pearson_corr,
                       'val_ddg_sp_corr': ddg_pc_corr,
                       'val_ddg_rmse': ddg_rmse}, run)
        else:
            wandb_log({'test_loss': val_loss,
                       'epoch': epoch,
                       'pc_corr': pc_corr,
                       'sp_corr': sp_corr,
                       'rmse': rmse,
                       'ddg_pc_corr': ddg_pearson_corr,
                       'ddg_sp_corr': ddg_pc_corr,
                       'ddg_rmse': ddg_rmse}, run)
        # Save datafeame
        # val_df.to_csv("val_df.csv",index=False)
        # --dump_energies: write per-protein Eu/Ef/dG so we can split across-protein variance and
        # correlate with wt_err (mirrors analysis/wt_dg_error/gat_gcn_energy.csv). Off the normal path.
        if args.dump_energies and energy_rows:
            edir = os.path.join(MODEL_PATH, MODEL_NAME)
            os.makedirs(edir, exist_ok=True)
            epath = os.path.join(edir, f'energies_epoch_{epoch}.csv')
            pd.DataFrame(energy_rows).to_csv(epath, index=False)
            print(f'[dump_energies] wrote {len(energy_rows)} proteins -> {epath}')
        # Selection / LR-scheduler metric matches the training objective:
        # ddG Pearson for (ddg, joint), deltaG Pearson for dg.
        selection_corr = ddg_pearson_corr if LOSS_MODE in ('ddg', 'joint') else pc_corr
        return selection_corr, val_loss, val_df
        
    def get_deltaG(self, batch, i):
        # move all to the same device
        one_hot_minibatch = batch['one_hot'][0,i: i + self.mini_batch_size].to(self.device)
        prott5_embedding_minibatch = batch['prott5'][0,i: i + self.mini_batch_size].to(self.device)
        batch['coords'] = batch['coords'].to(self.device)
        batch['masks'] = batch['masks'].to(self.device)
        # get the graph
        folded_graph_minibatch = torch.stack(
            [get_graph(batch['coords'].squeeze(), one_hot_minibatch[j].squeeze(), prott5_embedding_minibatch[j].squeeze(), batch['masks'].squeeze()) for j in
            range(prott5_embedding_minibatch.size(0))])
        unfolded_graph_minibatch = torch.stack(
            [get_unfolded_graph(batch['coords'].squeeze(), one_hot_minibatch[j].squeeze(), prott5_embedding_minibatch[j].squeeze(), batch['masks'].squeeze()) for j in
            range(prott5_embedding_minibatch.size(0))])

        all_graph_minibatch = torch.cat([folded_graph_minibatch, unfolded_graph_minibatch], dim=0)

        ca_coords = batch['coords'].squeeze()[:, 1, :].unsqueeze(0).expand(all_graph_minibatch.size(0), -1, -1)
        minibatch_energy = self.model(all_graph_minibatch, ca_coords=ca_coords)
        folded_energy = minibatch_energy[:minibatch_energy.size(0) // 2]
        unfolded_energy = minibatch_energy[minibatch_energy.size(0) // 2:]

        return unfolded_energy - folded_energy,unfolded_energy,folded_energy

    def get_wt_deltaG(self, batch):
        """Predicted deltaG of the wild-type (row 0 of the per-protein tensors).

        Recomputed inside the current autograd graph so its gradient flows into the
        direct-ddG loss. Mirrors get_deltaG for a single (the wild-type) variant.
        """
        one_hot_wt = batch['one_hot'][0, 0:1].to(self.device)
        prott5_wt = batch['prott5'][0, 0:1].to(self.device)
        coords = batch['coords'].to(self.device)
        masks = batch['masks'].to(self.device)
        folded = get_graph(coords.squeeze(), one_hot_wt[0].squeeze(), prott5_wt[0].squeeze(), masks.squeeze()).unsqueeze(0)
        unfolded = get_unfolded_graph(coords.squeeze(), one_hot_wt[0].squeeze(), prott5_wt[0].squeeze(), masks.squeeze()).unsqueeze(0)
        all_graph = torch.cat([folded, unfolded], dim=0)
        ca_coords = coords.squeeze()[:, 1, :].unsqueeze(0).expand(all_graph.size(0), -1, -1)
        energy = self.model(all_graph, ca_coords=ca_coords)
        folded_energy = energy[:1]
        unfolded_energy = energy[1:]
        return unfolded_energy - folded_energy, unfolded_energy, folded_energy


def _find_latest_checkpoint():
    """Return (path, epoch_index) of the highest epoch_N.pt in the model dir, or (None, -1)."""
    import glob, re
    cdir = os.path.join(MODEL_PATH, MODEL_NAME)
    best_e, best_p = -1, None
    for p in glob.glob(os.path.join(cdir, 'epoch_*.pt')):
        m = re.search(r'epoch_(\d+)\.pt$', os.path.basename(p))
        if m:
            e = int(m.group(1))
            if e > best_e:
                best_e, best_p = e, p
    return best_p, best_e


def run_training():
    """Run the training for all the proteins"""
    train_ds = MSDataset(tensor_root_dir=tensor_root_dir,
                                          mutations_root_dir=mutations_root_dir, train=True)

    test_ds = MSDataset(tensor_root_dir=tensor_root_dir,
                                            mutations_root_dir=mutations_root_dir, train=False)

     # Create the dataloaders
    train_ds = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=1, shuffle=True)

    # Create the model
    model = build_energy_model(
        model_arch=CFG.model_arch,
        layers=CFG.num_layers,
        gaussian_coef=CFG.gaussian_coef,
        dropout_rate=CFG.dropout_rate,
        light_attention=LIGHT_ATTENTION,
        emb_projection=CFG.emb_projection,
        gat_cutoff=CFG.gat_cutoff,
    ).to(DEVICE)
    if PRETRAINED:
        try:
            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
        except:
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))

    # --resume: pick up the latest saved epoch. epoch_N.pt is saved at the END of epoch N,
    # so a checkpoint at epoch N means epochs 0..N are done; restart at start_epoch = N+1.
    global FREEZE_LAYERS, LR
    start_epoch = 0
    if args.resume:
        ckpt, last_e = _find_latest_checkpoint()
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            start_epoch = last_e + 1
            print(f'[resume] loaded {ckpt}; restarting at epoch {start_epoch} '
                  f'(freeze stage 0..{EPOCHS_FREEZE-1}, unfrozen {EPOCHS_FREEZE}..{EPOCHS_FREEZE+EPOCHS_NO_FREEZE-1})')
        else:
            print('[resume] no epoch_*.pt found; starting fresh from epoch 0')

    # Stage 1: frozen. Only run the epochs of this stage not already completed.
    if start_epoch < EPOCHS_FREEZE:
        remaining_freeze = EPOCHS_FREEZE - start_epoch
        trainer = Trainer(model, train_ds, test_ds)
        model, pc_corr = trainer.train(epochs=remaining_freeze, s_epoch=start_epoch)
        unfrozen_start = EPOCHS_FREEZE
    else:
        # frozen stage already finished before the crash
        unfrozen_start = start_epoch

    # Stage 2: unfreeze the layers and train with lower learning rate.
    FREEZE_LAYERS = False
    LR = 1e-5
    remaining_unfrozen = (EPOCHS_FREEZE + EPOCHS_NO_FREEZE) - unfrozen_start
    if remaining_unfrozen > 0:
        print('Training the whole model with lower learning rate')
        trainer = Trainer(model, train_ds, test_ds)
        model, pc_corr = trainer.train(epochs=remaining_unfrozen, s_epoch=unfrozen_start)
    wandb.finish()

    print(f'Training completed with Pearson Correlation: {pc_corr}')
    
    
    
    
def get_valid_proteins(val_ds):
    # create dataframe and append the name of the protein and the mutations
    df = pd.DataFrame(columns=['name', 'mutations'])
    for i, batch in enumerate(val_ds):
        df = df.append({'name': batch['name'][0]}, ignore_index=True)
    
    df.to_csv('validation_proteins_mutations.csv', index=False)
    
    return df

def run_validation_metrics():
    """"Rum metrics for validations sets"""
    train_ds = MSDataset(tensor_root_dir=tensor_root_dir,
                                          mutations_root_dir=mutations_root_dir, train=True)
    
    test_ds = MSDataset(tensor_root_dir=tensor_root_dir,
                                            mutations_root_dir=mutations_root_dir, train=False)
    
     # Create the dataloaders
    train_ds = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=1, shuffle=True)
    # Create the model
    model = build_energy_model(
        model_arch=CFG.model_arch,
        layers=CFG.num_layers,
        gaussian_coef=CFG.gaussian_coef,
        dropout_rate=CFG.dropout_rate,
        light_attention=LIGHT_ATTENTION,
        emb_projection=CFG.emb_projection,
        gat_cutoff=CFG.gat_cutoff,
    ).to(DEVICE)
    if PRETRAINED: 
        try:
            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
        except:
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    # Train the model
    trainer = Trainer(model, train_ds, test_ds)
    model, pc_corr,val_df = trainer.validate(0)
    val_df.to_csv("./"+MODEL_NAME+".csv",index=False)
    
if __name__ == '__main__':
    tensor_root_dir = r'./data/MsDs/training_data'
    mutations_root_dir = r'./data/MsDs/mutation_files'
    CFG.dropout_rate = DROP_OUT
    run_training()
    # run_validation_metrics()
