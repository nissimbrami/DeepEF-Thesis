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
from model.hydro_net import PEM
from model.model_cfg import CFG
from train_utils import get_graph, get_unfolded_graph, load_checkpoint, precompute_graph_features, get_graph_fast
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
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--model_name', type=str, default='PEM_fine_tuned', help='Model name')
parser.add_argument('--dataset_type', type=str, default='pnas', help='Dataset type')
parser.add_argument('--unstable_mut', action='store_true', help='Save the unstable mutations')
parser.add_argument('--one_mut', action='store_true', help='Remove the multiple mutations when fine-tuning')
parser.add_argument('--freeze_layers',action = 'store_true', help ='Freeze model layers except mlp and LA')
parser.add_argument('--trained_model_path',type=str,default = "./res/trianed_models-light_attention/43_final_model.pt",help='Trained model path')
parser.add_argument('--dg_ml', action='store_true', help='Change deltaG threshold to [-1,5]')
# Phase 1 experiment flags
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'huber', 'huber_rank', 'l1_rank'],
                    help='Loss: l1=baseline, huber=Exp1.1, huber_rank=Exp1.2+1.1, l1_rank=Exp1.2')
parser.add_argument('--ranking_weight', type=float, default=0.5, help='Weight for pairwise ranking loss')
parser.add_argument('--lora', action='store_true', help='Use LoRA adapters on GNN layers (Exp 1.4)')
parser.add_argument('--no_pretrained', action='store_true', help='Train from random init (no checkpoint required)')
parser.add_argument('--epochs_freeze', type=int, default=5, help='Epochs with frozen GNN layers')
parser.add_argument('--epochs_unfreeze', type=int, default=10, help='Epochs with all layers unfrozen')
parser.add_argument('--max_muts', type=int, default=128, help='Max mutations sampled per protein per epoch (caps huge proteins)')
# Phase 2 architecture flags
parser.add_argument('--phase2', action='store_true', help='Enable all Phase 2 architecture improvements (multi-RBF, k-NN GAT, edge features, larger projection)')
parser.add_argument('--use_multi_rbf', action='store_true', help='Multi-center RBF distance encoding (Phase 2 individual flag)')
parser.add_argument('--use_knn_gat', action='store_true', help='k-NN graph for GAT instead of fully connected (Phase 2 individual flag)')
parser.add_argument('--use_edge_features', action='store_true', help='32-dim edge features in GAT (Phase 2 individual flag)')
parser.add_argument('--emb_proj_dim', type=int, default=None, help='Embedding projection output dim override')
parser.add_argument('--emb_proj_hidden', type=int, default=None, help='Embedding projection hidden dim override')
# Stage 1 training improvement flags
parser.add_argument('--cosine_lr', action='store_true', help='Cosine annealing LR (1e-4 → lr_min over total epochs)')
parser.add_argument('--lr_min', type=float, default=1e-6, help='Min LR for cosine annealing')
parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam weight decay')
# Stage 2 embedding flag
parser.add_argument('--emb_type', type=str, default='prott5',
                    choices=['prott5', 'esm2', 'saprot', 'dual', 'esm2_saprot'],
                    help='Embedding: prott5=1024, esm2=1280, saprot=1280, dual=2304, esm2_saprot=2560')

args = parser.parse_args()

# Apply Phase 2 settings to CFG before model instantiation
if args.phase2:
    CFG.use_multi_rbf = True
    CFG.use_knn_gat = True
    CFG.use_edge_features = True
    CFG.emb_proj_dim = 64
    CFG.emb_proj_hidden = 256
    CFG.gat_cutoff = None
if args.use_multi_rbf:
    CFG.use_multi_rbf = True
if args.use_knn_gat:
    CFG.use_knn_gat = True
if args.use_edge_features:
    CFG.use_edge_features = True
if args.emb_proj_dim is not None:
    CFG.emb_proj_dim = args.emb_proj_dim
if args.emb_proj_hidden is not None:
    CFG.emb_proj_hidden = args.emb_proj_hidden
# Resolve effective embedding projection type for model creation
EMB_PROJ = "mlp" if args.phase2 else "none"

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
EPOCHS_FREEZE = args.epochs_freeze if not DEBUG else 1
EPOCHS_NO_FREEZE = args.epochs_unfreeze if not DEBUG else 1
MAX_MUTS_PER_PROTEIN = args.max_muts
FREEZE_LAYERS = args.freeze_layers
CRITERION = "L1"
MODEL_PATH = './Megascale-fineTuning/models'
MINI_BATCH_SIZE = 16
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
SEED = args.seed
LOSS_TYPE = args.loss_type
RANKING_WEIGHT = args.ranking_weight
USE_LORA = args.lora
COSINE_LR = args.cosine_lr
LR_MIN = args.lr_min
GRAD_ACCUM = args.grad_accum
WEIGHT_DECAY = args.weight_decay
EMB_TYPE = args.emb_type
# Update embedding input dim for non-ProtT5 embeddings
if EMB_TYPE in ('esm2', 'saprot'):
    CFG.emb_input_dim = 1280
elif EMB_TYPE == 'dual':
    CFG.emb_input_dim = 2304
elif EMB_TYPE == 'esm2_saprot':
    CFG.emb_input_dim = 2560

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
    'seed': SEED,
    'loss_type': LOSS_TYPE,
    'ranking_weight': RANKING_WEIGHT,
    'use_lora': USE_LORA,
    'phase2': args.phase2,
    'use_multi_rbf': getattr(CFG, 'use_multi_rbf', False),
    'use_knn_gat': getattr(CFG, 'use_knn_gat', False),
    'use_edge_features': getattr(CFG, 'use_edge_features', False),
    'emb_proj_dim': CFG.emb_proj_dim,
    'emb_proj_hidden': CFG.emb_proj_hidden,
}

if not os.path.exists(os.path.join(MODEL_PATH, MODEL_NAME)):
    os.makedirs(os.path.join(MODEL_PATH, MODEL_NAME))

if not DEBUG:
    wandb.init(project='Megascale-fineTuning', config=config)
    wandb.run.name = MODEL_NAME

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


# LoRA adapter: wraps a nn.Linear with low-rank trainable residual
class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        d_in, d_out = original.in_features, original.out_features
        self.original = original
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def inject_lora(model, rank: int = 4):
    """Replace every nn.Linear inside GNN layers with LoRALinear adapters."""
    for module_name in ['GCN_layers', 'GAT_layers']:
        gnn_block = getattr(model, module_name, None)
        if gnn_block is None:
            continue
        for layer in gnn_block:
            for attr_name in dir(layer):
                sub = getattr(layer, attr_name, None)
                if isinstance(sub, nn.Linear):
                    setattr(layer, attr_name, LoRALinear(sub, rank=rank))
    return model


# Trainer class

class Trainer():
    def __init__(self, model, train_ds, val_ds, device = DEVICE):
        self.model = model.to(device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.device = device
        # Loss function (Exp 1.1: huber / Exp 1.2: ranking / combined)
        if LOSS_TYPE in ('huber', 'huber_rank'):
            self.criterion = nn.SmoothL1Loss(beta=1.0)
        else:
            self.criterion = nn.L1Loss()
        self.use_ranking = LOSS_TYPE in ('huber_rank', 'l1_rank')
        self.ranking_weight = RANKING_WEIGHT
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        if COSINE_LR:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=EPOCHS_FREEZE + EPOCHS_NO_FREEZE, eta_min=LR_MIN)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        self.model.to(self.device)
        self.mini_batch_size = MINI_BATCH_SIZE
        self.model_name = 'PEM_fine_tuned' if FREEZE_LAYERS else 'PEM_full_trained'

    def pairwise_ranking_loss(self, pred, true, margin: float = 0.5):
        """Pairwise margin ranking loss. Penalises wrong relative ordering."""
        n = len(pred)
        if n < 2:
            return torch.tensor(0.0, device=self.device)
        diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)          # [n,n]
        diff_true = true.unsqueeze(1) - true.unsqueeze(0)          # [n,n]
        sign_true = torch.sign(diff_true)
        loss = torch.clamp(margin - sign_true * diff_pred, min=0.0) # [n,n]
        mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1).bool()
        return loss[mask].mean()

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
            # Exp 1.4: LoRA — also train LoRA adapter params in GNN layers
            if USE_LORA:
                for module_name in ['GCN_layers', 'GAT_layers']:
                    gnn_block = getattr(self.model, module_name, None)
                    if gnn_block is None:
                        continue
                    for layer in gnn_block:
                        for sub in layer.modules():
                            if isinstance(sub, LoRALinear):
                                sub.lora_A.requires_grad_(True)
                                sub.lora_B.requires_grad_(True)
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
            accum_step = 0
            self.optimizer.zero_grad()
            for i, batch in enumerate(tqdm(self.train_ds, desc=f'Training Epoch: {epoch}')):
                # batch = normalize_batch(batch, True)
                if batch['delta_g'].size(1) == 1:
                    continue
                # Subsample mutations to cap per-protein compute time
                n_muts = batch['prott5'].size(1)
                if n_muts > MAX_MUTS_PER_PROTEIN:
                    idx = torch.randperm(n_muts)[:MAX_MUTS_PER_PROTEIN]
                    idx, _ = idx.sort()
                    batch = {k: (v[:, idx] if isinstance(v, torch.Tensor) and v.dim() > 1 and v.size(1) == n_muts else v) for k, v in batch.items()}
                torch.cuda.empty_cache()
                # Precompute structure features once per protein (shared across all mutations)
                coords = batch['coords'].squeeze().to(self.device)
                masks = batch['masks'].squeeze().to(self.device)
                ca_coords = coords[:, 1, :]  # [N, 3] CA atom for Phase 2 k-NN / edge features
                precomp = precompute_graph_features(coords, masks, use_multi_rbf=getattr(CFG, 'use_multi_rbf', False))
                batch_loss = 0
                batch_idx = 1
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    output,u_energy,f_energy = self.get_deltaG(batch, j, precomp=precomp, ca_coords=ca_coords)
                    delta_g = batch['delta_g'][0,j: j + self.mini_batch_size].to(self.device)
                    l1_loss = self.criterion(output, delta_g)
                    if FREEZE_LAYERS:
                        reg_loss = REG_LAMBDA * (F.mse_loss(self.model.fc1.weight,torch.zeros_like(self.model.fc1.weight)) + F.mse_loss(self.model.fc2.weight,torch.zeros_like(self.model.fc2.weight)))
                    else:
                        reg_loss = REG_LAMBDA * sum([F.mse_loss(param,torch.zeros_like(param)) for param in self.model.parameters()])
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    # Exp 1.2: pairwise ranking loss
                    if self.use_ranking and output.numel() > 1:
                        rank_loss = self.pairwise_ranking_loss(output, delta_g)
                        loss = l1_loss + reg_loss + energy_reg + self.ranking_weight * rank_loss
                    else:
                        rank_loss = torch.tensor(0.0, device=self.device)
                        loss = l1_loss + reg_loss + energy_reg
                    (loss / GRAD_ACCUM).backward()
                    accum_step += 1
                    if accum_step % GRAD_ACCUM == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    train_pc_corr = torch.corrcoef(torch.cat((output[None,:],delta_g[None,:])))[0, 1]
                    batch_loss += loss.item()
                    wandb_step += 1
                    wandb_log({'loss': loss.item(), 'epoch': epoch, 'batch': i,'l1_loss': l1_loss.item(), 'reg_loss': reg_loss.item(),
                               'energy_reg': energy_reg.item(), 'rank_loss': rank_loss.item(),
                               'wandb_step': wandb_step,
                               'train_pc_corr': train_pc_corr},run)
                
                    
            # save the model
            if not DEBUG:
                print(f"Saving model in path {os.path.join(MODEL_PATH, MODEL_NAME, f'epoch_{epoch}.pt')}")
                # save the model
                torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME, f'epoch_{epoch}.pt'))
            
            # flush any remaining accumulated gradients at epoch end
            if accum_step % GRAD_ACCUM != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            pc_corr, val_loss, _ = self.validate(epoch,run)
            self.model.train()
            # update the learning rate
            if COSINE_LR:
                self.scheduler.step()
            else:
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
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_ds,desc=f'Validation Epoch: {epoch}')):
                # batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                coords = batch['coords'].squeeze().to(self.device)
                masks = batch['masks'].squeeze().to(self.device)
                ca_coords = coords[:, 1, :]
                val_precomp = precompute_graph_features(coords, masks, use_multi_rbf=getattr(CFG, 'use_multi_rbf', False))
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    batch_idx += 1
                    output,u_energy,f_energy = self.get_deltaG(batch, j, precomp=val_precomp, ca_coords=ca_coords)
                    delta_g = batch['delta_g'][0,j: j + self.mini_batch_size].to(self.device)
                    loss = self.criterion(output,delta_g)
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    loss += energy_reg
                    batch_loss += loss.item()
                    val_dg = torch.cat((val_dg, delta_g), dim=0)
                    val_dg_pred = torch.cat((val_dg_pred, output), dim=0)
                    batch_df = pd.DataFrame([],columns=['protein','deltaG','pred_deltaG'])
                    batch_df['deltaG'] = delta_g.cpu().numpy()
                    batch_df['pred_deltaG']  = output.cpu().numpy()
                    batch_df['protein'] = [batch['name'][0] for i in range(len(delta_g))]
                    val_df = pd.concat([val_df,batch_df])
                    # clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
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
        ddg_rmse = float(torch.sqrt(torch.tensor(sum((a-b)**2 for a,b in zip(ddg_true,ddg_pred))/len(ddg_true)))) if len(ddg_true) > 1 else float('nan')
        # Pearson for ddG
        try:
            from scipy.stats import pearsonr
            ddg_pearson_corr, _ = pearsonr(ddg_true, ddg_pred)
        except ImportError:
            ddg_pearson_corr = float('nan')
        print(f'  Epoch {epoch} | PCC(dG)={pc_corr:.4f} | PCC(ddG)={ddg_pearson_corr:.4f} | RMSE={rmse:.4f}')
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
        return pc_corr, val_loss, val_df
        
    def get_deltaG(self, batch, i, precomp=None, ca_coords=None):
        one_hot_minibatch = batch['one_hot'][0,i: i + self.mini_batch_size].to(self.device)
        prott5_embedding_minibatch = batch['prott5'][0,i: i + self.mini_batch_size].to(self.device)
        n = prott5_embedding_minibatch.size(0)

        if precomp is not None:
            # Use precomputed structure features — avoids recomputing distance matrix per mutation
            D_f, Fb_f, D_u, Fb_u = precomp
            folded_graph_minibatch = torch.stack(
                [get_graph_fast(one_hot_minibatch[j].squeeze(), prott5_embedding_minibatch[j].squeeze(), D_f, Fb_f) for j in range(n)])
            unfolded_graph_minibatch = torch.stack(
                [get_graph_fast(one_hot_minibatch[j].squeeze(), prott5_embedding_minibatch[j].squeeze(), D_u, Fb_u) for j in range(n)])
        else:
            batch['coords'] = batch['coords'].to(self.device)
            batch['masks'] = batch['masks'].to(self.device)
            folded_graph_minibatch = torch.stack(
                [get_graph(batch['coords'].squeeze(), one_hot_minibatch[j].squeeze(), prott5_embedding_minibatch[j].squeeze(), batch['masks'].squeeze()) for j in range(n)])
            unfolded_graph_minibatch = torch.stack(
                [get_unfolded_graph(batch['coords'].squeeze(), one_hot_minibatch[j].squeeze(), prott5_embedding_minibatch[j].squeeze(), batch['masks'].squeeze()) for j in range(n)])

        all_graph_minibatch = torch.cat([folded_graph_minibatch, unfolded_graph_minibatch], dim=0)
        # Wire ca_coords for Phase 2: expand [N,3] -> [2n, N, 3] to match batch dim
        model_ca_coords = None
        if ca_coords is not None:
            ca_exp = ca_coords.unsqueeze(0).expand(n, -1, -1)  # [n, N, 3]
            model_ca_coords = torch.cat([ca_exp, ca_exp], dim=0)  # [2n, N, 3]
        minibatch_energy = self.model(all_graph_minibatch, ca_coords=model_ca_coords)
        folded_energy = minibatch_energy[:minibatch_energy.size(0) // 2]
        unfolded_energy = minibatch_energy[minibatch_energy.size(0) // 2:]
        return unfolded_energy - folded_energy, unfolded_energy, folded_energy


def run_training():
    """Run the training for all the proteins"""
    train_ds = MSDataset(tensor_root_dir=tensor_root_dir,
                                          mutations_root_dir=mutations_root_dir, train=True, emb_type=EMB_TYPE)

    test_ds = MSDataset(tensor_root_dir=tensor_root_dir,
                                            mutations_root_dir=mutations_root_dir, train=False, emb_type=EMB_TYPE)
    
     # Create the dataloaders
    train_ds = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=1, shuffle=True)

    # Create the model
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate,
                light_attention=LIGHT_ATTENTION, emb_projection=EMB_PROJ).to(DEVICE)
    if PRETRAINED:
        try:
            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
        except:
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))

    # Exp 1.4: inject LoRA adapters into GNN layers before training
    if USE_LORA:
        model = inject_lora(model, rank=4)
        print("LoRA adapters injected into GNN layers (rank=4)")

    # Train the model
    trainer = Trainer(model, train_ds, test_ds)
    model, pc_corr = trainer.train(epochs=EPOCHS_FREEZE)
    
    # Unfreeze the layers and train the model with lower learning rate
    global FREEZE_LAYERS, LR
    FREEZE_LAYERS = False
    LR = 1e-5
    
    print('Training the whole model with lower learning rate')
    trainer = Trainer(model, train_ds, test_ds)
    model, pc_corr = trainer.train(epochs=EPOCHS_NO_FREEZE, s_epoch=EPOCHS_FREEZE)
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
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate,
                light_attention=LIGHT_ATTENTION, emb_projection=EMB_PROJ).to(DEVICE)
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
