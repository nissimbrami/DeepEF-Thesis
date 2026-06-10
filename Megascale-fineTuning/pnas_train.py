# fine tuning the existing model with mega-scale data
# Path: Megascale-fineTuning/pnas_train.py

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
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--model_name', type=str, default='PEM_fine_tuned', help='Model name')
parser.add_argument('--dataset_type', type=str, default='pnas', help='Dataset type')
parser.add_argument('--unstable_mut', action='store_true', help='Save the unstable mutations')
parser.add_argument('--one_mut', action='store_true', help='Remove the multiple mutations when fine-tuning')
parser.add_argument('--freeze_layers',action = 'store_true', help ='Freeze model layers except mlp and LA')
parser.add_argument('--trained_model_path',type=str,default = "./res/trianed_models-light_attention/43_final_model.pt",help='Trained model path')
parser.add_argument('--dg_ml', action='store_true', help='Change deltaG threshold to [-1,5]')
# Stage 1: Training improvements
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--cosine_lr', action='store_true', help='Use cosine annealing LR schedule')
parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum LR for cosine schedule')
parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for Adam optimizer')
# Ablation best config: Huber + Ranking loss
parser.add_argument('--use_huber_loss', action='store_true', help='Use Huber loss instead of L1')
parser.add_argument('--huber_delta', type=float, default=1.0, help='Delta for Huber loss')
parser.add_argument('--use_ranking_loss', action='store_true', help='Add ranking (margin) loss')
parser.add_argument('--ranking_lambda', type=float, default=0.1, help='Weight for ranking loss')
parser.add_argument('--ranking_margin', type=float, default=0.1, help='Margin for ranking loss')
# Ablation best config: k-NN GAT
parser.add_argument('--use_knn', action='store_true', help='Use k-NN graph for GAT instead of fully connected')
parser.add_argument('--knn_k', type=int, default=30, help='k for k-NN graph construction')
# From-scratch training (no pretrained model)
parser.add_argument('--from_scratch', action='store_true', help='Train from scratch without pretrained model')
# Embedding type selection
parser.add_argument('--emb_type', type=str, default='prott5',
                    choices=['prott5', 'esmif_enc', 'dual_esmif', 'saprot', 'saprot_pm', 'dual_saprot_pm'],
                    help='Embedding type: prott5 (1024), esmif_enc (512), dual_esmif (1536), saprot (1280), saprot_pm (1280), dual_saprot_pm (2304)')
# New unified loss/config flags
parser.add_argument('--loss_type', type=str, default='l1',
                    choices=['l1', 'huber', 'huber_rank'],
                    help='Loss function type')
parser.add_argument('--ranking_weight', type=float, default=0.1, help='Weight for ranking loss component')
parser.add_argument('--use_knn_gat', action='store_true', help='Use k-NN GAT (k=30)')
parser.add_argument('--no_pretrained', action='store_true', help='Train from scratch (alias for --from_scratch)')
# Epoch control
parser.add_argument('--epochs_freeze', type=int, default=None, help='Epochs with frozen backbone (overrides --epochs)')
parser.add_argument('--epochs_unfreeze', type=int, default=None, help='Epochs with unfrozen backbone')
# Hardware control
parser.add_argument('--mini_batch_size', type=int, default=64, help='Mini-batch size for mutations within each protein')
parser.add_argument('--emb_projection', type=str, default='none', choices=['none', 'mlp', 'low_rank'],
                    help='Embedding projection mode: none (raw concat), mlp (project to 16-dim), low_rank')

args = parser.parse_args()

# Handle --no_pretrained as alias for --from_scratch
if args.no_pretrained:
    args.from_scratch = True

# Handle --loss_type shortcuts
if args.loss_type == 'huber':
    args.use_huber_loss = True
elif args.loss_type == 'huber_rank':
    args.use_huber_loss = True
    args.use_ranking_loss = True
    args.ranking_lambda = args.ranking_weight

# Handle --use_knn_gat shortcut
if args.use_knn_gat:
    args.use_knn = True
    args.knn_k = 30

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# Set embedding type in new_dataset module BEFORE any dataset is created
import new_dataset
new_dataset.EMB_TYPE = args.emb_type

# Set emb_input_dim based on embedding type
EMB_DIMS = {
    'prott5': 1024, 'esmif_enc': 512, 'dual_esmif': 1536,
    'saprot': 1280, 'saprot_pm': 1280, 'dual_saprot_pm': 2304,
}
CFG.emb_input_dim = EMB_DIMS.get(args.emb_type, 1024)
CFG.emb_projection = args.emb_projection
print(f"Embedding type: {args.emb_type} (dim={CFG.emb_input_dim}), projection={args.emb_projection}")

# Constants
COORDS = 'coords_tensor.pt'
DELTA_G = 'deltaG.pt'
MASKS = 'mask_tensor.pt'
ONE_HOT = 'one_hot_encodings.pt'
PROTT5_EMBEDDINGS = 'prott5_embeddings'
VAL_RATIO = 0.2
RANDOM_SEED = args.seed
NANO_TO_ANGSTROM = 0.1
DEBUG  = args.debug
EPOCHS_FREEZE = 20 if not DEBUG else 1
EPOCHS_NO_FREEZE = 60 if not DEBUG else 1
FREEZE_LAYERS = args.freeze_layers
CRITERION = "L1"
MODEL_PATH = './Megascale-fineTuning/models'
MINI_BATCH_SIZE = args.mini_batch_size
DEVICE = 'cuda'# if torch.cuda.is_available() else 'cpu'
TRAINED_MODEL_PATH = args.trained_model_path
BASE_MODEL_NAME = TRAINED_MODEL_PATH.split('/')[-2]
MODEL_NAME = args.model_name
PRETRAINED = not args.from_scratch
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
# Stage 1 training improvements
COSINE_LR = args.cosine_lr
LR_MIN = args.lr_min
GRAD_ACCUM = args.grad_accum
WEIGHT_DECAY = args.weight_decay
# Ablation best: Huber + Ranking
USE_HUBER = args.use_huber_loss
HUBER_DELTA = args.huber_delta
USE_RANKING = args.use_ranking_loss
RANKING_LAMBDA = args.ranking_lambda
RANKING_MARGIN = args.ranking_margin
# Ablation best: k-NN GAT
USE_KNN = args.use_knn
KNN_K = args.knn_k

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
    'epochs': args.epochs,
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
    # Stage 1
    'seed': args.seed,
    'cosine_lr': COSINE_LR,
    'lr_min': LR_MIN,
    'grad_accum': GRAD_ACCUM,
    'weight_decay': WEIGHT_DECAY,
    'use_huber_loss': USE_HUBER,
    'huber_delta': HUBER_DELTA,
    'use_ranking_loss': USE_RANKING,
    'ranking_lambda': RANKING_LAMBDA,
    'ranking_margin': RANKING_MARGIN,
    'use_knn': USE_KNN,
    'knn_k': KNN_K,
    'from_scratch': args.from_scratch,
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


def ranking_loss(pred, target, margin=RANKING_MARGIN):
    """Pairwise ranking loss: if target_i > target_j, then pred_i should be > pred_j by margin.

    Samples random pairs from the mini-batch for efficiency.
    """
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    # Sample pairs (at most 128 pairs for efficiency)
    n_pairs = min(n * (n - 1) // 2, 128)
    idx_i = torch.randint(0, n, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=pred.device)
    # Ensure different indices
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % n

    # Compute pairwise differences
    pred_diff = pred[idx_i] - pred[idx_j]
    target_diff = target[idx_i] - target[idx_j]

    # Sign: +1 if target_i > target_j, -1 otherwise
    sign = torch.sign(target_diff)
    # Margin ranking loss: max(0, -sign * pred_diff + margin)
    loss = torch.clamp(-sign * pred_diff + margin, min=0.0)
    return loss.mean()


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
        # Loss function: Huber (from ablation) or L1 (original)
        if USE_HUBER:
            self.criterion = nn.HuberLoss(delta=HUBER_DELTA)
        else:
            self.criterion = nn.L1Loss()
        # Optimizer with weight decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # LR scheduler: Cosine Annealing or ReduceLROnPlateau
        if COSINE_LR:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=LR_MIN)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        self.model.to(self.device)
        self.mini_batch_size = MINI_BATCH_SIZE
        self.model_name = 'PEM_fine_tuned' if FREEZE_LAYERS else 'PEM_full_trained'
        self.grad_accum_steps = GRAD_ACCUM

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
        best_pc_corr = -float('inf')

        for epoch in range(s_epoch, s_epoch + epochs):
            self.model.train()
            accum_count = 0
            self.optimizer.zero_grad()

            for i, batch in enumerate(tqdm(self.train_ds, desc=f'Training Epoch: {epoch}')):
                # batch = normalize_batch(batch, True)
                if batch['delta_g'].size(1) == 1:
                    continue
                batch_loss = 0
                batch_idx = 1
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0,j: j + self.mini_batch_size].to(self.device)

                    # Primary loss (Huber or L1)
                    primary_loss = self.criterion(output, delta_g)

                    # Regularization losses
                    if FREEZE_LAYERS:
                        reg_loss = REG_LAMBDA * (F.mse_loss(self.model.fc1.weight,torch.zeros_like(self.model.fc1.weight)) + F.mse_loss(self.model.fc2.weight,torch.zeros_like(self.model.fc2.weight)))
                    else:
                        reg_loss = REG_LAMBDA * sum([F.mse_loss(param,torch.zeros_like(param)) for param in self.model.parameters()])
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))

                    # Ranking loss (from ablation best config)
                    rank_loss = torch.tensor(0.0, device=self.device)
                    if USE_RANKING:
                        rank_loss = RANKING_LAMBDA * ranking_loss(output, delta_g, margin=RANKING_MARGIN)

                    loss = primary_loss + reg_loss + energy_reg + rank_loss

                    # Scale loss for gradient accumulation
                    scaled_loss = loss / self.grad_accum_steps
                    scaled_loss.backward()

                    accum_count += 1

                    # Step optimizer after accumulation
                    if accum_count % self.grad_accum_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=CFG.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    train_pc_corr = torch.corrcoef(torch.cat((output[None,:],delta_g[None,:])))[0, 1]
                    batch_loss += loss.item()
                    wandb_step += 1
                    log_dict = {
                        'loss': loss.item(), 'epoch': epoch, 'batch': i,
                        'primary_loss': primary_loss.item(), 'reg_loss': reg_loss.item(),
                        'energy_reg': energy_reg.item(), 'wandb_step': wandb_step,
                        'train_pc_corr': train_pc_corr
                    }
                    if USE_RANKING:
                        log_dict['ranking_loss'] = rank_loss.item()
                    wandb_log(log_dict, run)

            # Handle remaining accumulated gradients at end of epoch
            if accum_count % self.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=CFG.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # save the model
            if not DEBUG:
                print(f"Saving model in path {os.path.join(MODEL_PATH, MODEL_NAME, f'epoch_{epoch}.pt')}")
                # save the model
                torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME, f'epoch_{epoch}.pt'))

            pc_corr, val_loss, _ = self.validate(epoch,run)
            self.model.train()

            # Track best model
            if pc_corr > best_pc_corr:
                best_pc_corr = pc_corr
                if not DEBUG:
                    torch.save(self.model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME, 'best_model.pt'))
                    print(f"  New best PCC: {best_pc_corr:.4f}")

            # Update learning rate
            if COSINE_LR:
                self.scheduler.step()
            else:
                self.scheduler.step(pc_corr)
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb_log({'epoch': epoch, 'lr': current_lr, 'best_pc_corr': best_pc_corr}, run)

        print(f"\nTraining complete. Best validation PCC: {best_pc_corr:.4f}")
        return self.model, best_pc_corr

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
        print(f'  Epoch {epoch} | PCC: {pc_corr:.4f} | Spearman: {sp_corr:.4f} | RMSE: {rmse:.4f} | ddG PCC: {ddg_pearson_corr:.4f}')
        return pc_corr, val_loss, val_df

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

        # k-NN GAT: compute CA coordinates for distance-based edges
        ca_coords = None
        if USE_KNN:
            # CA is atom index 1 (N=0, CA=1, C=2, CB=3)
            coords_squeezed = batch['coords'].squeeze()  # [seq_len, 4, 3]
            ca_pos = coords_squeezed[:, 1, :]  # [seq_len, 3]
            # Compute k-NN based cutoff: find the distance that includes k nearest neighbors
            dists = torch.cdist(ca_pos.unsqueeze(0), ca_pos.unsqueeze(0)).squeeze(0)  # [N, N]
            # For each residue, get the k-th nearest neighbor distance
            k = min(KNN_K, dists.size(0) - 1)
            kth_dist, _ = dists.topk(k + 1, dim=1, largest=False)  # +1 because self-distance=0
            cutoff = kth_dist[:, -1].max().item()  # use max k-th distance as cutoff
            # Expand ca_coords for the full batch (folded + unfolded)
            batch_size = all_graph_minibatch.size(0)
            ca_coords = ca_pos.unsqueeze(0).expand(batch_size, -1, -1)
            # Temporarily set model's gat_cutoff
            self.model.gat_cutoff = cutoff

        minibatch_energy = self.model(all_graph_minibatch, ca_coords=ca_coords)
        folded_energy = minibatch_energy[:minibatch_energy.size(0) // 2]
        unfolded_energy = minibatch_energy[minibatch_energy.size(0) // 2:]

        return unfolded_energy - folded_energy,unfolded_energy,folded_energy


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
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate,
                light_attention=LIGHT_ATTENTION, emb_projection=args.emb_projection).to(DEVICE)
    if PRETRAINED:
        try:
            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
        except:
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))

    # Single-stage training with specified epochs
    epochs = args.epochs
    trainer = Trainer(model, train_ds, test_ds)
    model, pc_corr = trainer.train(epochs=epochs)

    if not DEBUG:
        wandb.finish()

    print(f'Training completed with Best Pearson Correlation: {pc_corr:.4f}')


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
                light_attention=LIGHT_ATTENTION, emb_projection=args.emb_projection).to(DEVICE)
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
    # Ensure new_dataset module picks up EMB_TYPE (guard against stale cache)
    import new_dataset as _nd
    print(f"[DEBUG] new_dataset loaded from: {_nd.__file__}")
    print(f"[DEBUG] EMB_TYPE={_nd.EMB_TYPE}, DS_TYPE={_nd.DS_TYPE}")
    _nd.EMB_TYPE = args.emb_type
    _nd.DS_TYPE = DS_TYPE
    _nd.DG_ML = DG_ML
    _nd.ONE_MUT = ONE_MUT
    _nd.UNSTABLE_MUT = UNSTABLE_MUT
    run_training()
    # run_validation_metrics()
