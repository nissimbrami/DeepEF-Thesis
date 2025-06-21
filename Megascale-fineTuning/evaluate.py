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
from train_utils import get_graph, get_unfolded_graph, load_checkpoint
import wandb
from tqdm import tqdm
from sklearn.model_selection import KFold
import gc
import pandas as pd
# parser
import argparse

parser = argparse.ArgumentParser(description='Train the model with the mega-scale data')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--model_name', type=str, default='PEM_fine_tuned', help='Model name')
parser.add_argument('--dataset_type', type=str, default='pnas', help='Dataset type')
parser.add_argument('--unstable_mut', action='store_true', help='Save the unstable mutations')
parser.add_argument('--one_mut', action='store_true', help='Remove the multiple mutations when fine-tuning')
parser.add_argument('--freeze_layers',action = 'store_true', help ='Freeze model layers except mlp and LA')
parser.add_argument('--trained_model_path',type=str,default = "./res/trianed_models-light_attention/43_final_model.pt",help='Trained model path')
parser.add_argument('--dg_ml', action='store_true', help='Change deltaG threshold to [-1,5]')

args = parser.parse_args()

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
EPOCHS = 30 if not DEBUG else 1
FREEZE_LAYERS = args.freeze_layers
CRITERION = "L1"
MODEL_PATH = './Megascale-fineTuning/models'
MINI_BATCH_SIZE = 64
DEVICE = 'cuda'# if torch.cuda.is_available() else 'cpu'
TRAINED_MODEL_PATH = args.trained_model_path
BASE_MODEL_NAME = TRAINED_MODEL_PATH.split('/')[-2]
MODEL_NAME = args.model_name
PRETRAINED = True
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
    'ds_type': DS_TYPE,
    'one_mutation': ONE_MUT,
    'dG_ml': DG_ML
}

if not os.path.exists(os.path.join(MODEL_PATH, MODEL_NAME)):
    os.makedirs(os.path.join(MODEL_PATH, MODEL_NAME))

if not DEBUG:
    wandb.init(project='Megascale-fineTuning-evaluation', config=config)
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
        coords_tensor = torch.load(os.path.join(protein_dir, COORDS),weights_only=True)
        delta_g_tensor = torch.load(os.path.join(protein_dir, DELTA_G),weights_only=True)
        mask_tensor = torch.load(os.path.join(protein_dir, MASKS),weights_only=True)
        one_hot_tensor = torch.load(os.path.join(protein_dir, ONE_HOT),weights_only=True)
        embedding_tensor = self.load_embedding_tensor(os.path.join(protein_dir, PROTT5_EMBEDDINGS))
        
        # Add ddg column
        mutations['dg'] = delta_g_tensor.cpu().numpy()
        mutations['ddg'] = mutations['dg'] - mutations['dg'].iloc[0]
        
        # If dG_ml is check save the threshold of -1 and 5
        if self.dG_ml:
            threshold = [-1.0,5.0]
            delta_g_tensor = torch.where(delta_g_tensor > threshold[0], delta_g_tensor, threshold[0])
            delta_g_tensor = torch.where(delta_g_tensor < threshold[1], delta_g_tensor, threshold[1])
        
        indexes = set(mutations.index)
        # remove unstable mut
        if not self.unstable_mut:
            indexes -= set(mutations[mutations['ddG_ML'] == '-'].index)
                 
        # remove the mutations with more than one mutation
        if self.one_mut:
            indexes -= set(mutations[mutations['mut_type'].str.contains(':')].index)
        
        if self.ds_type in ('pnas','deepef1') :
            # get the pnas mutations indexes
            indexes -= set(mutations[~mutations['name'].isin(self.test_mutations['name'])].index)
        
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
            'masks': mask_tensor,
            'ddg': torch.tensor(mutations['ddg'].values)
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)
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
                batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                for j in range(0, batch['prott5'].size(1), self.mini_batch_size):
                    self.optimizer.zero_grad()
                    output,u_energy,f_energy = self.get_deltaG(batch, j)
                    delta_g = batch['delta_g'][0,j: j + self.mini_batch_size].to(self.device)
                    l1_loss = self.criterion(output, delta_g)
                    if FREEZE_LAYERS:
                        reg_loss = REG_LAMBDA * (F.mse_loss(self.model.fc1.weight,torch.zeros_like(self.model.fc1.weight)) + F.mse_loss(self.model.fc2.weight,torch.zeros_like(self.model.fc2.weight)))
                    else:
                        reg_loss = REG_LAMBDA * sum([F.mse_loss(param,torch.zeros_like(param)) for param in self.model.parameters()])
                    energys = torch.cat((u_energy,f_energy),dim=0)
                    energy_reg = E_REG_LAMBDA * (F.mse_loss(energys,torch.zeros_like(energys)))
                    loss = l1_loss + reg_loss +energy_reg
                    loss.backward()
                    self.optimizer.step()
                    train_pc_corr = torch.corrcoef(torch.cat((output[None,:],delta_g[None,:])))[0, 1]
                    batch_loss += loss.item()
                    wandb_step += 1
                    wandb_log({'loss': loss.item(), 'epoch': epoch, 'batch': i,'l1_loss': l1_loss.item(), 'reg_loss': reg_loss.item(), 
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
        val_ddg = torch.tensor([],device=self.device)
        val_ddg_pred = torch.tensor([],device=self.device)
        val_df = pd.DataFrame([],columns=['protein','deltaG','pred_deltaG','ddG','pred_ddG'])
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_ds,desc=f'Validation Epoch: {epoch}')):
                batch = normalize_batch(batch, True)
                batch_loss = 0
                batch_idx = 1
                protein_df = pd.DataFrame([],columns=['protein','deltaG','pred_deltaG','ddG','pred_ddG'])
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
                    batch_df = pd.DataFrame([],columns=['protein','deltaG','pred_deltaG','ddG','pred_ddG'])
                    batch_df['deltaG'] = delta_g.cpu().numpy()
                    batch_df['pred_deltaG']  = output.cpu().numpy()
                    batch_df['protein'] = [batch['name'][0] for i in range(len(delta_g))]
                    protein_df = pd.concat([protein_df,batch_df])
                    # clear memory
                    torch.cuda.empty_cache()
                    gc.collect()
                batch_loss /= batch_idx
                protein_df['ddG'] = protein_df['deltaG'] - protein_df['deltaG'].iloc[0]
                protein_df['pred_ddG'] = protein_df['pred_deltaG'] - protein_df['pred_deltaG'].iloc[0]
                val_df = pd.concat([val_df, protein_df])
            val_loss += batch_loss
        val_loss /= len(self.val_ds)
        print(f'Validation Loss: {val_loss}')
        pc_corr = torch.corrcoef(torch.cat((val_dg[None,:],val_dg_pred[None,:])))[0, 1]
        if not test:
            wandb_log({'val_loss': val_loss,'epoch': epoch, 'val_pc_corr': pc_corr},run)
        else:
            wandb_log({'test_loss': val_loss,'epoch': epoch, 'pc_corr': pc_corr},run)
        
        # Save datafeame
        # val_df.to_csv("val_df.csv",index=False)
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

        minibatch_energy = self.model(all_graph_minibatch)
        folded_energy = minibatch_energy[:minibatch_energy.size(0) // 2]
        unfolded_energy = minibatch_energy[minibatch_energy.size(0) // 2:]
        
        return unfolded_energy - folded_energy,unfolded_energy,folded_energy


def run_training():
    """Run the training for all the proteins"""
    train_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                          mutations_root_dir=mutations_root_dir, train=True)
    
    test_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                            mutations_root_dir=mutations_root_dir, train=False)
    
     # Create the dataloaders
    train_ds = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=1, shuffle=True)

    # Create the model
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate,
                light_attention=LIGHT_ATTENTION).to(DEVICE)
    if PRETRAINED: 
        try:
            model, _, _, _, _ = load_checkpoint(TRAINED_MODEL_PATH, model)
        except:
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
    
    # Train the model
    trainer = Trainer(model, train_ds, test_ds)
    model, pc_corr = trainer.train(epochs=EPOCHS)
    
    # Unfreeze the layers and train the model with lower learning rate
    global FREEZE_LAYERS, LR
    FREEZE_LAYERS = False
    LR = 1e-5
    
    print('Training the whole model with lower learning rate')
    trainer = Trainer(model, train_ds, test_ds)
    model, pc_corr = trainer.train(epochs=EPOCHS, s_epoch=EPOCHS)
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
    train_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                          mutations_root_dir=mutations_root_dir, train=True)
    
    test_ds = AllProteinValidationDataset(tensor_root_dir=tensor_root_dir,
                                            mutations_root_dir=mutations_root_dir, train=False)
    
     # Create the dataloaders
    train_ds = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=1, shuffle=True)
    # Create the model
    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef, dropout_rate=CFG.dropout_rate,
                light_attention=LIGHT_ATTENTION).to(DEVICE)
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
    tensor_root_dir = r'./data/Processed_K50_dG_datasets/training_data'
    mutations_root_dir = r'./data/Processed_K50_dG_datasets/mutation_datasets'
    CFG.dropout_rate = DROP_OUT
    # run_training()
    run_validation_metrics()
