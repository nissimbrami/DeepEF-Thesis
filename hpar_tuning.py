#### Hyper prameter tuning for the model using optuna

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
import pandas as pd
import numpy as np
import wandb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from validation.validation import run_validation

CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG.debug = False

def objective(trial):
    CFG.data_path = './data/casp12_data_30/'
    CFG.model_path = './res/trianed_models-hyper_'+str(trial.number)+'/'
    if CFG.debug:
        CFG.model_path = './res/trianed_models-hyper-debug/'
    # Define the hyperparameters to optimize
    CFG.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    CFG.reg_alpha = trial.suggest_float("reg_alpha", 1e-1, 1, log=True)
    CFG.gaussian_coef = -1*trial.suggest_float("gaussian_coef", 1e-5, 1e-2, log=True)
    CFG.num_layers = trial.suggest_int("num_layers", 1, 4) if not CFG.debug else 1
    CFG.clip_grad_norm = trial.suggest_categorical("clip_grad_norm", [True, False])
    CFG.max_grad_norm = trial.suggest_float("max_grad_norm", 1e-2, 1, log=True)
    CFG.dropout_rate = trial.suggest_float("dropout_rate", 0, 0.8)
    CFG.num_epochs = 5 if not CFG.debug else 1
    CFG.precision = torch.float32
    
    # Set the wandb project
    wandb.init(project="PEM-Hyperparameter-Tuning")
    
    # Define the data loader
    d_params = data_params(num_workers =CFG.num_workers, batch_size=CFG.batch_size,cuda=CFG.cuda,constraint=CFG.constraint, 
                           debug=CFG.debug,dataset='scn',LLM_EMB=True)
    train_loader, valid_loader,test_loader = fetch_dataloader(data_dir=CFG.data_path, params=d_params)
    # Define the model
    model = PEM(layers=CFG.num_layers,gaussian_coef=CFG.gaussian_coef,dropout_rate=CFG.dropout_rate).to(CFG.device)
    model.name = "PEM-With LLM embedding"
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    # Define the learning rate scheduler based on loss
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    # configurate wandb
    wandb_config(wandb, model, optimizer, scheduler, train_loader,
                 CFG.model_path,CFG.reg_alpha,CFG.gaussian_coef,CFG.lr,
                 CFG.num_layers,CFG.dropout_rate,CFG.precision)
    # Run training
     # setup half precision training
    scaler = torch.cuda.amp.GradScaler()
    valid_loss = 100
    for epoch in (range(CFG.num_epochs)):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        model,epoch_train_loss,valid_loss = train_one_epoch(model, optimizer, train_loader, CFG.device,epoch,valid_loader,valid_loss, scheduler,scaler)
        if np.isnan(epoch_train_loss).any():
            break
        print(f'epoch: {epoch}, valid_loss: {valid_loss}')
    # Get correlation of the model
    if np.isnan(epoch_train_loss).any():
        return 0
    print('running megascale validation for model: ', CFG.model_path)
    run_validation(r'./data/Processed_K50_dG_datasets', mode='evaluation', model_path=CFG.model_path+'best_model.pt',model=model, debug = CFG.debug)
    print('getting validation results for model: ', CFG.model_path)
    results = get_reults('./data/Processed_K50_dG_datasets/', 'mutation_datasets/', CFG.model_path+'best_model.pt')
    pcc = results[['inferred_dG', 'deltaG']].corr(method='pearson').iloc[0,1]
    spc = results[['inferred_dG', 'deltaG']].corr(method='spearman').iloc[0,1]
    rmse = np.sqrt(((results['inferred_dG'] - results['deltaG'])**2).mean())
    wandb.log({"pcc": pcc, "spc": spc, "rmse": rmse})
    return pcc
        
# define validation function
def validation(model, dataloader, device,epoch,N,optimizer,val_type = 'robust'):
    """
    Validation function for the model.
    """
    valid_loss = 0
    valid_lossd = 0
    valid_lossg = 0
    valid_lossc = 0
    n_skips = 0
    model.eval() # cant use eval because of the loss function calculation
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Validation: Epoch {epoch}")
        for index, data in (enumerate(tepoch)):
            # Clean the GPU cache
            if(device.type == "cuda" or device.type == "mps"):    
                torch.cuda.empty_cache()
            gc.collect()
            # zero the parameter gradients
            optimizer.zero_grad()
            Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2 = get_noised_proteins(data,device,CFG)
            X = torch.cat((Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2),dim=0)
            
            with torch.no_grad():
                # half precision validation
                with torch.amp.autocast(device_type="cuda", dtype=CFG.precision):
                    # calculate the energy for the folded unfolded and decoy structure
                    E = model(X)
                    Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2 = E[0], E[1], E[2], E[3], E[4], E[5], E[6]
                    # calculate the loss   
                    loss ,lossd, lossg,lossc = criterion(Ejf, Eju, Exd, Xjf, Ecd, Exdu, Ecy1, Ecy2, with_grad = False)
                
            # Add gradient penalty
            Ejf_grad = torch.tensor(0.0).to(device)
            if CFG.gradient_penalty:
                # zero the parameter gradients
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect( )
                # half precision training
                with torch.amp.autocast(device_type="cuda", dtype=CFG.precision):
                    # calculate the energy for the wild type
                    Xjf.requires_grad = True
                    Ejf_grad = model(Xjf)[0]
                    lossg = gradient_penalty(Xjf, Ejf_grad)
                    
                loss += lossg # add the gradient penalty to the loss
            
            valid_loss += loss.item() 
            valid_lossd += lossd.item()
            valid_lossg += lossg.item()
            valid_lossc += lossc.item()
            
            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            if index % 1000 == 999:
                print(f"Validation loss: {round(valid_loss/(index + 1),2)}, index: {index}, n_skips: {n_skips}")
            
    return valid_loss/len(dataloader),valid_lossd/len(dataloader),valid_lossg/len(dataloader),valid_lossc/len(dataloader)

def train_one_epoch(model, optimizer, dataloader, device,epoch,valid_loader,best_val=1000,scheduler=None,scaler=None):
    """
    Training function for the model.
    
    """
    model.train()
    epoch_train_loss = []
    model.train()
    running_loss = 0.0
    n_skips = 0
    ds_length = len(dataloader)
    with tqdm(dataloader, unit="batch") as tepoch:
        # set progress bar description
        tepoch.set_description(f"Epoch {epoch}")
        for index, data in enumerate(tepoch):
            # Clean the GPU cache
            torch.cuda.empty_cache()
            gc.collect()
             # zero the parameter gradients
            optimizer.zero_grad()
            Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2 = get_noised_proteins(data,device,CFG)
            if Xjf is None:
                n_skips += 1
                continue
            X = torch.cat((Xjf,Xju,Xd,Xcd,Xdu,Xcy1,Xcy2),dim=0)
            
            # half precision training
            with torch.amp.autocast(device_type="cuda", dtype=CFG.precision):
                # calculate the energy for the folded unfolded and decoy structure
                E = model(X)
                Ejf, Eju, Exd, Ecd, Exdu, Ecy1, Ecy2 = E[0], E[1], E[2], E[3], E[4], E[5], E[6]
                # calculate the loss   
                loss ,lossd, lossg,lossc = criterion(Ejf, Eju, Exd, Xjf, Ecd, Exdu, Ecy1, Ecy2, with_grad = False)
            
            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()

            # Clip gradients to a maximum norm of max_grad_norm to prevent exploding gradients
            if CFG.clip_grad_norm:
                clip_grad_norm(model.parameters(), CFG.max_grad_norm)
            
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            # Add gradient penalty
            Ejf_grad = torch.tensor(0.0).to(device)
            if CFG.gradient_penalty:
                # zero the parameter gradients
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect( )
                # half precision training
                with torch.amp.autocast(device_type="cuda", dtype=CFG.precision):
                    # calculate the energy for the wild type
                    Xjf.requires_grad = True
                    Ejf_grad = model(Xjf)[0]
                    lossg = gradient_penalty(Xjf, Ejf_grad)
                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(lossg).backward()

                # Unscales gradients and calls
                # or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
                loss += lossg # add the gradient penalty to the loss
                
            # print statistics
            running_loss += loss.item()
            if index % 1000 == 999 :    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 1000:.3f}')
                print(f"skipped {n_skips}")
                epoch_train_loss.append(running_loss/1000)
                if not CFG.debug:
                    wandb.log({"epoch": epoch,"running_loss": running_loss/1000,"running_lossIndex":ds_length*epoch+index})
                running_loss = 0.0

            torch.cuda.empty_cache()
            gc.collect()
            # update the progress bar
            tepoch.set_postfix({"loss":round(loss.item(),3),"running loss":round(running_loss/(index%1000 + 1),3),"lossd":round(lossd.item(),3),"lossg":round(lossg.item(),3),"lossc":round(lossc.item(),3),"sequence_len": Xjf.shape[1]})
            # Log metrics
            if not CFG.debug:
                wandb.log({"epoch": epoch, "loss": loss.item(),"lossc":lossc.item(),"lossd":lossd.item(),"lossg":lossg.item(), "sequence_len": Xjf.shape[1],
                           "Exd":Exd.item(),"Eju": Eju.item(), "Ejf":Ejf.item(), "Ecd":Ecd.item(),
                           "step": ds_length*epoch+index,"Ejf_grad":Ejf_grad.item(), "Exdu":Exdu.item(),"Ecy1":Ecy1.item(),"Ecy2":Ecy2.item()})
            
            # if nan break the loop
            if torch.isnan(loss):
                print('nan loss')
                break
            
        print(f"skipped {n_skips}")
        save_checkpoint(epoch, model, optimizer, loss,0,CFG.model_path+str(epoch)+"_final_model.pt")
        # evaluate the model
        val_loss, val_lossd,val_lossg,valid_lossc = validation(model, valid_loader,CFG.device,epoch, CFG.N, optimizer , val_type = 'robust')
         # update wandb metrics
        if not CFG.debug:
            wandb.log({"epoch" : epoch ,"validation loss": val_loss, "learning rate": optimizer.param_groups[0]["lr"], "validation lossd": val_lossd, "validation lossg": val_lossg, "validation lossc": valid_lossc})
         # Update the learning rate based on the validation loss
        scheduler.step()
        print (f"validation loss: {val_loss}")
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
    scaler = torch.cuda.amp.GradScaler()
    for epoch in (range(EPOCH,CFG.num_epochs+EPOCH)):  # loop over the dataset multiple times

        
        torch.cuda.empty_cache()
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

def criterion(Ejf, Eju, Exd, X_native, Ecd, Exdu, Ecy1, Ecy2, with_grad = True , reg_alpha = CFG.reg_alpha):
    """
    The loss function for the model corresponds to 3 main losses:
    1. lossg: the partial derivative of the energy with respect to the native structure
    2. lossd: the energy of the native structure divided by the decoy energy
    3. lossc: the energy softplus function for the native and mutant structure(unfolded and folded)
    Args:
        Ejf (tensor): The energy of the folded native structure
        Ekf (tensor): The energy of the folded mutant structure
        Eju (tensor): The energy of the unfolded native structure
        Eku (tensor): The energy of the unfolded mutant structure
        Exd (tensor): The energy of the decoy sequence
        Ecd (tensor): The energy of the decoy structure
        Exdu (tensor): The energy of the decoy structure unfolded
        Ecy1 (tensor): The energy of the cycle permutation structure first amino acid
        Ecy2 (tensor): The energy of the cycle permutation structure last amino acid
    output:
        loss (tensor): The loss of the model
        lossd (tensor): The loss of the model due to the energy of the native structure divided by the decoy energy
        lossg (tensor): The loss of the model due to the partial derivative of the energy with respect to the native structure
        lossc (tensor): The loss of the model due to the energy softplus function for the native and mutant structure(unfolded and folded)
    """
    lossg = gradient_penalty(X_native, Ejf) if with_grad else torch.tensor(0.0).to(Ejf.device)
    lossd = lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju, Ecy1, Ecy2)
    # lossc = energy_softplus(Ejf, Ekf, Eju, Eku)
    # lossc will be regularization term of sum of squered energys 
    lossc = (torch.cat([Ejf.unsqueeze(0)[None,:], Eju.unsqueeze(0)[None,:],
                        Exd.unsqueeze(0)[None,:], Ecd.unsqueeze(0)[None,:],
                        Ecy1.unsqueeze(0)[None,:], Ecy2.unsqueeze(0)[None,:]])**2).mean()
    lossc = reg_alpha * lossc
    
    return lossd+lossg+lossc , lossd, lossg, lossc
  
def lossd_fucntion(Ejf, Exd, Ecd, Exdu, Eju, Ecy1, Ecy2):
    """Decoy loss:
    - the energy of a decoy sequece is greater than the energy of the wild-type structure (Ejf<Exd)
    - the energy of a decoy structure is greater than the energy of the wild-type structure (Ejf<Ecd)
    - the energy of a folded decoy is greater than the energy of an unfolded decoy (Exdu<Exd)
    - the energy of a decoy structure is greater than the energy of the unfolded native structure (Eju<Ecd)
    - the energy of the wild-type structure is lower than the energy of the cycle permutation (Ejf<Ecy1)
    - the energy of the wild-type structure is lower than the energy of the cycle permutation (Ejf<Ecy2)
    """
    # loss_decoy = lambda x,y: torch.log((x+1) / (y+1) +1)
    loss_decoy = lambda x,y: x - y
    loss = torch.cat([loss_decoy(Ejf, Exd).unsqueeze(0)[None,:], loss_decoy(Ejf, Ecd).unsqueeze(0)[None,:], 
                      loss_decoy(Eju, Ecd).unsqueeze(0)[None,:], loss_decoy(Ejf, Eju).unsqueeze(0)[None,:], 
                      loss_decoy(Ejf, Ecy1).unsqueeze(0)[None,:], loss_decoy(Ejf, Ecy2).unsqueeze(0)[None,:]])
    loss = torch.mean(loss)
    return loss

    
def main():
    # Run trailes
    # Create a study with a TPE sampler and a median pruner using SQLite storage
    sampler = TPESampler()
    pruner = MedianPruner()
    study = optuna.create_study(study_name='distributed-study',direction="maximize",
                                storage='sqlite:///example.db', sampler=sampler, pruner=pruner,
                                load_if_exists =  True)

    study.optimize(objective, n_trials=100)
    print(study.best_params)
    # Save the best hyperparameters
    df = pd.DataFrame(study.best_params)
    path = '/data/hyperparameter/'  
    os.makedirs(path, exist_ok=True)
    df.to_csv(path+'best_hyperparameters.csv')
    
    
    return 1

  
# Function to remove rows where '1' is between two letters in a specific column
def remove_rows_with_pattern(df, column_name, pattern):
    mask = df[column_name].str.contains(pattern)
    df_filtered = df[~mask]
    return df_filtered

def get_reults(base_pred_dir, experiment_dir, model_res_dir):
    results = pd.DataFrame()
    model_res_dir = os.path.join(base_pred_dir,'mutation_outputs',model_res_dir.split('/')[-2],model_res_dir.split('/')[-1]) + '/'
    for file in tqdm(os.listdir(model_res_dir)):
        if file.endswith(".csv"):
            # Load the predictions
            experiment_csv = pd.read_csv(base_pred_dir + experiment_dir + file, index_col=False)
            experiment_csv = experiment_csv[~experiment_csv['name'].str.contains('ins|del')].reset_index(drop=True)
            inference_csv = pd.read_csv(model_res_dir + file, index_col=False).drop(['mut_type'], axis=1)
            aggregated_df = pd.concat([experiment_csv, inference_csv], axis=1)
            aggregated_df['inferred_dG'] = aggregated_df['unfolded_energies'] - aggregated_df['folded_energies']

            wt = aggregated_df[aggregated_df['mut_type'] == 'wt'].iloc[0]
            aggregated_df['inferred_ddG'] = aggregated_df['inferred_dG'] - wt['inferred_dG'] 
            aggregated_df['ddG'] = aggregated_df['deltaG'] - wt['deltaG']

            mutation_df = aggregated_df['mut_type'].str.split(':', expand=True).apply(lambda x: pd.Series(list(x)))
            aggregated_df[[f'mutation_{i}' for i in range(mutation_df.shape[1])]] = mutation_df
            aggregated_df = aggregated_df[aggregated_df['mutation_1'].isna()] if 'mutation_1' in aggregated_df.columns else aggregated_df

            # aggregated_df = aggregated_df[aggregated_df['mutation_0'] != 'wt']

            # remove muratation_0 rows that contain 1 between two letters
            # Remove rows where '1' is between two letters in the 'text' column
            pattern = r'(?<=[a-zA-Z])1(?=[a-zA-Z])'
            aggregated_df = remove_rows_with_pattern(aggregated_df, "mutation_0", pattern)
            
            # normelize inffered _ddG and deltaG row with mean 0 and std 1
            # aggregated_df['inferred_dG'] = (aggregated_df['inferred_dG'] - aggregated_df['inferred_dG'].mean()) / aggregated_df['inferred_dG'].std()
            # aggregated_df['deltaG'] = (aggregated_df['deltaG'] - aggregated_df['deltaG'].mean()) / aggregated_df['deltaG'].std()
            
            aggregated_df['protein_name'] = file.split('.')[0]
            
            results = pd.concat([results, aggregated_df], axis=0)
            results = results.reset_index(drop=True)
    return results 
    
if __name__ == '__main__':
    main()
