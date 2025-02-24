# This script runs the validation and analysis for all models in the models_list
import os 
import subprocess
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from validation.validation import run_validation
from analysis.analysis_runner import run_analysis
import wandb

DEBUG = False
BASE_MODEL = './res/trianed_models-light_attention'


if not DEBUG:
    wandb.init(project="MS validation",name = BASE_MODEL.split('/')[-1])
    # Add config

def get_TMprotein():
    TM_path = "./data/ThermoMPNN/mega_test.csv"
    TM_df = pd.read_csv(TM_path)
    TM_proteins = TM_df["name_original"].unique().tolist()
    return TM_proteins

def get_reults(base_pred_dir='./data/Processed_K50_dG_datasets/', experiment_dir='mutation_datasets/', 
               model_res_dir = './data/Processed_K50_dG_datasets/mutation_outputs/trianed_models-cycle_per_norm/7_final_model.pt/' ):
    results = pd.DataFrame()
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

            
            # normelize unfolded_energies and folded_energies row with mean 0 and std 1
            aggregated_df['norm_folded_energies'] = (aggregated_df['folded_energies'] - aggregated_df['folded_energies'].mean()) / aggregated_df['folded_energies'].std()
            aggregated_df['norm_unfolded_energies'] = (aggregated_df['unfolded_energies'] - aggregated_df['unfolded_energies'].mean()) / aggregated_df['unfolded_energies'].std()
            # get norm foldeded_energies and norm_unfolded_energies diffrence
            aggregated_df['norm_avg'] = (aggregated_df['norm_unfolded_energies'] + aggregated_df['norm_folded_energies'])/2
            aggregated_df['protein_name'] = file.split('.')[0]
            
            results = pd.concat([results, aggregated_df], axis=0)
    results = results.reset_index(drop=True)
    return results

def print_stat(model_path):
    
    model_res_dir = './data/Processed_K50_dG_datasets/mutation_outputs/' + ('/').join(model_path.split('/')[-2:]) + '/'
    epoch = int(model_path.split('/')[-1].split('_')[0])
    results = get_reults(model_res_dir=model_res_dir)
    print('results: ', len(results))
    # pc, sp and RMSE
    ddg_pearson = results[['inferred_ddG', 'ddG']].corr(method='pearson').iloc[0,1]
    ddg_spearman = results[['inferred_ddG', 'ddG']].corr(method='spearman').iloc[0,1]
    ddg_rmse = np.sqrt(np.mean((results['inferred_ddG'] - results['ddG'])**2))
    dG_pearson = results[['inferred_dG', 'deltaG']].corr(method='pearson').iloc[0,1]
    dG_spearman = results[['inferred_dG', 'deltaG']].corr(method='spearman').iloc[0,1]
    dG_rmse = np.sqrt(np.mean(results['inferred_dG'] - results['deltaG'])**2)
    # log stats
    if not DEBUG:
        wandb.log({'epoch':epoch, 'ddg_pearson': ddg_pearson, 'ddg_spearman': ddg_spearman, 'ddg_rmse': ddg_rmse, 'dG_pearson': dG_pearson, 'dG_spearman': dG_spearman, 'dG_rmse': dG_rmse})
    print(f"pearson coorelation DDG {ddg_pearson}")
    print(f"spearman coorelation DDG {ddg_spearman}")
    print(f"RMSE DDG {ddg_rmse}")
    print(f"pearson coorelation dG {dG_pearson}")
    print(f"spearman coorelation dG {dG_spearman}")
    print(f"RMSE dG {dG_rmse}")
    # Group by protein
    print('Group by protein')
    protein_results = results.groupby('protein_name')
    protein_ddg_pearson = protein_results.apply(lambda x: x['inferred_ddG'].corr(x['ddG'], method='pearson')).mean()
    protein_ddg_spearman = protein_results.apply(lambda x: x['inferred_ddG'].corr(x['ddG'], method='spearman')).mean()
    protein_ddg_rmse = protein_results.apply(lambda x: np.sqrt(np.mean((x['inferred_ddG'] - x['ddG'])**2))).mean()
    protein_dG_pearson = protein_results.apply(lambda x: x['inferred_dG'].corr(x['deltaG'], method='pearson')).mean()
    protein_dG_spearman = protein_results.apply(lambda x: x['inferred_dG'].corr(x['deltaG'], method='spearman')).mean()
    protein_dG_rmse = protein_results.apply(lambda x: np.sqrt(np.mean((x['inferred_dG'] - x['deltaG'])**2))).mean()
    # log stats
    if not DEBUG:
        wandb.log({'protein_ddg_pearson': protein_ddg_pearson, 'protein_ddg_spearman': protein_ddg_spearman, 'protein_ddg_rmse': protein_ddg_rmse, 'protein_dG_pearson': protein_dG_pearson, 'protein_dG_spearman': protein_dG_spearman, 'protein_dG_rmse': protein_dG_rmse})
    print(f"pearson coorelation DDG {protein_ddg_pearson}")
    print(f"spearman coorelation DDG {protein_ddg_spearman}")
    print(f"RMSE DDG {protein_ddg_rmse}")
    print(f"pearson coorelation dG {protein_dG_pearson}")
    print(f"spearman coorelation dG {protein_dG_spearman}")
    print(f"RMSE dG {protein_dG_rmse}")
    
    
    TM_proteins = get_TMprotein()
    TM_results = results[results['name'].isin(TM_proteins)]
    print('TM_proteins: ', len(TM_proteins))
    print('TM_results: ', len(TM_results))
    # pc, sp and RMSE
    print(f"pearson coorelation DDG {TM_results[['inferred_ddG', 'ddG']].corr(method='pearson').iloc[0,1]}")
    print(f"spearman coorelation DDG {TM_results[['inferred_ddG', 'ddG']].corr(method='spearman').iloc[0,1]}")
    print(f"RMSE DDG {np.sqrt(np.mean((TM_results['inferred_ddG'] - TM_results['ddG'])**2))}")
    # Group by protein
    print('Group by protein')
    protein_results = TM_results.groupby('protein_name')
    print(f"pearson coorelation DDG {protein_results.apply(lambda x: x['inferred_ddG'].corr(x['ddG'], method='pearson')).mean()}")
    print(f"spearman coorelation DDG {protein_results.apply(lambda x: x['inferred_ddG'].corr(x['ddG'], method='spearman')).mean()}")
    print(f"RMSE DDG {protein_results.apply(lambda x: np.sqrt(np.mean((x['inferred_ddG'] - x['ddG'])**2))).mean()}")
    # log stats
    if not DEBUG:
        wandb.log({'epoch':epoch, 'TM_ddg_pearson': TM_results[['inferred_ddG', 'ddG']].corr(method='pearson').iloc[0,1], 'TM_ddg_spearman': TM_results[['inferred_ddG', 'ddG']].corr(method='spearman').iloc[0,1], 'TM_ddg_rmse': np.sqrt(np.mean((TM_results['inferred_ddG'] - TM_results['ddG'])**2))})
        wandb.log({'TM_protein_ddg_pearson': protein_results.apply(lambda x: x['inferred_ddG'].corr(x['ddG'], method='pearson')).mean(), 'TM_protein_ddg_spearman': protein_results.apply(lambda x: x['inferred_ddG'].corr(x['ddG'], method='spearman')).mean(), 'TM_protein_ddg_rmse': protein_results.apply(lambda x: np.sqrt(np.mean((x['inferred_ddG'] - x['ddG'])**2))).mean()})
    


def main():
    model_list = []
    for i in range(58 , 59):
    #    model_list.append(f'res/trianed_models-cycle2_5_outline2/{i}_final_model.pt')
    #    model_list.append(f'res/trianed_models-cycle2_5_outline3/{i}_final_model.pt')
    #    model_list.append(BASE_MODEL+f"/{i}_final_model.pt")
        model_list.append("./Megascale-fineTuning/models/deepef2_ds_model/epoch_83.pt")
        model_list.append("./Megascale-fineTuning/models/deepef1_ds_model/epoch_83.pt")
        model_list.append("./Megascale-fineTuning/models/pnas_ds_model/epoch_83.pt")
        # model_list.append(f'Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf/kf_4_epoch_49.pt')
    #    model_list.append(f'res/trianed_models-droupout-0.8/{i}_final_model.pt')
        # model_list.append(f'res/trianed_models-cycle_per_2_norm/{i}_final_model.pt')
        # model_list.append(f'res/trianed_models-cycle2_5/{i}_final_model.pt')
        # model_list.append(f'res/trianed_models-droupout-0.8,gs/{i}_final_model.pt')
    # model_list = ['Megascale-fineTuning/models/PEM_full_trained-PEM_fine_tuned-trianed_models-cycle_perL1L1/1.pt']
    mini_batch_size = 32
    # run validation for all models
    for model in model_list:
        # eval_path ='/'.join(model.split('/')[-2:])
        # if os.path.exists(eval_path):
        #     print('model already validated: ', model)
        #     print('printing stats for model: ', model)
        #     print_stat(model)
        #     continue
        try: 
            print('running validation for model: ', model)
            run_validation(r'./data/Processed_K50_dG_datasets',model_path=r'./'+model, mini_batch_size=mini_batch_size)
            # clear memory
            gc.collect()
            print('validation done for model: ', model)
            print('running analysis for model: ', model)
            # run_analysis(r'./'+model)
            print('analysis done for model: ', model)
        except Exception as e:
            print('error occured for model: ', model)
            print(e)
            continue
        # print stats
        print('printing stats for model: ', model)
        print_stat(model)
        gc.collect()



if __name__ == "__main__":
    
    main()