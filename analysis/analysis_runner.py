import glob
import os
from pathlib import Path
import pandas as pd

from analysis.utils import aggregate_mutation_results, plot_experiment_to_inferred_dg, plot_diff_hist, \
    plot_dg_per_mutation, plot_violin, plot_dg_per_mutation_diff, plot_experiment_to_folded_energy
from model.model_cfg import CFG


def get_plots(model_name, protein_name, df, metric='dG'):
    for plot_function in [plot_experiment_to_inferred_dg, plot_diff_hist,
                          plot_dg_per_mutation, plot_violin,plot_dg_per_mutation_diff,
                          plot_experiment_to_folded_energy]:
        plt = plot_function(protein_name, df, metric)
        protein_graph_folder = os.path.join('.', model_name, protein_name)
        os.makedirs(protein_graph_folder, exist_ok=True)
        plt.savefig(os.path.join(protein_graph_folder, f'{plot_function.__name__}.png'), dpi=300)

def run_analysis(model_path = CFG.model_path):
    model = '/'.join(model_path.split('/')[-2:])
    print(f'taking model {model}')
    df_correlation = pd.DataFrame(columns=['Protein', 'Pearson_dG', 'Spearman_dG','Pearson_dG_energy', 'Spearman_dG_energy','Pearson_dG_unEnergy', 'Spearman_dG_unEnergy','Pearson_unEnergy_energy', 'Spearman_unEnergy_energy',
                                           'Pearson_norm_avg', 'Spearman_norm_avg'])
    experiment_outputs = './data/Processed_K50_dG_datasets/mutation_datasets'
    inference_outputs = './data/Processed_K50_dG_datasets/mutation_outputs' +'/'+model+'/*'
    for inference_data in glob.glob(str(inference_outputs)):
        print(f'analysing {Path(inference_data).stem}')
        experiment_data = Path(experiment_outputs) / f'{Path(inference_data).name}'
        aggregated_df, corr_dict = aggregate_mutation_results(experiment_data, inference_data)
        corr_dict['Protein'] = Path(inference_data).stem
        # append to df_correlation
        df_correlation = df_correlation.append(corr_dict, ignore_index=True)
        get_plots(model, Path(inference_data).stem, aggregated_df)
        
         # save pearson and spearman correlation
        protein_graph_folder = os.path.join('.', model)
        os.makedirs(protein_graph_folder, exist_ok=True)
        df_correlation.to_csv(os.path.join(protein_graph_folder, f'correlation.csv'),index=False)


if __name__ == '__main__':
    model = Path(CFG.model_path).stem
    print(f'taking model {model}')
    experiment_outputs = r'..\data\Processed_K50_dG_datasets\mutation_datasets'
    inference_outputs = Path(r'..\data\Processed_K50_dG_datasets\mutation_outputs') / model / '*'
    for inference_data in glob.glob(str(inference_outputs)):
        print(f'analysing {Path(inference_data).stem}')
        experiment_data = Path(experiment_outputs) / f'{Path(inference_data).name}'
        aggregated_df = aggregate_mutation_results(experiment_data, inference_data)
        get_plots(model, Path(inference_data).stem, aggregated_df)
