import glob
import os
from pathlib import Path

from analysis.utils import aggregate_mutation_results, plot_experiment_to_inferred_dg, plot_diff_hist, \
    plot_dg_per_mutation, plot_violin
from model.model_cfg import CFG


def get_plots(model_name, protein_name, df, metric='dG'):
    for plot_function in [plot_experiment_to_inferred_dg, plot_diff_hist,
                          plot_dg_per_mutation, plot_violin]:
        plt = plot_function(protein_name, df, metric)
        protein_graph_folder = os.path.join('.', model_name, protein_name)
        os.makedirs(protein_graph_folder, exist_ok=True)
        plt.savefig(os.path.join(protein_graph_folder, f'{plot_function.__name__}.png'), dpi=300)


if __name__ == '__main__':
    model = Path(CFG.model_path).stem
    print(f'taking model {model}')
    experiment_outputs = Path('..') / 'data' / 'Processed_K50_dG_datasets' / 'mutation_datasets'
    mutation_outputs = Path('..') / 'data' / 'Processed_K50_dG_datasets' / 'mutation_outputs'
    inference_outputs = mutation_outputs / model / '*'
    for inference_data in glob.glob(str(inference_outputs)):
        print(f'analysing {Path(inference_data).stem}')
        experiment_data = Path(experiment_outputs) / f'{Path(inference_data).name}'
        aggregated_df = aggregate_mutation_results(experiment_data, inference_data)
        get_plots(model, Path(inference_data).stem, aggregated_df)
