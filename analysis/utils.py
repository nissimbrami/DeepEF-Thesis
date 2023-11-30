import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

from model.model_cfg import CFG


def aggregate_mutation_results(experiment_data, inference_data):
    experiment_csv = pd.read_csv(experiment_data, index_col=False)
    experiment_csv = experiment_csv[~experiment_csv['name'].str.contains('ins|del')].reset_index(drop=True)
    inference_csv = pd.read_csv(inference_data, index_col=False).drop(['mut_type'], axis=1)
    aggregated_df = pd.concat([experiment_csv, inference_csv], axis=1)
    aggregated_df['inferred_dG'] = aggregated_df['unfolded_energies'] - aggregated_df['folded_energies']

    wt = aggregated_df[aggregated_df['mut_type'] == 'wt'].iloc[0]
    aggregated_df['inferred_ddG'] = wt['inferred_dG'] - aggregated_df['inferred_dG']

    mutation_df = aggregated_df['mut_type'].str.split(':', expand=True).apply(lambda x: pd.Series(list(x)))
    aggregated_df[[f'mutation_{i}' for i in range(mutation_df.shape[1])]] = mutation_df
    aggregated_df = aggregated_df[aggregated_df['mutation_1'].isna()] if 'mutation_1' in aggregated_df.columns else aggregated_df

    aggregated_df = aggregated_df[aggregated_df['mutation_0'] != 'wt']

    return aggregated_df


def get_exp_dg(metric):
    if metric == 'dG':
        return 'deltaG'
    elif metric == 'ddG':
        return 'ddG_ML'


def print_metric(metric):
    if metric == 'dG':
        return 'ΔG'
    elif metric == 'ddG':
        return 'ΔΔG'


def get_regressor(experimented_dG, inferred_dG):
    # Fit a linear regression model
    X = np.array(experimented_dG).reshape(-1, 1)
    y = np.array(inferred_dG)

    regressor = LinearRegression()
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    coef = regressor.coef_[0]
    intercept = regressor.intercept_

    r_squared = r2_score(y, y_pred)

    return y_pred, r_squared, coef, intercept


def plot_experiment_to_inferred_dg(protein_name, df, metric='dG'):
    plt.clf()
    experimented_metric = df[get_exp_dg(metric)]
    inferred_metric = df[f'inferred_{metric}']

    y_pred, coef, r_squared, intercept = get_regressor(experimented_metric, inferred_metric)

    plt.scatter(experimented_metric, inferred_metric, label="Data Points")
    plt.plot(experimented_metric, y_pred, color='red', linewidth=2,
             label=f"Linear Regression: {coef:.4f}x + {intercept:.4f}, r_squared: {r_squared:.4f}")
    plt.xlabel(f"Experimented {print_metric(metric)}")
    plt.ylabel(f"Inferred {print_metric(metric)}")
    plt.title(f"Scatterplot for {protein_name}")
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    # if CFG.debug:
    #     plt.show()
    return fig


def plot_violin(protein_name, df, metric='dG'):
    plt.clf()
    df.loc[:, 'from_aa'] = df['mutation_0'].str[0:-1]
    df.loc[:, 'to_aa'] = df['mutation_0'].str[-1]

    plt.figure(figsize=(25, 15))  # Adjust the figure size as needed
    sns.violinplot(data=df, x='from_aa', y=f'inferred_{metric}', inner='quart', palette="Set1")
    plt.xlabel("Category")
    plt.ylabel(f'inferred_{print_metric(metric)}')
    plt.title(f"Violin Plot {protein_name}")

    fig = plt.gcf()
    # if CFG.debug:
    #     plt.show()
    return fig


def plot_diff_hist(protein_name, df, metric='dG'):
    plt.clf()
    experimented_metric = df[get_exp_dg(metric)]
    inferred_metric = df[f'inferred_{metric}']

    y_pred, coef, r_squared, intercept = get_regressor(experimented_metric, inferred_metric)

    residuals = (inferred_metric*coef + intercept) - experimented_metric
    plt.hist(residuals, bins=100, color='blue', alpha=0.7, edgecolor='black')

    plt.xlabel(f"Residuals (Inferred - Experimented {print_metric(metric)})")
    plt.ylabel("No. occurrences")
    plt.title(f"Histogram of Differences for {protein_name}")
    plt.grid(True)
    fig = plt.gcf()
    # if CFG.debug:
    #     plt.show()
    return fig


def plot_dg_per_mutation(protein_name, df, metric='dG'):
    plt.clf()
    df.loc[:, 'from_aa'] = df['mutation_0'].str[0:-1]
    df.loc[:, 'to_aa'] = df['mutation_0'].str[-1]

    # aa_to_aa_agg_df = df.groupby(['from_aa', 'to_aa']).mean()
    # pos_agg_df = df.groupby('mutation_position').mean()

    aa_to_aa_df = df.pivot_table(index='from_aa', columns='to_aa', values=f'inferred_{metric}').T[
        df['from_aa'].drop_duplicates().to_list()]
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.heatmap(aa_to_aa_df, cmap='coolwarm', fmt="", cbar=True, cbar_kws={'label': f'{metric}'}, mask=aa_to_aa_df.isnull(),
                xticklabels=True, yticklabels=True)
    plt.xlabel("Original")
    plt.ylabel("Mutated amino acid")
    plt.title(f"{print_metric(metric)} per mutation per position for {protein_name}")
    fig = plt.gcf()
    # if CFG.debug:
    #     plt.show()
    return fig
