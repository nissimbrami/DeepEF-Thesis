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
    aggregated_df['inferred_dG'] = (aggregated_df['inferred_dG'] - aggregated_df['inferred_dG'].mean()) / aggregated_df['inferred_dG'].std()
    aggregated_df['deltaG'] = (aggregated_df['deltaG'] - aggregated_df['deltaG'].mean()) / aggregated_df['deltaG'].std()
    # normelize unfolded_energies and folded_energies row with mean 0 and std 1
    aggregated_df['norm_folded_energies'] = (aggregated_df['folded_energies'] - aggregated_df['folded_energies'].mean()) / aggregated_df['folded_energies'].std()
    aggregated_df['norm_unfolded_energies'] = (aggregated_df['unfolded_energies'] - aggregated_df['unfolded_energies'].mean()) / aggregated_df['unfolded_energies'].std()
    # get norm foldeded_energies and norm_unfolded_energies diffrence
    aggregated_df['norm_avg'] = (aggregated_df['norm_unfolded_energies'] + aggregated_df['norm_folded_energies'])/2
    
    # add sperman and pearson correlation to inferred_dG and experiment_dG
    # Calculate Pearson correlation and Spearman correlation
    pearson_corr_dg = aggregated_df[['inferred_dG','deltaG']].corr(method='pearson')
    spearman_corr_dg = aggregated_df[['inferred_dG','deltaG']].corr(method='spearman')
    pearson_corr_dg_energy = aggregated_df[['folded_energies','deltaG']].corr(method='pearson')
    spearman_corr_dg_energy = aggregated_df[['folded_energies','deltaG']].corr(method='spearman')
    pearson_corr_dg_unEnergy = aggregated_df[['unfolded_energies','deltaG']].corr(method='pearson')
    spearman_corr_dg_unEnergy = aggregated_df[['unfolded_energies','deltaG']].corr(method='spearman')
    pearson_corr_unEnergy_energy = aggregated_df[['unfolded_energies','folded_energies']].corr(method='pearson')
    spearman_corr_unEnergy_energy = aggregated_df[['unfolded_energies','folded_energies']].corr(method='spearman')
    pearson_corr_avg = aggregated_df[['norm_avg','deltaG']].corr(method='pearson')
    spearman_corr_avg = aggregated_df[['norm_avg','deltaG']].corr(method='spearman')
    corr_dict = {"Pearson_dG": pearson_corr_dg['inferred_dG'].loc['deltaG'],"Spearman_dG": spearman_corr_dg['inferred_dG'].loc['deltaG'],
                 "Pearson_dG_energy": pearson_corr_dg_energy['folded_energies'].loc['deltaG'],"Spearman_dG_energy": spearman_corr_dg_energy['folded_energies'].loc['deltaG'],
                 "Pearson_dG_unEnergy": pearson_corr_dg_unEnergy['unfolded_energies'].loc['deltaG'],"Spearman_dG_unEnergy": spearman_corr_dg_unEnergy['unfolded_energies'].loc['deltaG'],
                 "Pearson_unEnergy_energy": pearson_corr_unEnergy_energy['unfolded_energies'].loc['folded_energies'],"Spearman_unEnergy_energy": spearman_corr_unEnergy_energy['unfolded_energies'].loc['folded_energies'],
                 "Pearson_norm_avg": pearson_corr_avg['norm_avg'].loc['deltaG'],"Spearman_norm_avg": spearman_corr_avg['norm_avg'].loc['deltaG']}
    return aggregated_df, corr_dict

# Function to remove rows where '1' is between two letters in a specific column
def remove_rows_with_pattern(df, column_name, pattern):
    mask = df[column_name].str.contains(pattern)
    df_filtered = df[~mask]
    return df_filtered

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


def plot_experiment_to_folded_energy(protein_name, df, metric='dG'):
    plt.clf()
    experimented_metric = df[get_exp_dg(metric)]
    inferred_metric = df[f'folded_energies']

    y_pred, coef, r_squared, intercept = get_regressor(experimented_metric, inferred_metric)

    plt.scatter(experimented_metric, inferred_metric, label="Data Points")
    plt.plot(experimented_metric, y_pred, color='red', linewidth=2,
             label=f"Linear Regression: {coef:.4f}x + {intercept:.4f}, r_squared: {r_squared:.4f}")
    plt.xlabel(f"Experimented {print_metric(metric)}")
    plt.ylabel(f"folded_energies")
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

def plot_dg_per_mutation_diff(protein_name, df, metric='dG'):
    plt.clf()
    df.loc[:, 'from_aa'] = df['mutation_0'].str[0:-1]
    df.loc[:, 'to_aa'] = df['mutation_0'].str[-1]

    # aa_to_aa_agg_df = df.groupby(['from_aa', 'to_aa']).mean()
    # pos_agg_df = df.groupby('mutation_position').mean()

    aa_to_aa_df = df.pivot_table(index='from_aa', columns='to_aa', values=f'inferred_{metric}').T[
        df['from_aa'].drop_duplicates().to_list()]
    aa_to_aa_df_exp = df.pivot_table(index='from_aa', columns='to_aa', values=f'deltaG').T[
        df['from_aa'].drop_duplicates().to_list()]
    
    aa_to_aa_df = aa_to_aa_df - aa_to_aa_df_exp
    
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.heatmap(aa_to_aa_df, cmap='coolwarm', fmt="", cbar=True, cbar_kws={'label': f'{metric} diff'}, mask=aa_to_aa_df.isnull(),
                xticklabels=True, yticklabels=True)
    plt.xlabel("Original")
    plt.ylabel("Mutated amino acid")
    plt.title(f"{print_metric(metric)} per mutation per position diffrence from experemantal {protein_name}")
    fig = plt.gcf()
    # if CFG.debug:
    #     plt.show()
    return fig
