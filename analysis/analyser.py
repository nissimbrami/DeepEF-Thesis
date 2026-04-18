import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import train_utils
import seaborn as sns


protein_path = "./data/Processed_K50_dG_datasets/mutation_datasets/1A0N.csv"
AA_MAP = train_utils.AA_MAP
AA_KEYS = AA_MAP.keys()
df_protein = pd.read_csv(protein_path)
df_protein['aa_len'] = df_protein['aa_seq'].apply(lambda x: len(x))

aa_matrix = np.zeros((len(AA_KEYS), df_protein['aa_len'].max()))
# Iterate over the protein dataframe
for i, row in df_protein.iterrows():
    # Iterate over the amino acids in the sequence
    for j, aa in enumerate(row['aa_seq']):
        # Get the index of the amino acid type in the AA_MAP
        aa_index = AA_MAP[aa]
        # Increment the value in the matrix at the row of the amino acid type and the column of the position
        aa_matrix[aa_index, j] += 1
# print len of 
print(len(df_protein))
 
# plot the matrix
plt.figure(figsize=(20, 10))
sns.heatmap(aa_matrix, cmap='Blues', xticklabels=range(1, df_protein['aa_len'].max() + 1), yticklabels=AA_KEYS)
plt.xlabel('Position')
plt.ylabel('Amino Acid')
plt.title(f'Amino Acid Frequency for {protein_path.split("/")[-1].split(".")[0]}')
plt.show()

print("done")
