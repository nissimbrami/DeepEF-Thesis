import glob
import os
from pathlib import Path

import pandas as pd


def preprocess_dataset2_3(dataset_path, all_pdbs, out_path, columns=None):
    if not columns:
        columns = ['name', 'aa_seq', 'mut_type', 'deltaG', 'ddG_ML', 'Stabilizing_mut']
    dataset2_3 = pd.read_csv(dataset_path, usecols=columns)
    # dataset2_3['name'] = dataset2_3.apply(lambda x: x[0].replace('|', '_').split('.pdb')[0], axis=1)
    for pdb in all_pdbs:
        pdb_dataset = dataset2_3[dataset2_3['name'].str.startswith(pdb)]
        pdb_dataset.to_csv(str(out_path / f'{pdb}.csv'), index=False)
        print(f'done {pdb}')


if __name__ ==  '__main__':
    basepath = Path('../../data/Processed_K50_dG_datasets')

    csv_out_path = basepath / 'mutation_datasets'
    os.makedirs(csv_out_path, exist_ok=True)

    pdb_files = Path(basepath / 'AlphaFold_model_PDBs')
    pdb_files = glob.glob(str(Path(pdb_files) / '*'))
    pdb_names = [str(Path(x).stem) for x in pdb_files]
    dataset2_3_path = str(Path(basepath / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'))

    preprocess_dataset2_3(dataset2_3_path, pdb_names, csv_out_path)