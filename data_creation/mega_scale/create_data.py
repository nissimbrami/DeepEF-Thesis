import glob
import io
import os
from pathlib import Path

import pandas as pd
import torch
from Bio.PDB import PDBParser
from tqdm import tqdm

from constants import AMINO_ACID_MAPPER, atom_types, MISSING_COORD, AA_SEQ_COL
from data_creation.protT5_utils import load_model, get_embeddings


# from proT5_emb_create import get_emb

def create_one_hot_encoding(seq):
    one_hot_seq = torch.zeros((len(seq), len(AMINO_ACID_MAPPER)))
    for i, char in enumerate(seq):
        one_hot_seq[i, int(AMINO_ACID_MAPPER[char])] = 1
    return one_hot_seq


def get_coords(pdb_file_path, protein_name):
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(protein_name, pdb_file_path)
    out_data = dict(dict())
    sequence_dict = dict()
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.id[1] - 1
                atom_data_dict = {atom_type: [] for atom_type in atom_types}
                if 'CA' in residue:
                    for atom in residue:
                        # Check if the atom type is in the list of atom types to save
                        if atom.get_name() in atom_types:
                            # Get the coordinates of the atom
                            coord = atom.get_coord()
                            # Add the coordinates to the corresponding atom type list in the data dictionary
                            atom_data_dict[atom.get_name()] = torch.from_numpy(coord)
                    if 'C' not in residue:
                        atom_data_dict['C'] = MISSING_COORD
                    if 'N' not in residue:
                        atom_data_dict['N'] = MISSING_COORD
                    if 'CB' not in residue and residue.get_resname() == 'GLY':
                        try:
                            atom_data_dict['CB'] = torch.from_numpy(residue['2HA'].get_coord())
                        except KeyError:
                            atom_data_dict['CB'] = torch.from_numpy(residue['HA2'].get_coord())
                        # atom_data_dict['CB'] = MISSING_COORD
                    # store the atoms foreach residue id, id is stored for mask later
                    sequence_dict[residue_id] = atom_data_dict
    return torch.stack([torch.stack(list(v.values())) for v in sequence_dict.values()])


def create_training_data(csv_files, pdb_files, base_out_path, chunksize=128):
    model, tokenizer = load_model()
    for csv_file_path in glob.glob(str(csv_files / '*')):
        pdb_file_paths = [x for x in pdb_files if Path(x).stem == Path(csv_file_path).stem]
        assert len(pdb_file_paths) == 1
        pdb_file_path = pdb_file_paths[0]
        out_dir = base_out_path / Path(csv_file_path).stem
        # if os.path.isdir(out_dir):
        #     continue
        name = Path(csv_file_path).stem
        print(f'protein {name}')
        df = pd.read_csv(csv_file_path)
        if df.empty:
            continue
        coords_tensor = get_coords(pdb_file_path, name)
        mask_tensor = (coords_tensor != 0).all(dim=2).all(dim=1).float()
        df = df[~df['name'].str.contains('ins|del')]  # same protein size
        assert len(set([len(x) for x in df[AA_SEQ_COL]])) == 1

        os.makedirs(out_dir, exist_ok=True)
        embedding_dir = out_dir / 'prott5_embeddings'
        os.makedirs(embedding_dir, exist_ok=True)

        one_hot_encodings = torch.stack([create_one_hot_encoding(x) for x in df[AA_SEQ_COL]])
        deltaG = torch.tensor(df['deltaG'].to_list())

        torch.save(one_hot_encodings, str(out_dir / f'one_hot_encodings.pt'))
        torch.save(deltaG, str(out_dir / 'deltaG.pt'))
        torch.save(coords_tensor, str(out_dir / 'coords_tensor.pt'))
        torch.save(mask_tensor, str(out_dir / 'mask_tensor.pt'))

        save_prott5_embeddings_by_chunksize(name, df, embedding_dir, model, tokenizer, chunksize)


def save_prott5_embeddings_by_chunksize(protein_name, data, embedding_dir, model, tokenizer, chunksize):
    for enum, start in tqdm(enumerate(range(0, data.shape[0], chunksize))):
        chunk = data.iloc[start:start + chunksize]
        prott5_embedding = get_embeddings(model, tokenizer, chunk[AA_SEQ_COL])[:, :-1, :]
        torch.save(prott5_embedding, str(embedding_dir / f'prott5_embedding_{enum}.pt'))
        print(f'finished {protein_name}: {start}/{len(data)}')


if __name__ == '__main__':
    basepath = Path('../../data/Processed_K50_dG_datasets')
    print(f'basepath: {basepath.resolve()}')

    csv_out_path = basepath / 'mutation_datasets'
    os.makedirs(csv_out_path, exist_ok=True)

    pdb_files = Path(basepath / 'AlphaFold_model_PDBs')
    pdb_files = glob.glob(str(Path(pdb_files) / '*'))
    pdb_names = [str(Path(x).stem) for x in pdb_files]
    dataset2_3_path = str(Path(basepath / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'))

    data_out_path = basepath / 'training_data'
    os.makedirs(data_out_path, exist_ok=True)

    create_training_data(csv_out_path, pdb_files, data_out_path)
