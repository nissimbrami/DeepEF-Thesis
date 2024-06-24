import pandas as pd
import os
import re
from tqdm import tqdm

# Function to split each segment
def split_segment(segment):
    match = re.match(r"([A-Za-z])(\d+)([A-Za-z])", segment)
    if match:
        return match.groups()
    return (segment)

# Function to handle multiple segments
def split_string(s):
    segments = s.split(':')
    split_segments = [split_segment(segment) for segment in segments]
    return split_segments

def create_rosetta_csv():
    """Create a csv file with rosetta data."""
    pdb_dir = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/"
    mutation_dir ="./data/Processed_K50_dG_datasets/mutation_datasets/"
    rosetta_df = pd.DataFrame()
    # Get all data
    protein_list = os.listdir(mutation_dir)
    for protein in tqdm(protein_list):
        protein_df = pd.DataFrame(columns=["name","pdb_path", "mut_type", "deltaG"])
        protein_mutation = pd.read_csv(mutation_dir + protein )
        protein_df["name"] = protein_mutation['name']
        protein_df["pdb_path"] = pdb_dir + protein.replace(".csv", "")+".pdb"
        protein_df["mut_type"] = protein_mutation["mut_type"]
        protein_df["deltaG"] = protein_mutation["deltaG"]
        
        # split the mut_type to tuples
        protein_df['mut_tuple'] = protein_df['mut_type'].apply(split_string)
        
        rosetta_df = pd.concat([rosetta_df, protein_df])
    rosetta_df.to_csv("./data/Processed_K50_dG_datasets/rosetta.csv", index=False)    

    
create_rosetta_csv()

