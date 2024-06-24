import pyrosetta
pyrosetta.init()


import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.toolbox.mutants import mutate_residue
from dataset import benckmark_datasets
import re
from tqdm import tqdm
import pandas as pd
# Initialize PyRosetta
pyrosetta.init()

# Function to split each segment
def split_segment(segment):
    match = re.match(r"([A-Za-z])(\d+)([A-Za-z])", segment)
    if match:
        return match.groups()
    return (segment)

def split_string(s):
    segments = s.split(':')
    split_segments = [split_segment(segment) for segment in segments]
    return split_segments

def calculate_delta_g(pdb_file, mutations):
    # Load the PDB file
    pose = pose_from_pdb(pdb_file)
    
    # Set up the score function
    scorefxn = pyrosetta.get_fa_scorefxn()
    
    # Calculate the initial free energy
    initial_score = scorefxn(pose)
    
    # Apply mutations
    for old_amino_acif, residue_number, new_amino_acid in mutations:
        mutate_residue(pose, int(residue_number), new_amino_acid)
    
    # Calculate the final free energy
    final_score = scorefxn(pose)
    
    # Compute ΔG
    delta_g = final_score - initial_score
    
    return initial_score, final_score, delta_g


def test_calculate_delta_g():
   # List of PDB files and their mutations
    base_dir = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/"
    pdb_files = [base_dir + "1A0N.pdb"]
    mutations_dict = {
        "A10N.pdb": [(10, 'P')]
    }

    results = []

    for pdb_file in pdb_files:
        mutations = mutations_dict.get(pdb_file, [])
        initial_score, final_score, delta_g = calculate_delta_g(pdb_file, mutations)
        results.append((pdb_file, initial_score, final_score, delta_g))
        print(f"Protein: {pdb_file}, Initial Score: {initial_score}, Final Score: {final_score}, ΔG: {delta_g}")

def validate_deltaG():
    dataset = benckmark_datasets("Rosetta")
    df_results = pd.DataFrame(columns=["name","pdb_path", "mut_type", "deltaG"])
    for i in tqdm(range(len(dataset.dataset))):
        item = dataset.get_item(i)
        pdb_file = item['pdb_path']
        if (item['mut_type'] != 'wt'):
            mutations = split_string(item['mut_type'])
            initial_score, final_score, delta_g = calculate_delta_g(pdb_file, mutations)
            df_results.loc[i] = [item['name'], item['pdb_path'], item['mut_type'], delta_g]
            if(delta_g != 0):
                print(f"Protein: {pdb_file}, Initial Score: {initial_score}, Final Score: {final_score}, ΔG: {delta_g}")
   
        df_results.to_csv("./data/Processed_K50_dG_datasets/rosetta_valid.csv", index=False)

if __name__ == "__main__":
    # test_calculate_delta_g()
    validate_deltaG()