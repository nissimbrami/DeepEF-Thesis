import pyrosetta
pyrosetta.init()


import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.toolbox.mutants import mutate_residue

# Initialize PyRosetta
pyrosetta.init()

def calculate_delta_g(pdb_file, mutations):
    # Load the PDB file
    pose = pose_from_pdb(pdb_file)
    
    # Set up the score function
    scorefxn = pyrosetta.get_fa_scorefxn()
    
    # Calculate the initial free energy
    initial_score = scorefxn(pose)
    
    # Apply mutations
    for residue_number, new_amino_acid in mutations:
        mutate_residue(pose, residue_number, new_amino_acid)
    
    # Calculate the final free energy
    final_score = scorefxn(pose)
    
    # Compute ΔG
    delta_g = final_score - initial_score
    
    return initial_score, final_score, delta_g

# List of PDB files and their mutations
pdb_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
mutations_dict = {
    "protein1.pdb": [(10, 'A'), (25, 'V')],
    "protein2.pdb": [(15, 'L'), (30, 'F')],
    "protein3.pdb": [(20, 'T'), (35, 'Y')]
}

results = []

for pdb_file in pdb_files:
    mutations = mutations_dict.get(pdb_file, [])
    initial_score, final_score, delta_g = calculate_delta_g(pdb_file, mutations)
    results.append((pdb_file, initial_score, final_score, delta_g))
    print(f"Protein: {pdb_file}, Initial Score: {initial_score}, Final Score: {final_score}, ΔG: {delta_g}")

# Optionally, save the results to a file or further process them

