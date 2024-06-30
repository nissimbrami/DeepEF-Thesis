import os
import subprocess

# Path to FoldX executable
foldx_path = "/cs/casp15/Shahar/foldx/foldx_20241231"

# Path to the original PDB file
pdb_file = "./data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/1A32.pdb"

# List of mutations in FoldX format, e.g., ["A23T", "B42Y"]
mutations = ["V1Q", "T2G"]

# Create a directory for FoldX output
output_dir = "./data/foldx_output"
os.makedirs(output_dir, exist_ok=True)

# Run FoldX stability analysis
foldx_command = f"{foldx_path} --command=Stability --pdb={pdb_file}"# --output-dir={output_dir}"
subprocess.run(foldx_command, shell=True)


# def run_foldx(stability_analysis_command):
#     """
#     Runs FoldX with the provided command.
#     Parameters:
#         stability_analysis_command (str): The FoldX command to run.

#     Returns:
#         str: Output from the FoldX command.
#     """
#     result = subprocess.run(stability_analysis_command, shell=True, capture_output=True, text=True)
#     return result.stdout

# def predict_stability(pdb_file, output_dir='FoldX_Output', foldx_path=foldx_path):
#     """
#     Predicts the delta G stability of a protein using FoldX.

#     Parameters:
#         pdb_file (str): Path to the PDB file of the protein.
#         output_dir (str): Directory to save FoldX output files. Default is 'FoldX_Output'.

#     Returns:
#         str: Output from the FoldX stability analysis.
#     """
#     # Ensure the output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Prepare the FoldX command for stability analysis
#     foldx_command = f"{foldx_path} --command=Stability --pdb={pdb_file} --output-dir={output_dir}"
    
#     # Run FoldX
#     foldx_output = run_foldx(foldx_command)
    
#     return foldx_output

# if __name__ == "__main__":
#     # Example usage
#     pdb_file_path = pdb_file  # Replace with the path to your PDB file
#     output = predict_stability(pdb_file_path,output_dir)
#     print("FoldX Output:\n", output)
