# DeepEF (Deep Protein Energy Function)

A deep learning framework for predicting protein thermodynamic stability using graph neural networks and protein language model embeddings.

## What This Project Does

DeepEF predicts protein stability properties:
- **dG prediction** — folding free energy (folded vs unfolded state)
- **ddG prediction** — change in stability upon mutation (wild-type vs mutant)
- **Native vs decoy discrimination** — distinguishing correct protein folds from incorrect ones

## Architecture

The core model (PEM — Protein Energy Model) works as follows:
1. **Input:** Protein 3D coordinates (backbone atoms N, CA, C, CB) + sequence embeddings from protein language models (ESM-2 / ProtT5)
2. **Graph construction:** Atoms as nodes, edges from distance matrices with Gaussian kernels, bonded/non-bonded features
3. **GNN layers:** Graph Convolutional / Graph Attention layers (configurable depth, default 3 layers)
4. **Output:** Energy values for folded/unfolded states; ddG computed as difference between mutant and wild-type energies

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `model/` | Core neural network: `hydro_net.py` (PEM model), `model_cfg.py` (hyperparameters), `data_loader.py` (PyTorch dataset) |
| `supervised_model/` | Fine-tuning pipeline for mutation datasets (per-protein and dataset-level training) |
| `data_creation/` | Scripts to convert PDB files + mutation CSVs into tensor format |
| `data_loaders/` | Alternative dataset implementations for dG prediction |
| `benchmarks/` | Baseline comparisons (FoldX, Rosetta, BLOSUM62, DSSP) |
| `analysis/` | Results analysis and visualization |
| `validation/` | Validation against experimental data |
| `experiments/` | Jupyter notebooks (EDA, inference, embedding analysis) |
| `tests/` | Experimental validation scripts (DSM variants, convergence tests) |
| `Megascale-fineTuning/` | Fine-tuning pipeline on the Megascale/PNAS mutation dataset |

## Key Files (Root Level)

| File | Purpose |
|------|---------|
| `train.py` | **Main training script** — light attention variant with cycle permutations |
| `train_utils.py` | Shared utilities: checkpointing, graph construction, noise injection, validation plotting |
| `protein_inference.py` | Single-protein inference using ProtT5 embeddings |
| `run_validation.py` | Validation pipeline on K50 mutation datasets |
| `hpar_tuning.py` | Optuna-based hyperparameter search |
| `get_embeddings.py` | ESM-2 embedding generation |
| `constants.py` | Amino acid and atom type mappings |
| `evaluate_train.py` | Post-training evaluation |

## Tech Stack

- **PyTorch** + **PyTorch Geometric** — model and graph neural networks
- **ESM-2** (`fair-esm`) — protein language model embeddings
- **ProtT5** (`transformers`) — alternative protein embeddings
- **BioPython** — PDB file parsing
- **Optuna** — hyperparameter tuning
- **Weights & Biases** (`wandb`) — experiment tracking
- **scikit-learn / scipy** — metrics (Pearson, Spearman correlation)

## Data Format

Each protein is stored as a folder of `.pt` tensors:
```
protein_name/
├── crd_backbone.pt    # Backbone coordinates [seq_len, 3, 3]
├── mask.pt            # Valid residue mask
├── seq_one_hot.pt     # One-hot amino acid encoding [seq_len, 20]
├── seq.pt             # Raw sequence string
├── ang.pt             # Dihedral angles
├── proT5_emb.pt       # ProtT5 embeddings [seq_len, 1024]
├── proT5_mut.pt       # Mutant embeddings
├── crd_decoy.pt       # Decoy structure coordinates
└── id.pt              # Protein identifier
```

## Setup & Running

```bash
# Environment setup
conda env create -f environment.yml
conda activate esm2_env
# or: pip install -r requirements.txt

# Training
python train.py

# Supervised fine-tuning on mutations
python supervised_model/runner.py ../data/Processed_K50_dG_datasets --mode train_single_proteins

# Inference
python protein_inference.py

# Validation
python run_validation.py

# Hyperparameter tuning
python hpar_tuning.py
```

## Model Hyperparameters

Configured in `model/model_cfg.py`:
- `num_layers = 3` — GNN depth
- `dropout_rate = 0.2`
- `gaussian_coef = -0.08` — distance kernel coefficient
- `batch_size = 1` — limited by protein graph size / GPU memory
- `lr = 0.0001`
- `reg_alpha = 0.1` — regularization weight

## Datasets

- **Training:** CASP12 structures with decoys, ProTherm experimental stability data
- **Validation:** K50 mutation datasets (S669, etc.), ThermoMPNN benchmark proteins
- **Metrics:** Pearson/Spearman correlation, RMSE, native vs decoy classification accuracy
