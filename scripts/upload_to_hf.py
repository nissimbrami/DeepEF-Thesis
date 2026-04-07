"""
Upload DeepEF datasets to Hugging Face Hub.

Usage:
    # Login first (one-time):
    huggingface-cli login

    # Upload everything:
    python scripts/upload_to_hf.py --repo YOUR_HF_USERNAME/deepef-data

    # Upload only a specific dataset:
    python scripts/upload_to_hf.py --repo YOUR_HF_USERNAME/deepef-data --dataset casp12
    python scripts/upload_to_hf.py --repo YOUR_HF_USERNAME/deepef-data --dataset k50
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

DATASETS = {
    "casp12": {
        "local_path": DATA_DIR / "casp12_data_30",
        "repo_path": "casp12_data_30",
        "description": "CASP12 protein structures with ProtT5 embeddings (~110GB)",
    },
    "k50": {
        "local_path": DATA_DIR / "Processed_K50_dG_datasets",
        "repo_path": "Processed_K50_dG_datasets",
        "description": "K50 ddG mutation datasets with AlphaFold PDBs (~2.2GB)",
    },
    "megascale_csv": {
        "local_path": DATA_DIR / "megascale_proteins.csv",
        "repo_path": "megascale_proteins.csv",
        "description": "Megascale protein list CSV",
    },
}


def upload_dataset(api: HfApi, repo_id: str, name: str, info: dict, num_workers: int):
    local = info["local_path"]
    repo_path = info["repo_path"]

    if not local.exists():
        print(f"[{name}] Skipping — path not found: {local}")
        return

    print(f"\n[{name}] Uploading {local} → {repo_id}/{repo_path}")
    print(f"  {info['description']}")

    if local.is_dir():
        # upload_large_folder uploads from the given folder_path as root.
        # To preserve the directory name in the repo, upload from the parent
        # and filter to only the target subfolder.
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local.parent),
            allow_patterns=[f"{local.name}/*"],
            num_workers=num_workers,
        )
    else:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(local),
            path_in_repo=repo_path,
        )

    print(f"[{name}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Upload DeepEF datasets to HF Hub")
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. username/deepef-data")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to upload (default: all)",
    )
    parser.add_argument("--private", action="store_true", help="Create as private repo")
    parser.add_argument("--workers", type=int, default=8, help="Upload parallelism (default: 8)")
    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/verifying repo: {args.repo}")
    create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # Upload README card
    readme = f"""---
license: mit
tags:
  - protein
  - structural-biology
  - thermodynamic-stability
---

# DeepEF Datasets

Protein structure datasets used for training and evaluating [DeepEF](https://github.com/shaharec/DeepEF) —
a deep learning framework for predicting protein thermodynamic stability.

## Contents

| Path | Size | Description |
|------|------|-------------|
| `casp12_data_30/` | ~110GB | CASP12 structures with ProtT5 embeddings; per-protein `.pt` tensors split into train/test/valid-* |
| `Processed_K50_dG_datasets/` | ~2.2GB | K50 ddG mutation datasets with AlphaFold PDB models |
| `megascale_proteins.csv` | small | Megascale protein list |

## Data Format (casp12_data_30)

Each protein is a folder under `train/`, `test/`, or `valid-*/`:
```
PROTEIN_ID/
├── crd_backbone.pt       # Backbone coordinates [seq_len, 4, 3]
├── ang.pt                # Dihedral angles
├── mask.pt               # Valid residue mask
├── seq_one_hot.pt        # One-hot amino acid encoding [seq_len, 20]
├── seq.pt                # Raw sequence string
├── seq_mut.pt            # Mutant sequence string
├── proT5_emb.pt          # ProtT5 embeddings [seq_len, 1024]
├── proT5_emb_cycle1-4.pt # Cyclic permutation embeddings
└── proT5_emb_mut.pt      # Mutant ProtT5 embeddings
```

## Usage

```python
from huggingface_hub import snapshot_download, hf_hub_download
import torch

# Download a single protein
path = hf_hub_download(
    repo_id="{args.repo}",
    repo_type="dataset",
    filename="casp12_data_30/train/1A0C_1_A/crd_backbone.pt",
)
coords = torch.load(path)

# Download an entire split (large!)
snapshot_download(
    repo_id="{args.repo}",
    repo_type="dataset",
    allow_patterns="casp12_data_30/train/*",
    local_dir="./data",
)
```
"""
    api.upload_file(
        repo_id=args.repo,
        repo_type="dataset",
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
    )

    # Upload selected datasets
    to_upload = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, info in to_upload.items():
        upload_dataset(api, args.repo, name, info, args.workers)

    print(f"\nAll done. View at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
