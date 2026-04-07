"""
Helper to access DeepEF datasets from Hugging Face Hub — usable from any machine.

Usage:
    from scripts.hf_dataset import DeepEFDataset, download_split

    # Stream a single protein tensor (no full download needed):
    dataset = DeepEFDataset("YOUR_HF_USERNAME/deepef-data")
    coords = dataset.load_tensor("12AS_1_A", "crd_backbone", split="train")

    # Download an entire split to a local cache:
    local_dir = download_split("YOUR_HF_USERNAME/deepef-data", split="train")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download, snapshot_download

Split = Literal["train", "test", "valid-10", "valid-20", "valid-30", "valid-40", "valid-50", "valid-70", "valid-90"]

_TENSOR_FILES = [
    "crd_backbone",
    "ang",
    "mask",
    "seq_one_hot",
    "seq",
    "seq_mut",
    "proT5_emb",
    "proT5_emb_mut",
    "proT5_emb_cycle1",
    "proT5_emb_cycle2",
    "proT5_emb_cycle3",
    "proT5_emb_cycle4",
    "id",
]


class DeepEFDataset:
    """Lazy accessor for the DeepEF HF dataset — downloads individual files on demand."""

    def __init__(self, repo_id: str, cache_dir: str | None = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir  # None → default HF cache (~/.cache/huggingface)

    def load_tensor(self, protein_id: str, tensor_name: str, split: Split = "train") -> torch.Tensor:
        """Download and load a single tensor for a protein.

        Args:
            protein_id: Folder name, e.g. "12AS_1_A"
            tensor_name: File stem, e.g. "crd_backbone" or "proT5_emb"
            split: Dataset split

        Returns:
            Loaded torch tensor
        """
        filename = f"casp12_data_30/{split}/{protein_id}/{tensor_name}.pt"
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=self.cache_dir,
        )
        return torch.load(local_path, weights_only=True)

    def load_protein(self, protein_id: str, split: Split = "train") -> dict[str, torch.Tensor]:
        """Download and load all tensors for a protein."""
        result = {}
        for name in _TENSOR_FILES:
            try:
                result[name] = self.load_tensor(protein_id, name, split)
            except Exception:
                pass  # Some files may not exist for all proteins
        return result


def download_split(
    repo_id: str,
    split: Split,
    local_dir: str | Path = "./data",
    cache_dir: str | None = None,
) -> Path:
    """Download a full split from the HF dataset.

    WARNING: train split is ~100GB. Use valid-* splits for testing.

    Returns the local path to the downloaded split directory.
    """
    local_dir = Path(local_dir)
    print(f"Downloading casp12_data_30/{split} from {repo_id}...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"casp12_data_30/{split}/*"],
        local_dir=str(local_dir),
        cache_dir=cache_dir,
    )
    dest = local_dir / "casp12_data_30" / split
    print(f"Downloaded to: {dest}")
    return dest


def download_k50(
    repo_id: str,
    local_dir: str | Path = "./data",
) -> Path:
    """Download the K50 ddG mutation datasets."""
    local_dir = Path(local_dir)
    print(f"Downloading Processed_K50_dG_datasets from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["Processed_K50_dG_datasets/*"],
        local_dir=str(local_dir),
    )
    dest = local_dir / "Processed_K50_dG_datasets"
    print(f"Downloaded to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Quick sanity-check / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else input("HF repo id (e.g. username/deepef-data): ")
    dataset = DeepEFDataset(repo)
    print("Fetching coords for 12AS_1_A (train)...")
    coords = dataset.load_tensor("12AS_1_A", "crd_backbone", split="train")
    print(f"  crd_backbone shape: {coords.shape}")
    print("OK — dataset is accessible.")
