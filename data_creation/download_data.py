"""
DeepEF Data Download and Preparation Script
============================================
Downloads and prepares all data needed for training:

1. K50/Tsuboyama mutation data (from Zenodo) — for fine-tuning/validation
2. CASP12 pre-training data (from SidechainNet) — for pre-training
3. ProtT5 embeddings — generated from protein sequences

Usage:
    # Download everything (K50 + CASP12 + embeddings)
    python download_data.py --all

    # Download only K50 mutation data
    python download_data.py --k50

    # Download and convert CASP12 data from SidechainNet
    python download_data.py --casp12

    # Generate ProtT5 embeddings for CASP12 data (requires GPU, run after --casp12)
    python download_data.py --embeddings

    # Process K50 data (run after --k50 to create training tensors)
    python download_data.py --process-k50
"""

import argparse
import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Zenodo URLs for Tsuboyama 2023 dataset (Nature publication version)
ZENODO_BASE = "https://zenodo.org/records/7992926/files"
K50_URLS = {
    "Processed_K50_dG_datasets.zip": f"{ZENODO_BASE}/Processed_K50_dG_datasets.zip?download=1",
    "AlphaFold_model_PDBs.zip": f"{ZENODO_BASE}/AlphaFold_model_PDBs.zip?download=1",
}


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"Downloading {dest_path.name}...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest_path), reporthook=_progress)
    print()  # newline after progress


def download_k50():
    """Download K50/Tsuboyama mutation datasets from Zenodo."""
    os.makedirs(DATA_DIR, exist_ok=True)
    k50_dir = DATA_DIR / "Processed_K50_dG_datasets"

    if k50_dir.exists() and any(k50_dir.iterdir()):
        print(f"K50 data already exists at {k50_dir}, skipping download.")
        return

    for filename, url in K50_URLS.items():
        zip_path = DATA_DIR / filename
        if not zip_path.exists():
            download_file(url, zip_path)

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"  Extracted to {DATA_DIR}")

        # Move AlphaFold PDBs into the K50 directory if extracted separately
        af_standalone = DATA_DIR / "AlphaFold_model_PDBs"
        af_target = k50_dir / "AlphaFold_model_PDBs"
        if af_standalone.exists() and not af_target.exists():
            os.makedirs(k50_dir, exist_ok=True)
            shutil.move(str(af_standalone), str(af_target))
            print(f"  Moved AlphaFold PDBs to {af_target}")

        # Clean up zip
        zip_path.unlink()
        print(f"  Removed {filename}")

    print("K50 data download complete.")


def process_k50():
    """Process K50 data: split CSV by protein and create training tensors."""
    k50_dir = DATA_DIR / "Processed_K50_dG_datasets"
    if not k50_dir.exists():
        print("ERROR: K50 data not found. Run with --k50 first.")
        return

    # Step 1: Preprocess — split master CSV into per-protein CSVs
    print("Step 1: Splitting master CSV into per-protein files...")
    sys.path.insert(0, str(BASE_DIR))
    from data_creation.mega_scale.preprocess_data import preprocess_dataset2_3
    import glob

    csv_out_path = k50_dir / "mutation_datasets"
    os.makedirs(csv_out_path, exist_ok=True)

    pdb_dir = k50_dir / "AlphaFold_model_PDBs"
    pdb_files = glob.glob(str(pdb_dir / "*"))
    pdb_names = [str(Path(x).stem) for x in pdb_files]

    dataset_csv = str(k50_dir / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
    if not os.path.exists(dataset_csv):
        # Try to find it
        candidates = list(k50_dir.glob("*Dataset2*Dataset3*.csv"))
        if candidates:
            dataset_csv = str(candidates[0])
        else:
            print(f"ERROR: Cannot find Tsuboyama CSV in {k50_dir}")
            return

    preprocess_dataset2_3(dataset_csv, pdb_names, csv_out_path)
    print(f"  Created per-protein CSVs in {csv_out_path}")

    # Step 2: Create training tensors (coordinates, one-hot, embeddings)
    print("Step 2: Creating training tensors (this requires ProtT5 and may take a while)...")
    from data_creation.mega_scale.create_data import create_training_data

    data_out_path = k50_dir / "training_data"
    os.makedirs(data_out_path, exist_ok=True)

    create_training_data(csv_out_path, pdb_files, data_out_path)
    print(f"  Created training data in {data_out_path}")
    print("K50 processing complete.")


def download_casp12(thinning=100):
    """Download and convert CASP12 data from SidechainNet."""
    try:
        import sidechainnet as scn
    except ImportError:
        print("ERROR: sidechainnet not installed. Run: pip install sidechainnet")
        return

    import torch
    import numpy as np

    casp_dir = DATA_DIR / f"casp12_data_{thinning}"
    if casp_dir.exists() and any(casp_dir.iterdir()):
        print(f"CASP12 data already exists at {casp_dir}, skipping.")
        return

    print(f"Loading SidechainNet CASP12 (thinning={thinning})...")
    print("  This downloads ~1-2 GB and may take several minutes on first run.")
    data = scn.load(casp_version=12, thinning=thinning)

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    def seq_to_one_hot(seq_str):
        """Convert amino acid sequence string to one-hot tensor."""
        one_hot = torch.zeros(len(seq_str), 20)
        for i, aa in enumerate(seq_str):
            if aa in aa_to_idx:
                one_hot[i, aa_to_idx[aa]] = 1
        return one_hot

    total_saved = 0
    for split_name in data.splits:
        split_dir = casp_dir / split_name
        os.makedirs(split_dir, exist_ok=True)
        print(f"  Processing split: {split_name}")

        prot_ids = data.split_to_ids.get(split_name, [])
        for prot_id in prot_ids:
            try:
                protein = data[prot_id]
            except (KeyError, Exception):
                continue

            seq = protein.seq
            # Skip very short or very long sequences
            if len(seq) < 10 or len(seq) > 600:
                continue

            # Coordinates: [L, num_atoms, 3] — take N(0), CA(1), C(2) for backbone
            coords = protein.coords
            if coords is None or len(coords) == 0:
                continue
            if isinstance(coords, np.ndarray):
                coords = torch.tensor(coords, dtype=torch.float32)
            # coords shape is [L, 15, 3] (15 atoms per residue)
            backbone = coords[:, :3, :].clone()  # [L, 3, 3] = N, CA, C

            # Mask string (already '+'/'-' format from SidechainNet)
            mask_str = protein.mask

            # One-hot encoding
            one_hot = seq_to_one_hot(seq)

            # Angles [L, 12]
            angles = protein.angles
            if angles is not None:
                if isinstance(angles, np.ndarray):
                    ang = torch.tensor(angles, dtype=torch.float32)
                else:
                    ang = torch.tensor(angles, dtype=torch.float32) if not isinstance(angles, torch.Tensor) else angles.float()
            else:
                ang = torch.zeros(len(seq), 12)

            # Clean protein ID for filesystem
            prot_id_clean = prot_id.replace("/", "_").replace("\\", "_")
            prot_dir = split_dir / prot_id_clean
            os.makedirs(prot_dir, exist_ok=True)

            # Save tensors
            torch.save(prot_id, str(prot_dir / "id.pt"))
            torch.save(backbone, str(prot_dir / "crd_backbone.pt"))
            torch.save(mask_str, str(prot_dir / "mask.pt"))
            torch.save(one_hot, str(prot_dir / "seq_one_hot.pt"))
            torch.save(seq, str(prot_dir / "seq.pt"))
            torch.save(ang, str(prot_dir / "ang.pt"))

            total_saved += 1
            if total_saved % 500 == 0:
                print(f"    Saved {total_saved} proteins...")

    print(f"  CASP12 conversion complete: {total_saved} proteins saved to {casp_dir}")
    print("  NOTE: ProtT5 embeddings still need to be generated. Run with --embeddings")


def generate_embeddings():
    """Generate ProtT5 embeddings for CASP12 proteins."""
    import torch
    import re

    casp_dir = DATA_DIR / "casp12_data_30"
    if not casp_dir.exists():
        print("ERROR: CASP12 data not found. Run with --casp12 first.")
        return

    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers sentencepiece")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: Generating embeddings on CPU will be very slow. GPU recommended.")

    # Load ProtT5
    print("Loading ProtT5 model...")
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    model = T5EncoderModel.from_pretrained(transformer_link)
    if device.type == "cuda":
        model = model.half()
    model = model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

    def get_emb(seq):
        """Get ProtT5 embedding for a single sequence."""
        seq_spaced = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        ids = tokenizer([seq_spaced], add_special_tokens=True, padding="longest", return_tensors="pt")
        input_ids = ids["input_ids"].to(device)
        attention_mask = ids["attention_mask"].to(device)
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)
        return result.last_hidden_state[0, : len(seq)].cpu()

    def cycle_seq(seq, shift):
        """Cycle-permute a sequence by shift positions."""
        return seq[-shift:] + seq[:-shift] if shift > 0 else seq[shift:] + seq[:shift]

    # Process all protein directories
    n_processed = 0
    for root, dirs, files in os.walk(casp_dir):
        if "seq.pt" not in files:
            continue
        root_path = Path(root)

        # Skip if already has embeddings
        if (root_path / "proT5_emb.pt").exists() and (root_path / "proT5_emb_cycle6.pt").exists():
            continue

        seq = torch.load(str(root_path / "seq.pt"), weights_only=False)
        if not isinstance(seq, str):
            continue

        try:
            # Wild-type embedding
            if not (root_path / "proT5_emb.pt").exists():
                emb = get_emb(seq)
                torch.save(emb, str(root_path / "proT5_emb.pt"))

            # Mutant embedding (2 random AA swaps)
            if not (root_path / "proT5_emb_mut.pt").exists():
                mix_idx = torch.randperm(len(seq))[:2]
                seq_mut = list(seq)
                seq_mut[mix_idx[0]], seq_mut[mix_idx[1]] = seq_mut[mix_idx[1]], seq_mut[mix_idx[0]]
                seq_mut = "".join(seq_mut)
                torch.save(seq_mut, str(root_path / "seq_mut.pt"))
                emb_mut = get_emb(seq_mut)
                torch.save(emb_mut, str(root_path / "proT5_emb_mut.pt"))

            # Cycle permutation embeddings
            cycle_shifts = {
                "proT5_emb_cycle1.pt": -1,
                "proT5_emb_cycle2.pt": 1,
                "proT5_emb_cycle3.pt": -2,
                "proT5_emb_cycle4.pt": -5,
                "proT5_emb_cycle5.pt": 2,
                "proT5_emb_cycle6.pt": 5,
            }
            for fname, shift in cycle_shifts.items():
                if not (root_path / fname).exists():
                    cycled = cycle_seq(seq, shift)
                    emb_cyc = get_emb(cycled)
                    torch.save(emb_cyc, str(root_path / fname))

        except Exception as e:
            print(f"  Error processing {root_path.name}: {e}")
            continue

        n_processed += 1
        if n_processed % 100 == 0:
            print(f"  Generated embeddings for {n_processed} proteins...")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Embedding generation complete: {n_processed} proteins processed.")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare DeepEF training data")
    parser.add_argument("--all", action="store_true", help="Download and prepare everything")
    parser.add_argument("--k50", action="store_true", help="Download K50/Tsuboyama mutation data from Zenodo")
    parser.add_argument("--process-k50", action="store_true", help="Process K50 data into training tensors")
    parser.add_argument("--casp12", action="store_true", help="Download and convert CASP12 from SidechainNet")
    parser.add_argument("--embeddings", action="store_true", help="Generate ProtT5 embeddings for CASP12")
    parser.add_argument("--thinning", type=int, default=100, help="SidechainNet thinning level (default: 100)")

    args = parser.parse_args()

    if not any([args.all, args.k50, args.process_k50, args.casp12, args.embeddings]):
        parser.print_help()
        print("\nExample: python download_data.py --all")
        return

    if args.all or args.k50:
        download_k50()

    if args.all or args.process_k50:
        process_k50()

    if args.all or args.casp12:
        download_casp12(thinning=args.thinning)

    if args.all or args.embeddings:
        generate_embeddings()

    print("\nDone! Data is ready in ./data/")
    print("  - K50 mutation data: ./data/Processed_K50_dG_datasets/")
    print("  - CASP12 pre-training: ./data/casp12_data_30/")


if __name__ == "__main__":
    main()
