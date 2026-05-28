"""
Generate ESM-IF1 ENCODER per-residue structural representations for all proteins.

Unlike generate_esmif_scores.py (which only extracts scalar log-likelihood ratios),
this script extracts the FULL 512-dim encoder representation at every residue position.

These features capture the structural context of each residue — what amino acids would
be favorable at this position given the local backbone geometry and neighborhood.
This is the same signal ThermoMPNN gets from ProteinMPNN's decoder, but from ESM-IF1.

Output: data/MsDs/training_data/{protein}/esmif_enc.pt — tensor [seq_len, 512]

Key insight: ONE forward pass per protein gives [L, 512] features that can be used
for ALL mutations at ALL positions. No need to re-run per mutation.

Usage:
    cd /home/nissimb/workspace/DeepPEF
    source /home/nissimb/pytorch_env/bin/activate
    python data_creation/generate_esmif_encoder_features.py [--force]

Time: ~1 hour for 368 proteins (one forward pass per protein, ~0.5s each)
"""

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append('./')

TRAINING_DATA = './data/MsDs/training_data'


def generate_all(force=False):
    import esm
    from esm.inverse_folding.util import CoordBatchConverter

    print("=" * 60)
    print("Generating ESM-IF1 ENCODER features (512-dim per residue)")
    print("=" * 60)
    print()
    print("Loading ESM-IF1 model (142M params)...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().cuda()
    batch_converter = CoordBatchConverter(alphabet)
    print(f"  ESM-IF1 loaded on CUDA")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    proteins = sorted(os.listdir(TRAINING_DATA))
    total = len(proteins)
    generated = 0
    skipped = 0
    failed = []

    for idx, protein_name in enumerate(tqdm(proteins, desc="Generating ESM-IF1 encoder features")):
        out_path = os.path.join(TRAINING_DATA, protein_name, 'esmif_enc.pt')
        if os.path.exists(out_path) and not force:
            skipped += 1
            continue

        coords_path = os.path.join(TRAINING_DATA, protein_name, 'coords.pt')
        if not os.path.exists(coords_path):
            failed.append(protein_name)
            continue

        try:
            # Load coordinates [seq_len, 4, 3] (N, CA, C, CB)
            coords_tensor = torch.load(coords_path, weights_only=False)

            # ESM-IF1 encoder needs [L, 3, 3] (N, CA, C only — no CB)
            coords_ncc = coords_tensor[:, :3, :].numpy()  # [L, 3, 3]

            # Get encoder representation using batch_converter (handles device)
            batch = [(coords_ncc, None, None)]
            coords_batch, confidence, _, _, padding_mask = batch_converter(batch)
            # Move to CUDA
            coords_batch = coords_batch.cuda()
            confidence = confidence.cuda()
            padding_mask = padding_mask.cuda()

            with torch.no_grad():
                encoder_out = model.encoder.forward(
                    coords_batch, padding_mask, confidence,
                    return_all_hiddens=False)
                # encoder_out['encoder_out'][0] shape: [L+2, 1, 512] (includes BOS/EOS)
                # Remove BOS and EOS tokens
                rep = encoder_out['encoder_out'][0][1:-1, 0]  # [L, 512]

            # Verify shape matches protein length
            if rep.shape[0] != coords_tensor.shape[0]:
                print(f"  WARNING {protein_name}: rep shape {rep.shape[0]} != coords {coords_tensor.shape[0]}")
                # Trim to match
                min_len = min(rep.shape[0], coords_tensor.shape[0])
                rep = rep[:min_len]

            # Save as float16 to save disk space (512 * L * 2 bytes per protein)
            torch.save(rep.cpu().half(), out_path)
            generated += 1

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [{idx+1}/{total}] {protein_name}: shape {rep.shape}, "
                      f"range [{rep.min():.3f}, {rep.max():.3f}]")

        except Exception as e:
            print(f"  [{idx+1}/{total}] ERROR {protein_name}: {e}")
            failed.append(protein_name)

        # Clear CUDA cache periodically
        if (idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    print()
    print(f"Done. Generated={generated}, Skipped={skipped}, Failed={len(failed)}")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Regenerate all')
    args = parser.parse_args()
    generate_all(force=args.force)
