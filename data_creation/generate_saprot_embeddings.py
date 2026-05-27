"""
Generate SaProt 650M embeddings (structure-aware) for all MsDs proteins.

Uses Foldseek 3Di tokens already generated at data/foldseek_3di/3di_tokens.tsv.
Saves saprot_wt.pt ([seq_len, 1280], fp16) alongside emb.pt in each protein dir.

SaProt input format: interleaved AA+3Di tokens, e.g. "MdEvVpQpLdVyQdYaKv"
Each bigram (e.g. "Md") is ONE token in SaProt's vocabulary.
"#" is used for low-pLDDT regions (plddt < 70).

Usage:
    cd /home/nissimb/workspace/DeepPEF
    source /home/nissimb/pytorch_env/bin/activate
    python data_creation/generate_saprot_embeddings.py [--force]
"""

import os
import sys
import torch
import argparse

sys.path.append('./')

TRAINING_DATA = './data/MsDs/training_data'
FOLDSEEK_TSV = './data/foldseek_3di/3di_tokens.tsv'
SAPROT_MODEL = 'westlake-repl/SaProt_650M_AF2'


def load_3di_map(tsv_path):
    """Parse Foldseek TSV → dict {protein_name: (aa_seq, 3di_seq)}."""
    mapping = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            name_raw, aa_seq, struc_seq = parts[0], parts[1], parts[2]
            # name_raw looks like "1A0N.pdb_A" → strip chain suffix, remove .pdb
            protein_name = name_raw.split('.pdb')[0]
            if protein_name not in mapping:  # take first chain if multiple
                mapping[protein_name] = (aa_seq, struc_seq)
    return mapping


def make_sa_sequence(aa_seq, struc_seq):
    """Interleave amino acid and 3Di tokens for SaProt input.

    Each residue becomes a bigram: AA_char + 3Di_char.
    Example: aa='MKQ', 3di='dpa' → 'MdKpQa'
    SaProt tokenizer treats each bigram as one token.
    """
    return ''.join(a + s for a, s in zip(aa_seq, struc_seq))


def generate_all(force=False):
    from transformers import EsmTokenizer, EsmModel

    print(f"Loading SaProt model from {SAPROT_MODEL}...")
    tokenizer = EsmTokenizer.from_pretrained(SAPROT_MODEL)
    model = EsmModel.from_pretrained(SAPROT_MODEL)
    model = model.eval().half().cuda()
    print("SaProt loaded.")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    print(f"Loading 3Di token map from {FOLDSEEK_TSV}...")
    di_map = load_3di_map(FOLDSEEK_TSV)
    print(f"  {len(di_map)} proteins with 3Di tokens.")

    proteins = sorted(os.listdir(TRAINING_DATA))
    total = len(proteins)
    generated = 0
    skipped = 0
    failed = []

    # Validate tokenizer on first protein to catch issues early
    first_protein = None
    for p in proteins:
        if p in di_map:
            first_protein = p
            break
    if first_protein:
        aa_test, struc_test = di_map[first_protein]
        sa_test = make_sa_sequence(aa_test[:10], struc_test[:10])
        test_tokens = tokenizer.tokenize(sa_test)
        test_inputs = tokenizer(sa_test, return_tensors='pt', add_special_tokens=True)
        n_input_ids = test_inputs['input_ids'].shape[1]
        print(f"  Tokenizer validation: '{sa_test[:20]}...' -> {len(test_tokens)} tokens, input_ids shape: {test_inputs['input_ids'].shape}")
        # Expected: 10 tokens for 10 residues, input_ids [1, 12] (10 + BOS + EOS)
        if len(test_tokens) != 10:
            print(f"  WARNING: Expected 10 tokens for 10 residues but got {len(test_tokens)}!")
            print(f"  Tokens: {test_tokens}")
            print(f"  This means the tokenizer is NOT splitting into bigrams correctly.")
            print(f"  Trying space-separated format...")
            # SaProt tokenizer may need space-separated bigrams
            sa_test_spaced = ' '.join(a + s for a, s in zip(aa_test[:10], struc_test[:10]))
            test_tokens_spaced = tokenizer.tokenize(sa_test_spaced)
            print(f"  Space-separated: '{sa_test_spaced[:30]}...' -> {len(test_tokens_spaced)} tokens")
            if len(test_tokens_spaced) == 10:
                print(f"  SUCCESS: Space-separated format works! Using this format.")
                USE_SPACES = True
            else:
                print(f"  Still wrong. Checking if tokenizer needs different input format...")
                USE_SPACES = False
        else:
            print(f"  Tokenizer OK: correctly splits into bigram tokens.")
            USE_SPACES = False
    else:
        USE_SPACES = False

    for idx, protein_name in enumerate(proteins):
        out_path = os.path.join(TRAINING_DATA, protein_name, 'saprot_wt.pt')
        if os.path.exists(out_path) and not force:
            skipped += 1
            continue

        if protein_name not in di_map:
            if idx < 5:
                print(f"  [{idx+1}/{total}] SKIP {protein_name}: no 3Di tokens")
            failed.append(protein_name)
            continue

        aa_seq, struc_seq = di_map[protein_name]
        if len(aa_seq) != len(struc_seq):
            print(f"  [{idx+1}/{total}] SKIP {protein_name}: length mismatch aa={len(aa_seq)} 3di={len(struc_seq)}")
            failed.append(protein_name)
            continue

        seq_len = len(aa_seq)
        # SaProt max length is 1022 residues (like ESM-2 650M)
        if seq_len > 1022:
            aa_seq = aa_seq[:1022]
            struc_seq = struc_seq[:1022]
            seq_len = 1022

        # Build the SA-sequence in the correct format
        if USE_SPACES:
            sa_seq = ' '.join(a + s for a, s in zip(aa_seq, struc_seq))
        else:
            sa_seq = make_sa_sequence(aa_seq, struc_seq)

        try:
            inputs = tokenizer(sa_seq, return_tensors='pt', add_special_tokens=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}

            # Verify token count matches expected
            n_tokens = inputs['input_ids'].shape[1] - 2  # subtract BOS + EOS
            if n_tokens != seq_len:
                # Fallback: use actual token count for slicing
                if idx < 5:
                    print(f"  [{idx+1}/{total}] WARN {protein_name}: expected {seq_len} tokens, got {n_tokens}")

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=False)

            # Extract per-residue embeddings: skip BOS (pos 0), take n_tokens positions
            # Use min(seq_len, n_tokens) to be safe
            actual_len = min(seq_len, n_tokens)
            emb = outputs.last_hidden_state[0, 1:1+actual_len, :].cpu().half()

            # Verify shape
            if emb.shape[0] != seq_len and idx < 5:
                print(f"  [{idx+1}/{total}] WARN {protein_name}: emb shape {emb.shape}, expected [{seq_len}, 1280]")

            torch.save(emb, out_path)
            generated += 1

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [{idx+1}/{total}] {protein_name}: {emb.shape} saved")

        except Exception as e:
            print(f"  [{idx+1}/{total}] ERROR {protein_name}: {e}")
            failed.append(protein_name)

        # Clear cache every 50 proteins
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    print(f"\nDone. Generated={generated}, Skipped(exist)={skipped}, Failed={len(failed)}")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Regenerate even if saprot_wt.pt exists')
    args = parser.parse_args()
    generate_all(force=args.force)
