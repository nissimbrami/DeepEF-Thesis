"""
Fast ProtT5 embedding generation using batched inference.
Processes one variant type at a time across all proteins for maximum batch efficiency.

Optimizations:
- torch.inference_mode() instead of no_grad (faster, less overhead)
- GPU cache cleanup after every batch to prevent slowdown
- Skips cycle5/cycle6 (unused in training — see train-2_5_light_att.py lines 193, 389)
- Adaptive batch size based on sequence length
"""

import os
import re
import gc
import time
import torch
from pathlib import Path
from transformers import T5EncoderModel, AutoTokenizer

DATA_DIR = Path("./data/casp12_data_30")
MAX_BATCH_SIZE = 24  # sequences per batch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load model
print("Loading ProtT5 model...")
model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
model = T5EncoderModel.from_pretrained(model_name)
if device.type == "cuda":
    model = model.half()
model = model.to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")


def prep_seq(seq):
    return " ".join(list(re.sub(r"[UZOB]", "X", seq)))


def get_embeddings_batch(sequences):
    """Get ProtT5 embeddings for a batch of sequences. Returns list of [seq_len, 1024] CPU tensors."""
    prepped = [prep_seq(s) for s in sequences]
    ids = tokenizer(prepped, add_special_tokens=True, padding="longest", return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    with torch.inference_mode():
        result = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = []
        for i, seq in enumerate(sequences):
            emb = result.last_hidden_state[i, :len(seq)].cpu().float()
            embeddings.append(emb)

    # Explicit cleanup to prevent GPU memory accumulation
    del result, input_ids, attention_mask, ids
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return embeddings


def cycle_seq(seq, shift):
    return seq[shift:] + seq[:shift]


def make_mutant(seq):
    mix_idx = torch.randperm(len(seq))[:2]
    seq_list = list(seq)
    seq_list[mix_idx[0]], seq_list[mix_idx[1]] = seq_list[mix_idx[1]], seq_list[mix_idx[0]]
    return "".join(seq_list)


# cycle5 (shift=2) and cycle6 (shift=5) are never used in training:
# train-2_5_light_att.py line 193 only cats Xcy1-4, and lossd_function only takes Ecy1-4
EMB_VARIANTS = [
    ("proT5_emb.pt", None),
    ("proT5_emb_mut.pt", "mut"),
    ("proT5_emb_cycle1.pt", -1),
    ("proT5_emb_cycle2.pt", 1),
    ("proT5_emb_cycle3.pt", -2),
    ("proT5_emb_cycle4.pt", -5),
]


def collect_proteins():
    """Collect all protein dirs with sequences."""
    proteins = []
    for root, dirs, files in os.walk(DATA_DIR):
        if "seq.pt" not in files:
            continue
        root_path = Path(root)
        seq = torch.load(str(root_path / "seq.pt"), weights_only=False)
        if isinstance(seq, str) and 10 <= len(seq) <= 600:
            proteins.append((root_path, seq))
    return proteins


def main():
    print("Collecting proteins...")
    proteins = collect_proteins()
    print(f"Found {len(proteins)} total proteins.")

    # Sort by length for efficient batching
    proteins.sort(key=lambda x: len(x[1]))

    t0 = time.time()

    for var_name, variant in EMB_VARIANTS:
        # Collect proteins needing this variant
        todo = []
        for pdir, seq in proteins:
            if (pdir / var_name).exists():
                continue
            if variant is None:
                s = seq
            elif variant == "mut":
                s = make_mutant(seq)
            else:
                s = cycle_seq(seq, variant)
            todo.append((pdir, seq, s))

        if not todo:
            print(f"{var_name}: all done, skipping")
            continue

        print(f"\n{var_name}: {len(todo)} sequences to process")
        var_t0 = time.time()

        idx = 0
        while idx < len(todo):
            # Adaptive batch size based on sequence length
            seq_len = len(todo[idx][2])
            if seq_len > 400:
                batch_size = 4
            elif seq_len > 300:
                batch_size = 8
            elif seq_len > 200:
                batch_size = 12
            elif seq_len > 100:
                batch_size = 18
            else:
                batch_size = MAX_BATCH_SIZE

            batch = todo[idx:idx + batch_size]
            seqs = [item[2] for item in batch]

            try:
                embs = get_embeddings_batch(seqs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "MPS" in str(e):
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    # Retry with half batch
                    half = max(1, len(seqs) // 2)
                    try:
                        embs = get_embeddings_batch(seqs[:half]) + get_embeddings_batch(seqs[half:])
                    except Exception:
                        embs = []
                        for s in seqs:
                            try:
                                embs.extend(get_embeddings_batch([s]))
                            except Exception:
                                embs.append(torch.zeros(len(s), 1024))
                else:
                    raise

            for (pdir, orig_seq, var_seq), emb in zip(batch, embs):
                torch.save(emb, str(pdir / var_name))
                if variant == "mut":
                    torch.save(var_seq, str(pdir / "seq_mut.pt"))

            # Free batch embeddings
            del embs, batch, seqs

            idx += batch_size
            done = min(idx, len(todo))
            if done % 500 < batch_size or done == len(todo):
                elapsed = time.time() - var_t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / rate / 60 if rate > 0 else 0
                print(f"  {done}/{len(todo)} ({elapsed:.0f}s, {rate:.1f} seq/s, ETA {eta:.0f}min)")

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed/60:.1f} minutes!")


if __name__ == "__main__":
    main()
