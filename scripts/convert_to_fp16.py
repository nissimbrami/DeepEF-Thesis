"""
Convert all ProtT5 embedding tensors in casp12_data_30 from float32 to float16 in-place.

Since the float32 originals are already backed up on HF, this modifies the local files directly.

Usage:
    python scripts/convert_to_fp16.py
    python scripts/convert_to_fp16.py --workers 8 --dry-run
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent.parent / "data" / "casp12_data_30"

EMB_PATTERNS = [
    "proT5_emb.pt",
    "proT5_emb_mut.pt",
    "proT5_emb_cycle1.pt",
    "proT5_emb_cycle2.pt",
    "proT5_emb_cycle3.pt",
    "proT5_emb_cycle4.pt",
]


def convert_file(path: Path, dry_run: bool) -> tuple[str, bool, str]:
    """Convert a single .pt file to float16 in-place. Returns (path, changed, msg)."""
    try:
        t = torch.load(path, weights_only=True)
        if t.dtype == torch.float16:
            return str(path), False, "already fp16"
        if t.dtype != torch.float32:
            return str(path), False, f"skipped ({t.dtype})"
        if not dry_run:
            torch.save(t.half(), path)
        return str(path), True, f"float32→float16, shape={list(t.shape)}"
    except Exception as e:
        return str(path), False, f"ERROR: {e}"


def collect_files(data_dir: Path) -> list[Path]:
    files = []
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for protein_dir in split_dir.iterdir():
            if not protein_dir.is_dir():
                continue
            for name in EMB_PATTERNS:
                f = protein_dir / name
                if f.exists():
                    files.append(f)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Don't write, just report")
    args = parser.parse_args()

    print(f"Scanning {DATA_DIR} ...")
    files = collect_files(DATA_DIR)
    print(f"Found {len(files)} embedding files across all splits.")

    if args.dry_run:
        print("DRY RUN — no files will be modified.\n")

    converted = 0
    skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_file, f, args.dry_run): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            path, changed, msg = future.result()
            if "ERROR" in msg:
                errors += 1
                print(f"[{i}/{len(files)}] {msg} — {path}")
            elif changed:
                converted += 1
            else:
                skipped += 1

            if i % 5000 == 0 or i == len(files):
                print(f"Progress: {i}/{len(files)} | converted={converted} skipped={skipped} errors={errors}")

    print(f"\nDone. Converted={converted}, Already fp16={skipped}, Errors={errors}")
    if not args.dry_run and converted > 0:
        print("All embedding files are now float16. Run the upload script to sync to HF.")


if __name__ == "__main__":
    main()
