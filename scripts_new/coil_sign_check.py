#!/usr/bin/env python3
"""T-E4 — is the coil's failure a SCALE error or a SHAPE error?

WHY
---
Ten nu values all scored worse than no coil at all, and corr(b_p, length) grew
monotonically with nu (-0.215 -> -0.507). Two explanations remain, and they call for
opposite fixes:

  SCALE  the coil distances are far larger than the folded ones, so the Gaussian kernel
         exp(coef * d^2) returns near-zero everywhere and the unfolded state is FLATTENED.
         The model then sees an almost featureless reference. Fix: refit b, not nu.

  SHAPE  the kernel block is comparable in magnitude to the baseline, so the coil is
         producing a genuine but wrong geometry. Then the ten-nu rejection is final.

This distinguishes them without training and without a GPU.

WHAT IT MEASURES
    - folded CA-CA distance statistics (the physical scale the kernel was tuned for)
    - the post-kernel unfolded block under the tridiagonal baseline
    - the same under the coil at several nu, and at fitted vs fixed b
    - the ratio of coil block to baseline block

USAGE
    python coil_sign_check.py [--proteins P1 P2 ...] [--n 5]

    Run from the repo root so that `train_utils` and `model.model_cfg` import.

READ THE OUTPUT LIKE THIS
    ratio_to_baseline << 1   -> SCALE problem: the coil flattened the reference.
    ratio_to_baseline ~ 1    -> SHAPE problem: the coil is not a scale artefact.
    frac_nonzero collapsing  -> the kernel has saturated; distances are out of its range.
"""
import os
import sys

import numpy as np
import torch

# --------------------------------------------------------------------------------------
# Repo imports. Must be run from the repository root.
# --------------------------------------------------------------------------------------
try:
    from model.model_cfg import CFG
    from train_utils import get_unfolded_graph, get_dist_matrix, get_graph
except ImportError as exc:
    print(f"FAIL cannot import repo modules ({exc}).\n"
          f"     Run from the repository root, e.g.\n"
          f"     cd ~/DeepPEF && python cluster_run/code/coil_sign_check.py",
          file=sys.stderr)
    raise

# get_dist_matrix returns [N, N, 16] over atom pairs (N, CA, C, CB) x (N, CA, C, CB).
# CA is atom index 1, so the CA-CA channel is 1*4 + 1 = 5.
CA_CA_CHANNEL = 5          # [VERIFY] confirm against get_dist_matrix's ordering
DIST_BLOCK = 16            # the D block width in the node feature vector


def _block_stats(graph, name):
    """Summarise the post-kernel distance block of a node-feature tensor."""
    blk = graph[:, :DIST_BLOCK]
    finite = blk[torch.isfinite(blk)]
    return dict(
        name=name,
        mean=float(finite.mean()) if finite.numel() else float("nan"),
        std=float(finite.std()) if finite.numel() else float("nan"),
        max=float(finite.max()) if finite.numel() else float("nan"),
        frac_nonzero=float((blk.abs() > 1e-8).float().mean()),
    )


def probe_one(x, one_hot, emb, mask, label,
              nu_values=(0.5, 0.588, 0.62), b_modes=("fixed", "fitted")):
    """Compare the baseline unfolded block against the coil at several nu and b modes."""
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")

    n = int(mask.sum().item())

    # ---- folded reference: the physical scale the Gaussian kernel was tuned for ----
    D = get_dist_matrix(x)
    ca = D[:n, :n, CA_CA_CHANNEL]
    off = ca[~torch.eye(n, dtype=torch.bool, device=ca.device)]
    print(f"  folded CA-CA (A):  median {off.median():.2f}   "
          f"p95 {off.kthvalue(int(0.95 * off.numel()))[0]:.2f}   max {off.max():.2f}")
    print(f"  chain length: {n} residues")

    # ---- baseline unfolded ----
    prev_flory = getattr(CFG, "flory_unfolded", False)
    prev_nu = getattr(CFG, "flory_nu", 0.5)
    prev_b = getattr(CFG, "coil_b", "fitted")

    CFG.flory_unfolded = False
    base = get_unfolded_graph(x, one_hot, emb, mask)
    bs = _block_stats(base, "baseline (tridiagonal)")
    print(f"\n  {'condition':<28} {'mean':>10} {'std':>10} {'max':>10} "
          f"{'nonzero':>9} {'ratio':>8}")
    print(f"  {bs['name']:<28} {bs['mean']:>10.5f} {bs['std']:>10.5f} "
          f"{bs['max']:>10.5f} {bs['frac_nonzero']:>9.3f} {'--':>8}")

    rows = [bs]
    for b_mode in b_modes:
        for nu in nu_values:
            CFG.flory_unfolded = True
            CFG.flory_nu = nu
            CFG.coil_b = b_mode                        # [VERIFY] attribute name on CFG
            try:
                coil = get_unfolded_graph(x, one_hot, emb, mask)
            except Exception as exc:
                print(f"  ERROR nu={nu} b={b_mode}: {exc}")
                continue
            cs = _block_stats(coil, f"coil nu={nu} b={b_mode}")
            ratio = cs["mean"] / bs["mean"] if bs["mean"] else float("nan")
            cs["ratio_to_baseline"] = ratio
            rows.append(cs)
            print(f"  {cs['name']:<28} {cs['mean']:>10.5f} {cs['std']:>10.5f} "
                  f"{cs['max']:>10.5f} {cs['frac_nonzero']:>9.3f} {ratio:>8.3f}")

    CFG.flory_unfolded, CFG.flory_nu, CFG.coil_b = prev_flory, prev_nu, prev_b
    return rows


def diagnose(all_rows):
    """Turn the ratios into the one verdict this script exists to produce."""
    coil = [r for r in all_rows if "ratio_to_baseline" in r
            and np.isfinite(r["ratio_to_baseline"])]
    if not coil:
        print("\nNo coil rows to diagnose.")
        return

    ratios = np.array([r["ratio_to_baseline"] for r in coil])
    med = float(np.median(ratios))

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"  median coil/baseline block ratio: {med:.3f}")

    if med < 0.5:
        print("  SCALE PROBLEM. The coil block is much smaller than the baseline: the")
        print("  distances have run past the Gaussian kernel's useful range and the")
        print("  unfolded reference has been flattened toward a constant.")
        print("  ACTION: refit b so the coil's distance distribution overlaps the folded")
        print("  one, then re-test. Do NOT sweep nu again -- nu is not the free parameter")
        print("  that fixes a scale mismatch.")
    elif med > 2.0:
        print("  INVERSE SCALE PROBLEM. The coil block is much larger than the baseline:")
        print("  distances are too small and the kernel is saturating near 1, so the")
        print("  reference is again nearly constant, from the other side.")
        print("  ACTION: refit b upward.")
    else:
        print("  SHAPE, not scale. The coil block is comparable in magnitude to the")
        print("  baseline, so the ten-nu rejection was not a scale artefact and stands")
        print("  as a genuine negative result.")
        print("  ACTION: close the distance-map replacement. If the reference-state")
        print("  programme continues, it must be via the ENSEMBLE (T-E1), which changes")
        print("  what is averaged rather than what the mean distance is.")

    frac = np.array([r["frac_nonzero"] for r in coil])
    if frac.min() < 0.05:
        print("\n  WARN some coil configurations produce a nearly all-zero block. That is")
        print("  a degenerate reference state, and any energy computed from it is")
        print("  meaningless rather than merely wrong.")


def main(argv):
    n_proteins = 5
    if "--n" in argv:
        n_proteins = int(argv[argv.index("--n") + 1])

    # [VERIFY] Replace with the project's own loader. Kept deliberately explicit so a
    # wrong dataset cannot be used silently.
    try:
        from new_dataset import MSDataset
    except ImportError:
        print("FAIL cannot import MSDataset. Point this at the dataset loader in use and\n"
              "     supply x/one_hot/emb/mask for a handful of WT proteins.",
              file=sys.stderr)
        return 2

    ds = MSDataset(train=False, one_mut=True, dg_ml=True)   # [VERIFY] constructor args
    all_rows = []
    for i in range(min(n_proteins, len(ds))):
        b = ds[i]
        x = b["coords"].squeeze()
        one_hot = b["one_hot"][0] if b["one_hot"].dim() == 3 else b["one_hot"]
        emb = b["prott5"][0] if b["prott5"].dim() == 3 else b["prott5"]
        mask = b["masks"].squeeze()
        label = b.get("name", f"protein[{i}]")
        all_rows += probe_one(x, one_hot, emb, mask, label)

    diagnose(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
