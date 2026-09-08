#!/usr/bin/env python3
"""P0 — re-score every eval CSV on the canonical basis, and flag any that is not.

WHY THIS RUNS FIRST
-------------------
The epoch audit found a_p = 0.740 was the MAXIMUM over six test-scored epochs, while the
validation argmax is e10. That is test-set peeking -- the practice this project faulted in
the prior work. TASKS.md fixes the canonical basis as:

    27 test proteins (2K5H excluded) | ddG metric | val-selected epoch ONLY
    -> pooled 0.5772 / oracle 0.7156 / gain +0.1384

Headline figures of 0.6382 and 0.6801 are in circulation and are NOT known to be on this
basis. Until every number sits on one basis, no comparison in this project means anything.

WHAT "CANONICAL EPOCH" MEANS
----------------------------
The argmax of VALIDATION ddG PCC, parsed from the training log. Never the argmax over
test-scored epochs. Scoring several epochs on test is acceptable as diagnostics; reporting
more than one of them, or the best of them, is not.

USAGE
    python canonical_rescore.py <eval_dir> <log_dir> [out.tsv]

GATE
    1. Every tag contributes exactly one canonical row.
    2. The canonical arm reproduces pooled 0.5772 (+/- 0.001).
    If either fails, stop and report -- do not build on the table.
"""
import os
import re
import sys
import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --------------------------------------------------------------------------------------
# Canonical basis. Change these ONLY with a recorded decision.
# --------------------------------------------------------------------------------------
EXCLUDE_PROTEINS = {"2K5H"}      # reference row is a mutant; see TWO_PROTEINS.md
MIN_MUTS_FOR_PP = 3              # score_runs.py convention: groups with < 3 are skipped
EXPECTED_N_PROTEINS = 27
EXPECTED_CANONICAL_POOLED = 0.5772
POOLED_TOLERANCE = 0.001

# Validation-PCC line in the training log. Widened to tolerate formatting drift; if this
# matches nothing, the script says so loudly rather than silently selecting epoch 0.
VAL_PCC_PATTERNS = [
    re.compile(r"epoch[=: ]+(\d+).*?val[^\n]*?ddg[_ ]?pcc[=: ]+(-?\d*\.?\d+)", re.I),
    re.compile(r"\[metrics\][^\n]*?epoch[=: ]+(\d+)[^\n]*?ddG[_ ]?PCC[=: ]+(-?\d*\.?\d+)", re.I),
]

EVAL_NAME_RE = re.compile(r"^abl_(.+)_e(\d+)\.csv$")


def val_argmax_epoch(train_log):
    """Return (epoch, val_pcc) with the highest VALIDATION ddG PCC, or (None, None).

    Ties break to the EARLIER epoch: with equal validation evidence, the less-trained model
    is the more conservative choice and the one less likely to have memorised.
    """
    if not os.path.exists(train_log):
        return None, None
    best_e, best_v = None, None
    n_matched = 0
    with open(train_log, errors="ignore") as fh:
        for line in fh:
            for pat in VAL_PCC_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                n_matched += 1
                e, v = int(m.group(1)), float(m.group(2))
                if best_v is None or v > best_v:
                    best_e, best_v = e, v
                break
    if n_matched == 0:
        print(f"WARN no validation PCC lines matched in {train_log} -- "
              f"canonical epoch unknown, NOT defaulting to anything",
              file=sys.stderr)
        return None, None
    return best_e, best_v


def _pp_correlations(df, true_col, pred_col):
    """Per-protein Pearson, skipping groups that are too small or degenerate."""
    vals, skipped = [], 0
    for _, g in df.groupby("protein"):
        if len(g) < MIN_MUTS_FOR_PP:
            skipped += 1
            continue
        if g[true_col].std(ddof=0) == 0 or g[pred_col].std(ddof=0) == 0:
            skipped += 1          # constant column -> Pearson undefined, not zero
            continue
        vals.append(pearsonr(g[true_col], g[pred_col])[0])
    return vals, skipped


def score_csv(path):
    """Canonical pooled / per-protein ddG PCC, plus the offset-removal oracle.

    The oracle subtracts each protein's mean error -- it requires the true labels and is
    therefore an upper bound, never an achievable target.
    """
    df = pd.read_csv(path)
    for col in ("protein", "ddG", "pred_ddG"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}'")

    df = df[~df["protein"].isin(EXCLUDE_PROTEINS)].copy()
    df = df.dropna(subset=["ddG", "pred_ddG"])

    pooled = pearsonr(df["ddG"], df["pred_ddG"])[0]
    pooled_scc = spearmanr(df["ddG"], df["pred_ddG"])[0]
    rmse = float(np.sqrt(((df["ddG"] - df["pred_ddG"]) ** 2).mean()))

    pp_vals, skipped = _pp_correlations(df, "ddG", "pred_ddG")
    pp = float(np.mean(pp_vals)) if pp_vals else float("nan")

    # Offset-removal oracle: centre predictions and truth within each protein.
    d = df.copy()
    d["pred_c"] = d.groupby("protein")["pred_ddG"].transform(lambda s: s - s.mean())
    d["true_c"] = d.groupby("protein")["ddG"].transform(lambda s: s - s.mean())
    oracle = pearsonr(d["true_c"], d["pred_c"])[0]

    return dict(
        pooled=pooled, pooled_scc=pooled_scc, rmse=rmse, pp=pp,
        oracle_offset_removed=oracle, gain=oracle - pooled,
        n_proteins=int(df["protein"].nunique()), n_mutations=int(len(df)),
        n_pp_groups=len(pp_vals), n_pp_skipped=skipped,
    )


def main(eval_dir, log_dir, out="results/CANONICAL.tsv"):
    csvs = sorted(glob.glob(os.path.join(eval_dir, "abl_*_e*.csv")))
    if not csvs:
        print(f"FAIL no abl_*_e*.csv under {eval_dir}", file=sys.stderr)
        return 2

    rows = []
    for csv in csvs:
        m = EVAL_NAME_RE.match(os.path.basename(csv))
        if not m:
            continue
        tag, epoch = m.group(1), int(m.group(2))

        # Log naming: try the common variants rather than assuming one.
        can_e = can_v = None
        for cand in (f"{tag}.log", f"{tag}.out", f"train_{tag}.log"):
            p = os.path.join(log_dir, cand)
            if os.path.exists(p):
                can_e, can_v = val_argmax_epoch(p)
                break

        try:
            s = score_csv(csv)
        except Exception as exc:                       # a bad CSV must not abort the sweep
            print(f"ERROR {csv}: {exc}", file=sys.stderr)
            continue

        rows.append(dict(tag=tag, epoch=epoch, canonical_epoch=can_e,
                         val_pcc=can_v, is_canonical=(can_e is not None and can_e == epoch),
                         **{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in s.items()}))

    df = pd.DataFrame(rows).sort_values(["tag", "epoch"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    # ---------------- gates ----------------
    problems = []

    for tag, g in df.groupby("tag"):
        n_can = int(g.is_canonical.sum())
        if n_can == 0:
            problems.append(
                f"{tag}: NO canonical row. {len(g)} test-scored epochs "
                f"{sorted(g.epoch)}; canonical epoch "
                f"{g.canonical_epoch.dropna().unique().tolist() or 'unknown'}")
        elif n_can > 1:
            problems.append(f"{tag}: {n_can} rows flagged canonical -- ambiguous")
        if len(g) > 1:
            best_ep = int(g.loc[g.pooled.idxmax(), "epoch"])
            can_ep = g.canonical_epoch.dropna().unique()
            if len(can_ep) and best_ep != int(can_ep[0]):
                problems.append(
                    f"{tag}: PEEKING RISK -- best TEST epoch e{best_ep} "
                    f"(pooled {g.pooled.max():.4f}) != canonical e{int(can_ep[0])} "
                    f"(pooled {g.loc[g.epoch == int(can_ep[0]), 'pooled'].iloc[0]:.4f}). "
                    f"Report the canonical one.")

    canon = df[df.is_canonical]
    for _, r in canon.iterrows():
        if r.n_proteins != EXPECTED_N_PROTEINS:
            problems.append(f"{r.tag}: {r.n_proteins} proteins, expected "
                            f"{EXPECTED_N_PROTEINS}")

    print("\n=== CANONICAL ROWS ===")
    cols = ["tag", "epoch", "val_pcc", "pooled", "pp", "oracle_offset_removed",
            "gain", "rmse", "n_proteins", "n_mutations"]
    print(canon[cols].to_string(index=False) if len(canon) else "(none)")

    print("\n=== BASIS CHECK ===")
    hit = canon[np.isclose(canon.pooled, EXPECTED_CANONICAL_POOLED, atol=POOLED_TOLERANCE)]
    if len(hit):
        print(f"PASS  reproduces the canonical pooled {EXPECTED_CANONICAL_POOLED} "
              f"in: {hit.tag.tolist()}")
    else:
        pooled_min = canon.pooled.min() if len(canon) else float('nan')
        pooled_max = canon.pooled.max() if len(canon) else float('nan')
        print(f"WARN  no canonical row reproduces {EXPECTED_CANONICAL_POOLED}; "
              f"canonical pooled spans {pooled_min:.4f}..{pooled_max:.4f}. "
              f"Either the basis moved or a run is missing -- resolve before proceeding.")

    if problems:
        print("\n=== PROBLEMS ===")
        for p in problems:
            print("  " + p)
        print(f"\n{len(problems)} problem(s). Fix before quoting any number.")
        return 1

    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
