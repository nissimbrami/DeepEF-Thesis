"""Reusable benchmark metrics for DeepEF evaluation (WS5).

Pure numpy/scipy/sklearn — no torch, no GPU. Provides the rigorous statistics a
top-venue paper needs: point estimates with bootstrap confidence intervals,
classification AUC for destabilizing-mutation detection, and an anti-symmetry
report for reverse-mutation (Ssym-style) analysis.

Convention: ddG values follow the S669/Ssym convention where DDG_dir is the
experimental direct-mutation stability change. Anti-symmetry compares a method's
direct and inverse predictions: an ideal physics-based predictor satisfies
ddG_inv = -ddG_dir, i.e. the bias delta = ddG_dir + ddG_inv ~ 0.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _clean(*arrays):
    """Stack arrays, drop any row with a NaN/inf in any column, return columns."""
    cols = [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    n = min(len(c) for c in cols)
    cols = [c[:n] for c in cols]
    stack = np.vstack(cols)
    good = np.all(np.isfinite(stack), axis=0)
    return [c[good] for c in cols]


def pearson(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) < 3 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def spearman(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) < 3:
        return float("nan")
    return float(spearmanr(y_true, y_pred)[0])


def rmse(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def mae(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def kabsch_rmsd(P, Q):
    """CA-CA RMSD after optimal rigid superposition (Kabsch). P,Q: [N,3] arrays (same ordering)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Qa = Qc @ R.T
    return float(np.sqrt(np.mean(np.sum((Pc - Qa) ** 2, axis=1))))


def bootstrap_ci(y_true, y_pred, stat_fn, n_boot=10000, ci=95.0, seed=42):
    """Paired bootstrap CI for a (y_true, y_pred) statistic.

    Returns (point_estimate, lo, hi). Resamples mutation pairs with replacement.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    n = len(y_true)
    point = stat_fn(y_true, y_pred)
    if n < 3:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = stat_fn(y_true[idx], y_pred[idx])
    stats = stats[np.isfinite(stats)]
    lo = float(np.percentile(stats, (100 - ci) / 2))
    hi = float(np.percentile(stats, 100 - (100 - ci) / 2))
    return point, lo, hi


def regression_report(y_true, y_pred, n_boot=10000, ci=95.0, seed=42):
    """PCC/SCC/RMSE/MAE, each with a bootstrap CI. Returns a nested dict."""
    out = {}
    for name, fn in (("pcc", pearson), ("scc", spearman), ("rmse", rmse), ("mae", mae)):
        point, lo, hi = bootstrap_ci(y_true, y_pred, fn, n_boot=n_boot, ci=ci, seed=seed)
        out[name] = {"value": point, "lo": lo, "hi": hi}
    out["n"] = int(len(_clean(y_true, y_pred)[0]))
    return out


def classification_auc(ddg_exp, ddg_pred, threshold=0.0, destabilizing=True, n_boot=10000, seed=42):
    """AUC for detecting (de)stabilizing mutations from predicted ddG.

    destabilizing=True labels mutations with experimental ddG_dir > threshold as
    positive (uses the Ssym convention where positive ddG_dir = destabilizing).
    Returns {'auc', 'lo', 'hi', 'n_pos', 'n_neg'}.
    """
    from sklearn.metrics import roc_auc_score

    ddg_exp, ddg_pred = _clean(ddg_exp, ddg_pred)
    label = (ddg_exp > threshold).astype(int)
    if not destabilizing:
        label = 1 - label
    score = ddg_pred if destabilizing else -ddg_pred
    n_pos, n_neg = int(label.sum()), int((1 - label).sum())
    if n_pos < 2 or n_neg < 2:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_pos": n_pos, "n_neg": n_neg}
    auc = float(roc_auc_score(label, score))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(label)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if label[idx].sum() < 2 or (1 - label[idx]).sum() < 2:
            continue
        boots.append(roc_auc_score(label[idx], score[idx]))
    lo = float(np.percentile(boots, 2.5)) if boots else float("nan")
    hi = float(np.percentile(boots, 97.5)) if boots else float("nan")
    return {"auc": auc, "lo": lo, "hi": hi, "n_pos": n_pos, "n_neg": n_neg}


def antisymmetry_report(ddg_dir, ddg_inv):
    """Reverse-mutation anti-symmetry (Ssym).

    For an ideal predictor ddG_inv = -ddG_dir, so the per-mutation bias
    delta = ddG_dir + ddG_inv has mean 0. Returns the mean/std bias, the
    correlation between direct and inverse predictions (ideal = -1), and the
    correlation between ddG_dir and -ddG_inv (ideal = +1).
    """
    ddg_dir, ddg_inv = _clean(ddg_dir, ddg_inv)
    delta = ddg_dir + ddg_inv
    return {
        "n": int(len(ddg_dir)),
        "bias_mean": float(np.mean(delta)) if len(delta) else float("nan"),
        "bias_std": float(np.std(delta)) if len(delta) else float("nan"),
        "r_dir_inv": pearson(ddg_dir, ddg_inv),          # ideal -1
        "r_dir_neginv": pearson(ddg_dir, -ddg_inv),      # ideal +1
    }


def format_report(name, rep):
    """One-line human-readable summary of a regression_report dict."""
    def f(k):
        d = rep[k]
        return f"{d['value']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]"
    return (f"{name:<22} n={rep['n']:>5}  PCC={f('pcc')}  SCC={f('scc')}  "
            f"RMSE={f('rmse')}  MAE={f('mae')}")
