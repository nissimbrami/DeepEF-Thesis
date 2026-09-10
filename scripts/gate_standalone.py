"""GATE: measure a feature STANDALONE, on CPU, before any GPU arm is submitted.

WHY THIS EXISTS
    Of ~20 levers tested, one cleared significance. A null from the full model is
    UNINTERPRETABLE: it cannot separate "no information" / "model failed to exploit it" /
    "silently inert". Five silent no-ops have been found; the last burned ~32 GPU-hours.
    None of them needed a GPU to detect.

WHAT IT IS
    A RANKER, NOT A BLOCKER. It scores and orders; it never refuses a submission. A low
    tier plus a mechanistic argument is a valid reason to run an arm. Exit code is always
    0 in ranking mode -- a ranker that exits non-zero becomes a blocker the moment someone
    puts it in a submit script.

THE THREE QUESTIONS
    Q1  state-dependence: does the block differ folded vs unfolded? A per-residue block
        identical in both contributes EXACTLY ZERO to dG = E_f - E_u.
    Q2  residual explanation (decisive): does the feature explain the model's WITHIN-protein
        residual, leave-one-protein-out? (b_p is 30.2% of residual variance and is
        unpredictable by ten methods; a random split would let the model memorise it.)
    Q3  linear absorption: is the feature already reachable by a linear layer?

ANTI-OVERFITTING RULE
    Calibration is four points (W12/W15/W5/W6). If it fails, change the DESIGN once, derived
    from mechanism. Never tune thresholds until four known cases order correctly -- that
    produces a gate which has memorised four numbers.
"""
import os, sys, re, json, argparse, warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/nissimb/DeepPEF")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import pandas as pd
import torch

REPO = "/home/nissimb/DeepPEF"
TD   = os.path.join(REPO, "data/Processed_K50_dG_datasets/training_data")
MUT  = os.path.join(REPO, "data/Processed_K50_dG_datasets/mutation_datasets")
TMPN = os.path.join(REPO, "data/ThermoMPNN/mega_test.csv")
EVAL = os.path.join(REPO, "eval_results")

# the 6 same-config control seeds; residual correlates 0.9696 across them
CONTROL_SEEDS = ["calib_ctrl_repro2_e14", "sigma_seed1_e13", "sigma_seed2_e10",
                 "sigma_seed3_e13", "sigma_seed4_e14", "sigma_seed42_e9"]

AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,
                   1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL = dict(zip(AA, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,
                    162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHG = dict(zip(AA, [0,0,-1,-1,0,0,0.1,0,1,0,0,0,0,0,1,0,0,0,0,0]))

_ok = True
_fails = 0
def check(name, cond, extra=""):
    global _ok, _fails
    if not cond: _fails += 1
    _ok = _ok and bool(cond)
    print("%-56s %s %s" % (name, "PASS" if cond else "FAIL", extra))
    return bool(cond)

_MISSING = object()

@contextmanager
def cfg_flags(CFG, **flags):
    """Set CFG levers and RESTORE. CFG is a CLASS -> global process state.
    Restoring an absent attribute as False is NOT equivalent to deleting it."""
    old = {k: getattr(CFG, k, _MISSING) for k in flags}
    try:
        for k, v in flags.items(): setattr(CFG, k, v)
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                if hasattr(CFG, k): delattr(CFG, k)
            else:
                setattr(CFG, k, v)


@dataclass
class ProteinCtx:
    """Per-protein tensors. one_hot is sliced to 20 columns at THIS chokepoint only."""
    P: str
    coords: torch.Tensor        # [N,4,3]
    one_hot_all: torch.Tensor   # [V,N,20]  -- already sliced
    mask: torch.Tensor          # [N]
    N: int

    @staticmethod
    def load(P):
        x  = torch.load(os.path.join(TD, P, "coords_tensor.pt"), map_location="cpu").float()
        oh = torch.load(os.path.join(TD, P, "one_hot_encodings.pt"), map_location="cpu").float()
        mk = torch.load(os.path.join(TD, P, "mask_tensor.pt"), map_location="cpu").float()
        N = x.shape[0]
        if oh.dim() == 3:
            assert oh.shape[2] in (20, 21), "unexpected one_hot width %d" % oh.shape[2]
            if oh.shape[2] == 21:
                # the 21st column must be all-zero; if a future dataset puts something
                # there we must fail loudly rather than silently drop it
                assert float(oh[:, :, 20].abs().sum()) == 0.0, \
                    "one_hot column 21 is NOT all-zero for %s -- refusing to drop it" % P
            oh = oh[:, :, :20].contiguous()
        return ProteinCtx(P=P, coords=x, one_hot_all=oh, mask=mk.reshape(-1)[:N], N=N)

    def wt(self):
        return self.one_hot_all[0][:self.N].contiguous()

    def mutant(self, pos, aa):
        """wild-type one_hot with row `pos` replaced by amino acid `aa`."""
        o = self.wt().clone()
        o[pos] = 0.0
        o[pos, AA.index(aa)] = 1.0
        return o

# ======================================================================================
# The exact join. Verified: 27/27 proteins, 27,189 rows, max|deltaG diff| 2.382e-07.
# Do NOT join on rounded deltaG -- that key is not unique (81 dups in 3DKM), loses 273 rows.
# ======================================================================================

def loader_filter(P, tm_names):
    """Replay Megascale-fineTuning/evaluate.py:195-235 so row i <-> eval variant_idx == i."""
    m = pd.read_csv(os.path.join(MUT, "%s.csv" % P))
    m = m[~m["mut_type"].str.contains("ins|del")].reset_index(drop=True)
    dg = torch.load(os.path.join(TD, P, "deltaG.pt"), weights_only=True).cpu().numpy().ravel()
    m["dg"] = dg[:len(m)]
    idx  = set(m.index)
    idx -= set(m[m["ddG_ML"] == "-"].index)
    idx -= set(m[m["mut_type"].str.contains(":")].index)
    idx -= set(m[~m["name"].isin(tm_names)].index)
    return m.loc[sorted(idx)].reset_index(drop=True)


def build_index(eval_tag="abl_coiledge_dg_s42_e14", max_muts=None, quiet=False):
    """One row per scorable mutation: protein, variant_idx, pos, wt_aa, mut_aa.

    Parses mut_type, NEVER name: for 1A0N only 2 of 2961 names match the naive regex.
    """
    tm = pd.read_csv(TMPN)
    tmn = set(tm["name"])
    ev = pd.read_csv(os.path.join(EVAL, "%s.csv" % eval_tag))
    ev = ev[ev.protein != "2K5H"]
    assert "variant_idx" in ev.columns, "%s is schema B (no variant_idx)" % eval_tag
    rows, maxdiff, nprot = [], 0.0, 0
    for P in sorted(ev.protein.unique()):
        sub = loader_filter(P, tmn)
        e = ev[ev.protein == P].sort_values("variant_idx").reset_index(drop=True)
        if len(sub) != len(e):
            raise RuntimeError("JOIN MISMATCH %s: filter %d vs eval %d" % (P, len(sub), len(e)))
        maxdiff = max(maxdiff, float(np.abs(sub.dg.values - e.deltaG.values).max()))
        nprot += 1
        keep = 0
        for i, mt in enumerate(sub.mut_type.astype(str)):
            g = re.match(r"^([A-Z])(\d+)([A-Z])$", mt)
            if not g:
                continue                      # the 5 wt rows per protein
            pos = int(g.group(2)) - 1
            rows.append((P, int(e.variant_idx.values[i]), pos, g.group(1), g.group(3)))
            keep += 1
            if max_muts and keep >= max_muts:
                break
    IDX = pd.DataFrame(rows, columns=["protein", "variant_idx", "pos", "wt_aa", "mut_aa"])
    if not quiet:
        print("join: %d proteins, %d scorable mutations, max|deltaG diff| %.3e"
              % (nprot, len(IDX), maxdiff))
    return IDX, nprot, maxdiff


def residual_table(IDX, ref_tag="abl_coiledge_dg_s42_e14"):
    """Within-protein residual per mutation, one column per control seed.

    y = r - mean_p(r): drops the 30.2% b_p component by construction, so the gate neither
    cheats on it nor is penalised for the part nothing can predict.

    ALL SIX control seeds are schema B (no variant_idx). Their rows are the same content
    in a different order, so the mapping is recovered by sorting on (protein, deltaG)
    against a schema-A reference -- NOT by rounding deltaG, whose key is not unique.
    """
    ref = pd.read_csv(os.path.join(EVAL, "%s.csv" % ref_tag))
    ref = ref[ref.protein != "2K5H"].copy()
    ref["_ord"] = ref.groupby("protein")["deltaG"].rank(method="first").astype(int)
    ref_key = ref.set_index(["protein", "variant_idx"])["_ord"]
    want = ref_key.reindex(pd.MultiIndex.from_arrays(
        [IDX.protein.values, IDX.variant_idx.values])).values      # rank within protein

    out = {}
    for tag in CONTROL_SEEDS:
        f = os.path.join(EVAL, "abl_%s.csv" % tag)
        if not os.path.exists(f):
            continue
        ev = pd.read_csv(f)
        ev = ev[ev.protein != "2K5H"].copy()
        ev["_ord"] = ev.groupby("protein")["deltaG"].rank(method="first").astype(int)
        ev["_r"] = ev.pred_ddG - ev.ddG
        m = ev.set_index(["protein", "_ord"])["_r"]
        vals = m.reindex(pd.MultiIndex.from_arrays(
            [IDX.protein.values, want])).values
        s = pd.Series(vals)
        out[tag] = (s - s.groupby(IDX.protein.values).transform("mean")).values
    return pd.DataFrame(out)


# ======================================================================================
# Feature registry. Three kinds:
#   builder    -- wraps a live train_utils function (W5/W12/W15/W6/W11)
#   standalone -- pure function of coords+sequence; features NOT yet in the model
#   proxy      -- for internal representations; returns per-mutation features directly
# ======================================================================================

@dataclass
class FeatureSpec:
    name: str
    kind: str
    cfg_flags: Dict[str, object] = field(default_factory=dict)
    build: Optional[Callable] = None        # (ctx, one_hot, folded) -> [N,k] or None
    per_mut: Optional[Callable] = None      # proxy only: (ctx, IDXsub) -> (X_add, X_prod)
    state_dependent: bool = True
    notes: str = ""

REGISTRY: Dict[str, FeatureSpec] = {}
def register(spec):
    REGISTRY[spec.name] = spec
    return spec


def _builder_adapter(fn, takes_one_hot=True, takes_folded=True, takes_x=True):
    """Dispatch EXPLICITLY on signature. _desc_or_none takes no folded and no x;
    _lig_or_none takes no one_hot. Using *args here would silently mis-order them."""
    def go(ctx, one_hot, folded):
        if not takes_x:
            return fn(one_hot)                                  # _desc_or_none(one_hot)
        if not takes_one_hot:
            return fn(ctx.coords, ctx.mask, folded)              # _lig_or_none(x, mask, folded)
        if not takes_folded:
            return fn(ctx.coords, one_hot, ctx.mask)
        return fn(ctx.coords, one_hot, ctx.mask, folded)
    return go


def register_builders():
    from train_utils import (_solv_or_none, _sidechain_or_none, _w15_or_none, _desc_or_none)
    register(FeatureSpec("W5", "builder", {"burial_features": True},
                         _builder_adapter(_solv_or_none),
                         notes="bur, hyd, bur*hyd"))
    register(FeatureSpec("W12", "builder", {"sidechain_features": True},
                         _builder_adapter(_sidechain_or_none),
                         notes="side-chain chemistry x geometry"))
    register(FeatureSpec("W15", "builder", {"w15_features": True},
                         _builder_adapter(_w15_or_none),
                         notes="reach*env packing + clash"))
    register(FeatureSpec("W6_pca16", "builder", {"aa_descriptors": "mordred_pca16"},
                         _builder_adapter(_desc_or_none, takes_folded=False, takes_x=False),
                         state_dependent=False,
                         notes="state-independent by design: cancels in dG"))
    register(FeatureSpec("W6_unit", "builder", {"aa_descriptors": "mordred_pca16_unit"},
                         _builder_adapter(_desc_or_none, takes_folded=False, takes_x=False),
                         state_dependent=False,
                         notes="state-independent by design: cancels in dG"))


# ---------------------------------------------------------------- standalone features
def _neighbours(ctx, i, cutoff=8.0, cap=12):
    cb = ctx.coords[:, 3, :] if ctx.coords.shape[1] >= 4 else ctx.coords[:, 1, :]
    d = (cb - cb[i]).norm(dim=-1)
    d[i] = 1e9
    ok = (d < cutoff).nonzero().flatten()
    if len(ok) > cap:
        ok = ok[d[ok].argsort()[:cap]]
    return ok, d


def rotamer_clash(ctx, one_hot, folded):
    """Fraction of rotamers that clash: a big residue in a tight site is bad only if NO
    rotamer fits. Invisible to a single centroid direction -- W12's approximation.
    Zero in the unfolded state: an extended chain has no packing."""
    N = ctx.N
    out = torch.zeros(N, 3)
    if not folded:
        return out
    reach = torch.tensor([1.9,2.4,2.5,3.1,3.4,0.0,3.2,2.6,3.9,2.6,
                          3.2,2.5,1.9,3.0,4.1,1.4,1.9,2.2,4.0,3.8])
    idx = one_hot.argmax(1)
    cb = ctx.coords[:, 3, :] if ctx.coords.shape[1] >= 4 else ctx.coords[:, 1, :]
    d = torch.cdist(cb, cb)
    for i in range(N):
        nb, dd = _neighbours(ctx, i)
        if len(nb) == 0:
            continue
        R = float(reach[idx[i]])
        free = dd[nb] - 3.8
        n_tot = 8
        angles = torch.linspace(0.3, 1.0, n_tot) * R      # rotamer extents
        clash = ((angles.unsqueeze(1) > free.unsqueeze(0)).float().sum(1) > 0).float()
        out[i, 0] = clash.mean()                           # fraction clashing
        out[i, 1] = torch.relu(R - free.min()).item()      # depth of worst clash
        out[i, 2] = float(len(nb))                         # crowding
    return out


def register_standalone():
    register(FeatureSpec("rotamer", "standalone", {}, rotamer_clash,
                         notes="untested class: rotamer-aware clash"))

# ======================================================================================
# Aggregation: per-residue [N,k] -> per-mutation.
#
# MEASURED: the block changes at EXACTLY the mutated position and nowhere else
# (max delta 6.7-8.3 on W12/1GYZ). So value-at-the-mutated-position is not a lossy
# summary -- it is the complete change.
#
# COROLLARY: coordinates are SHARED across a protein's variants, so any block that does
# not read one_hot is CONSTANT within a protein and explains exactly zero within-protein
# residual by construction. That is why the mutant-dependent aggregators are primary.
# ======================================================================================

def aggregate_protein(spec, ctx, sub, CFG):
    """Return (X_feat, X_ctrl, diag) for one protein.

    X_feat  mutant-DEPENDENT  : dmut_f | ddG_like | nbhd_dmut     <- the feature
    X_ctrl  mutant-INDEPENDENT: pos_wt_f | dstate_wt | pos | i/N | wt one-hot
    """
    with cfg_flags(CFG, **spec.cfg_flags):
        wt = ctx.wt()
        B_wt_f = spec.build(ctx, wt, True)
        B_wt_u = spec.build(ctx, wt, False)
        if B_wt_f is None:
            return None, None, {}
        B_wt_f = B_wt_f.detach(); B_wt_u = B_wt_u.detach() if B_wt_u is not None else B_wt_f
        k = B_wt_f.shape[1]
        feat, ctrl = [], []
        nz_offsite = 0
        for _, row in sub.iterrows():
            i, aa = int(row.pos), row.mut_aa
            if not (0 <= i < ctx.N) or aa not in AA:
                feat.append(np.zeros(2 * k + 2)); ctrl.append(np.zeros(2 * k + 22)); continue
            om = ctx.mutant(i, aa)
            B_m_f = spec.build(ctx, om, True).detach()
            B_m_u = spec.build(ctx, om, False)
            B_m_u = B_m_u.detach() if B_m_u is not None else B_m_f
            dmut_f   = (B_m_f[i] - B_wt_f[i]).numpy()
            ddG_like = ((B_m_f[i] - B_wt_f[i]) - (B_m_u[i] - B_wt_u[i])).numpy()
            nb, _ = _neighbours(ctx, i)
            if len(nb):
                dn = (B_m_f[nb] - B_wt_f[nb]).norm(dim=1)
                nbhd = np.array([float(dn.mean()), float(dn.max())])
            else:
                nbhd = np.zeros(2)
            feat.append(np.concatenate([dmut_f, ddG_like, nbhd]))
            # diagnostic: does the block change anywhere OTHER than the mutated site?
            off = (B_m_f - B_wt_f).abs().sum(1)
            off[i] = 0.0
            if float(off.max()) > 1e-9: nz_offsite += 1
            wt_oh = np.zeros(20); wt_oh[AA.index(row.wt_aa)] = 1.0 if row.wt_aa in AA else 0.0
            ctrl.append(np.concatenate([B_wt_f[i].numpy(), (B_wt_f[i] - B_wt_u[i]).numpy(),
                                        [i, i / max(ctx.N, 1)], wt_oh]))
    return (np.array(feat), np.array(ctrl),
            {"k": k, "offsite": nz_offsite,
             "state_dist": float((B_wt_f - B_wt_u).abs().max())})


def aggregate_proxy(spec, ctx, sub, CFG):
    """Pair proxy: additive baseline vs PRODUCT extras.
    Measured justification: on an edge, differences and sums fit at R2=1.0000 while the
    product h_src*h_dst reaches only 0.1970. Sums die; products survive."""
    add, prod = [], []
    for _, row in sub.iterrows():
        i, aa = int(row.pos), row.mut_aa
        if not (0 <= i < ctx.N) or aa not in AA:
            add.append(np.zeros(6)); prod.append(np.zeros(4)); continue
        nb, d = _neighbours(ctx, i)
        idx = ctx.wt().argmax(1)
        hs = np.array([KD[AA[int(idx[j])]] for j in nb]) if len(nb) else np.zeros(1)
        vs = np.array([VOL[AA[int(idx[j])]] for j in nb]) if len(nb) else np.zeros(1)
        cs = np.array([CHG[AA[int(idx[j])]] for j in nb]) if len(nb) else np.zeros(1)
        dd = d[nb].numpy() if len(nb) else np.ones(1)
        hm, vm, cm = KD[aa], VOL[aa], CHG[aa]
        hw = KD.get(row.wt_aa, 0.0)
        add.append(np.array([hm, vm, cm, hs.mean(), vs.mean(), float(len(nb))]))
        prod.append(np.array([hm * hs.mean(), vm * vs.mean(),
                              cm * float((cs / np.maximum(dd, 1.0)).sum()),
                              (hm - hw) * hs.mean()]))
    return np.array(add), np.array(prod), {"k": 4, "offsite": 0, "state_dist": float("nan")}

# ======================================================================================
# Q2: does the feature explain the WITHIN-protein residual, leave-one-protein-out?
#
# GroupKFold by protein, NEVER a random split: b_p is a per-protein constant and a random
# split lets the model memorise it -- exactly the trap that made nine b_p predictors look
# fine in-sample and fail out-of-sample.
#
# Headline is dR2 OVER THE POSITIONAL CONTROL, not raw R2. Without that, any feature
# correlated with "buried positions are harder" gets credit for a fact we already know.
# Negative R2 is printed as-is, never clipped.
# ======================================================================================

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _r2_within(y, pred, groups):
    """R2 on pooled out-of-fold predictions, both re-centred within protein.
    Re-centring the OOF predictions matters: even with grouped CV an intercept fitted on
    26 proteins carries a global offset into the held-out one."""
    p = pd.Series(pred); g = pd.Series(groups)
    p = (p - p.groupby(g).transform("mean")).values
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _oof(X, y, groups, model="hgb"):
    n_g = len(np.unique(groups))
    if n_g < 3 or X.shape[1] == 0:
        return np.zeros_like(y)
    cv = GroupKFold(n_splits=min(n_g, 27))
    if model == "ridge":
        est = make_pipeline(StandardScaler(),
                            RidgeCV(alphas=np.logspace(-3, 3, 25)))
    else:
        est = HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                                            learning_rate=0.05, early_stopping=False,
                                            random_state=0)
    return cross_val_predict(est, X, y, groups=groups, cv=cv)


def q2_score(Xf, Xc, Y, groups, n_shuffle=20, model="hgb", seed=0):
    """Returns dict with dR2 per control seed, the shuffle null, and the control-only R2."""
    keep_f = np.isfinite(Xf).all(0) & (Xf.std(0) > 1e-12)
    keep_c = np.isfinite(Xc).all(0) & (Xc.std(0) > 1e-12)
    Xf2, Xc2 = Xf[:, keep_f], Xc[:, keep_c]
    both = np.hstack([Xc2, Xf2]) if Xf2.shape[1] else Xc2
    d = {}
    for col in Y.columns:
        y = Y[col].values
        ok = np.isfinite(y)
        if ok.sum() < 100:
            continue
        r_ctrl = _r2_within(y[ok], _oof(Xc2[ok], y[ok], groups[ok], model), groups[ok])
        r_both = _r2_within(y[ok], _oof(both[ok], y[ok], groups[ok], model), groups[ok])
        d[col] = (r_both - r_ctrl, r_ctrl, r_both)
    # B2: within-protein shuffle null -- a 4-col and a 16-col feature do NOT share a null
    rng = np.random.default_rng(seed)
    col0 = Y.columns[0]
    y0 = Y[col0].values
    ok = np.isfinite(y0)
    nulls = []
    for _ in range(n_shuffle):
        Xs = Xf2.copy()
        for g in np.unique(groups):
            m = groups == g
            perm = rng.permutation(m.sum())
            Xs[m] = Xs[m][perm]
        bs = np.hstack([Xc2, Xs]) if Xs.shape[1] else Xc2
        r_c = _r2_within(y0[ok], _oof(Xc2[ok], y0[ok], groups[ok], model), groups[ok])
        r_b = _r2_within(y0[ok], _oof(bs[ok], y0[ok], groups[ok], model), groups[ok])
        nulls.append(r_b - r_c)
    return d, float(np.percentile(nulls, 95)) if nulls else 0.0


def q3_absorption(Xf, Xc):
    """Is the feature linearly reconstructible from what the network already receives?
    R2 ~ 1 means a linear layer absorbs it. CONFOUND: for a state-INDEPENDENT block an
    'absorbed' verdict may reflect dG cancellation rather than redundancy."""
    keep_f = np.isfinite(Xf).all(0) & (Xf.std(0) > 1e-12)
    keep_c = np.isfinite(Xc).all(0) & (Xc.std(0) > 1e-12)
    A, B = Xc[:, keep_c], Xf[:, keep_f]
    if B.shape[1] == 0 or A.shape[1] == 0:
        return float("nan")
    A1 = np.hstack([A, np.ones((len(A), 1))])
    r2s = []
    for j in range(B.shape[1]):
        b = B[:, j]
        if b.std() < 1e-12:
            continue
        coef, *_ = np.linalg.lstsq(A1, b, rcond=None)
        resid = b - A1 @ coef
        r2s.append(1.0 - resid.var() / max(b.var(), 1e-12))
    return float(np.mean(r2s)) if r2s else float("nan")

# ======================================================================================
# Driver, self-test, calibration, output
# ======================================================================================

def run_feature(spec, IDX, Y, CFG, ctxs):
    F, C, diag = [], [], {"offsite": 0, "state_dist": []}
    for P, sub in IDX.groupby("protein", sort=True):
        ctx = ctxs[P]
        if spec.kind == "proxy":
            a, b, d = aggregate_proxy(spec, ctx, sub, CFG)
            f, c = b, a                       # extras are the feature, additive is control
        else:
            f, c, d = aggregate_protein(spec, ctx, sub, CFG)
            if f is None:
                return None
        F.append(f); C.append(c)
        diag["offsite"] += d.get("offsite", 0)
        if np.isfinite(d.get("state_dist", np.nan)):
            diag["state_dist"].append(d["state_dist"])
    Xf, Xc = np.vstack(F), np.vstack(C)
    groups = IDX.protein.values
    d, null95 = q2_score(Xf, Xc, Y, groups)
    dr = np.array([v[0] for v in d.values()]) if d else np.array([0.0])
    return {
        "name": spec.name, "kind": spec.kind, "notes": spec.notes,
        "q1": float(np.mean(diag["state_dist"])) if diag["state_dist"] else float("nan"),
        "dr2": float(dr.mean()), "sd": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
        "null95": null95, "q3": q3_absorption(Xf, Xc),
        "offsite": diag["offsite"], "nfeat": Xf.shape[1],
        "ddg_like_zero": bool(np.allclose(Xf[:, Xf.shape[1] // 2 - 1:Xf.shape[1] - 2], 0))
                          if spec.kind != "proxy" else False,
    }


def tier(r):
    if not np.isfinite(r["q1"]) and r["kind"] != "proxy": return "D"
    sig = r["dr2"] > r["null95"]
    if r["kind"] != "proxy" and np.isfinite(r["q1"]) and r["q1"] < 0.01: return "D"
    if r["dr2"] <= 0: return "D"
    if sig and r["q3"] < 0.9 and (r["kind"] == "proxy" or r["q1"] > 0.05): return "A"
    if sig: return "B"
    return "C"


BANNER = """
NOTES -- read before acting on the table above
 1. This gate RANKS, it does not BLOCK. A low tier plus a mechanistic argument is a valid
    reason to submit an arm.
 2. Calibrated on n=4 known outcomes (W12/W15/W5/W6). Rank agreement on 4 points is ~4%
    by chance. Treat tiers as coarse.
 3. Q2 is measured against the residual of ONE model class at 15 epochs. A feature this
    architecture cannot exploit may still be real.
 4. Within-protein target only: 30.2% of residual variance is a per-protein constant and
    is excluded by design. A feature that predicts b_p scores ZERO here.
 5. Q1 ~ 0 means the block cancels exactly in dG = E_f - E_u and cannot contribute."""


def print_table(res):
    res = sorted(res, key=lambda r: -r["dr2"])
    print("\n%-12s %-6s %8s %9s %7s %8s %7s %5s  %s" %
          ("FEATURE", "KIND", "Q1_state", "Q2_dR2", "+-sd", "vs_null", "Q3_abs", "TIER", "CAVEAT"))
    for r in res:
        sig = "SIG" if r["dr2"] > r["null95"] else ("~" if r["dr2"] > 0.5 * r["null95"] else "ns")
        cav = r["notes"]
        if np.isfinite(r["q1"]) and r["q1"] < 0.01 and (r["q3"] or 0) > 0.9:
            cav = "dG-cancels; Q3 confounded"
        if r["offsite"]:
            cav = (cav + " | %d off-site changes" % r["offsite"]).strip(" |")
        print("%-12s %-6s %8.3f %+9.4f %7.4f %8s %7.3f %5s  %s" %
              (r["name"][:12], r["kind"][:6], r["q1"], r["dr2"], r["sd"],
               sig, r["q3"] if np.isfinite(r["q3"]) else -1, tier(r), cav[:44]))
    print(BANNER)


def self_test(IDX, nprot, maxdiff, ctxs, CFG, MAXM=None):
    print("=== SELF-TEST ===")
    check("join: 27 proteins", nprot == 27, "got %d" % nprot)
    check("join: max|deltaG diff| < 1e-6", maxdiff < 1e-6, "%.3e" % maxdiff)
    exp = 25000 if MAXM is None else 20 * nprot
    check("index: enough scorable mutations", len(IDX) > exp,
          "%d (expect >%d)" % (len(IDX), exp))
    P0 = sorted(ctxs)[0]; c0 = ctxs[P0]
    check("one_hot sliced to 20 cols", c0.one_hot_all.shape[-1] == 20,
          "%s" % (tuple(c0.one_hot_all.shape),))
    # R1: the mutant block MUST differ from wild-type, else every builder scores zero
    from train_utils import _sidechain_or_none
    with cfg_flags(CFG, sidechain_features=True):
        wt = c0.wt(); Bw = _sidechain_or_none(c0.coords, wt, c0.mask, True)
        sub = IDX[IDX.protein == P0].head(30)
        ndiff = 0
        for _, row in sub.iterrows():
            om = c0.mutant(int(row.pos), row.mut_aa)
            Bm = _sidechain_or_none(c0.coords, om, c0.mask, True)
            if float((Bm - Bw).abs().max()) > 1e-9: ndiff += 1
    check("R1: mutant block differs from wild-type", ndiff > 20, "%d/30 differ" % ndiff)
    before = {k for k in vars(CFG) if not k.startswith("_")}
    with cfg_flags(CFG, burial_features=True): pass
    after = {k for k in vars(CFG) if not k.startswith("_")}
    check("CFG round-trip: no leak", before == after,
          "" if before == after else "leaked %s" % (after - before))
    return _ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-muts-per-protein", type=int, default=None)
    ap.add_argument("--features", type=str, default=None)
    a = ap.parse_args()

    from train_utils import CFG
    register_builders(); register_standalone()
    register(FeatureSpec("pair_proxy", "proxy", {}, None,
                         notes="proxy: LOWER BOUND on pairwise signal"))

    IDX, nprot, maxdiff = build_index(max_muts=a.max_muts_per_protein)
    ctxs = {P: ProteinCtx.load(P) for P in sorted(IDX.protein.unique())}
    Y = residual_table(IDX)
    print("control seeds with residuals: %d" % Y.shape[1])

    if a.self_test:
        ok = self_test(IDX, nprot, maxdiff, ctxs, CFG, a.max_muts_per_protein)
        print("\nSELF-TEST: %s (%d failure%s)" %
              ("ALL PASS" if ok else "FAILURES", _fails, "" if _fails == 1 else "s"))
        sys.exit(0 if ok else 1)

    names = (a.features.split(",") if a.features
             else (["W12", "W15", "W5", "W6_pca16"] if a.calibrate else list(REGISTRY)))
    res = []
    for nm in names:
        if nm not in REGISTRY:
            print("  unknown feature %s" % nm); continue
        r = run_feature(REGISTRY[nm], IDX, Y, CFG, ctxs)
        if r: res.append(r); print("  scored %-12s dR2 %+0.4f" % (nm, r["dr2"]))
    print_table(res)

    if a.calibrate:
        print("\n=== CALIBRATION (n=4, ~4%% by chance) ===")
        d = {r["name"]: r["dr2"] for r in res}
        if all(k in d for k in ("W12", "W15", "W5", "W6_pca16")):
            check("W12 > W15", d["W12"] > d["W15"], "%.4f vs %.4f" % (d["W12"], d["W15"]))
            check("W15 > W5", d["W15"] > d["W5"], "%.4f vs %.4f" % (d["W15"], d["W5"]))
            check("W5 > W6", d["W5"] > d["W6_pca16"], "%.4f vs %.4f" % (d["W5"], d["W6_pca16"]))
            w6 = [r for r in res if r["name"] == "W6_pca16"][0]
            check("W6 not significant", w6["dr2"] <= w6["null95"],
                  "%.4f vs null %.4f" % (w6["dr2"], w6["null95"]))
        print("\nCALIBRATION: %s (%d failure%s)" %
              ("ALL PASS" if _ok else "FAILURES", _fails, "" if _fails == 1 else "s"))
        sys.exit(0 if _ok else 1)
    sys.exit(0)          # ranking mode ALWAYS exits 0


if __name__ == "__main__":
    main()
