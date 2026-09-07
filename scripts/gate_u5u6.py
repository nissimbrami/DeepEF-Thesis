"""Gate for U5 (coil-consistent edge topology) and U6 (per-half ca_coords).

RUN THIS BEFORE ANY TRAINING RUN THAT PASSES --coil_edges. No GPU, no data, seconds.

    python gate_u5u6.py

Exit 0 = all gates pass. Non-zero = do not launch.

The two gates named in the item spec:
  G2-U5 : the unfolded edge count equals 2(N-1), NOT the k-NN / fully-connected count.
  G2-U6 : the unfolded half's CA coordinates DIFFER from the folded half's when the coil
          is on.

Plus the rules every lever in this tree must satisfy:
  OFF-PATH   : with --coil_edges off, every output is byte-identical to the input
               (torch.equal, never allclose -- allclose would hide a dtype or a
               last-bit change, and "byte-identical when off" is the whole contract).
  RAISE      : --coil_edges without --flory_unfolded RAISES, it does not silently no-op.
  FOLDED-ONLY: a batch with n_folded=None (or n_folded=B) is untouched -- this is the
               f_type='features' path in get_ddg_head, which must never be split.
  CACHE      : the edge cache key separates coil from non-coil and separates row splits.
"""

import sys
import torch

import coil_topology as CT


class _Cfg(object):
    """Stand-in for the module-level CFG object the real tree reads at call time."""
    def __init__(self, coil_edges=False, flory_unfolded=False, flory_nu=0.5,
                 gcn_span=1, gcn_bidir=False):
        self.coil_edges = coil_edges
        self.flory_unfolded = flory_unfolded
        self.flory_nu = flory_nu
        self.gcn_span = gcn_span
        self.gcn_bidir = gcn_bidir


_FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s  %s" % (name, detail))
        _FAILURES.append(name)


def fully_connected_edges(B, N, device):
    """Reproduces PEM.get_edge_index's fully-connected GAT branch exactly, so the gate is
    testing against the real baseline object and not a paraphrase of it."""
    offsets = torch.arange(B, device=device).unsqueeze(1) * N
    arange = torch.arange(N, device=device)
    src, dst = torch.meshgrid(arange, arange, indexing='ij')
    mask = src != dst
    s = (src[mask].unsqueeze(0) + offsets).reshape(-1)
    d = (dst[mask].unsqueeze(0) + offsets).reshape(-1)
    return torch.stack([s, d])


def main():
    dev = torch.device('cpu')
    N = 11          # small, odd, > 2 so 2(N-1) and N(N-1) are clearly different
    n_f = 4         # folded rows
    B = 2 * n_f     # the batch shape both real call sites build

    off = _Cfg(coil_edges=False, flory_unfolded=False)
    off_coil = _Cfg(coil_edges=False, flory_unfolded=True)   # coil node feats, U5 off
    on = _Cfg(coil_edges=True, flory_unfolded=True)

    base = fully_connected_edges(B, N, dev)

    print("\n[1] G2-U5  unfolded edge count == 2(N-1) per row, not the fully-connected count")
    out = CT.apply_coil_edges(base, B, N, n_f, dev, on)
    row = torch.div(out[0], N, rounding_mode='floor')
    n_unf_edges = int((row >= n_f).sum())
    n_fold_edges = int((row < n_f).sum())
    expect_coil = (B - n_f) * 2 * (N - 1)
    expect_fold = n_f * N * (N - 1)
    check("unfolded edges == 2(N-1) per row  [%d == %d]" % (n_unf_edges, expect_coil),
          n_unf_edges == expect_coil,
          "got %d expected %d" % (n_unf_edges, expect_coil))
    check("unfolded edges != fully-connected count  [%d != %d]"
          % (n_unf_edges, (B - n_f) * N * (N - 1)),
          n_unf_edges != (B - n_f) * N * (N - 1))
    check("folded half untouched  [%d == %d]" % (n_fold_edges, expect_fold),
          n_fold_edges == expect_fold,
          "got %d expected %d" % (n_fold_edges, expect_fold))

    # the coil edges must be exactly the bonded neighbours, nothing else
    unf = out[:, row >= n_f]
    sep = (unf[0] % N - unf[1] % N).abs()
    check("every unfolded edge has |i-j| == 1", bool((sep == 1).all()))
    same_row = torch.div(unf[0], N, rounding_mode='floor') == torch.div(
        unf[1], N, rounding_mode='floor')
    check("no unfolded edge crosses a batch row", bool(same_row.all()))
    # bidirectional: the reversed set equals the set
    fwd = set(zip(unf[0].tolist(), unf[1].tolist()))
    check("unfolded edge set is symmetric (bidirectional chain)",
          all((d, s) in fwd for (s, d) in fwd))

    print("\n[2] OFF-PATH  byte-identical when --coil_edges is off (torch.equal)")
    for label, cfg in (("flags fully off", off), ("coil node feats on, U5 off", off_coil)):
        got = CT.apply_coil_edges(base, B, N, n_f, dev, cfg)
        check("apply_coil_edges returns the SAME object (%s)" % label, got is base)
        check("apply_coil_edges byte-identical (%s)" % label, torch.equal(got, base))

    ca = torch.randn(B, N, 3)
    for label, cfg in (("flags fully off", off), ("coil node feats on, U5 off", off_coil)):
        got = CT.per_half_ca_coords(ca, B, N, n_f, cfg)
        check("per_half_ca_coords byte-identical (%s)" % label, torch.equal(got, ca))
    check("per_half_ca_coords(None) is None",
          CT.per_half_ca_coords(None, B, N, n_f, on) is None)

    print("\n[3] FOLDED-ONLY  n_folded=None / n_folded=B leaves the batch alone")
    # this is get_ddg_head's f_type='features' path: a folded-only batch must NEVER be split
    for nf in (None, B):
        got = CT.apply_coil_edges(base, B, N, nf, dev, on)
        check("apply_coil_edges untouched at n_folded=%r" % (nf,), torch.equal(got, base))
        gotc = CT.per_half_ca_coords(ca, B, N, nf, on)
        check("per_half_ca_coords untouched at n_folded=%r" % (nf,), torch.equal(gotc, ca))

    print("\n[4] RAISE  --coil_edges without --flory_unfolded must raise, not no-op")
    bad = _Cfg(coil_edges=True, flory_unfolded=False)
    raised = False
    try:
        CT.apply_coil_edges(base, B, N, n_f, dev, bad)
    except ValueError as e:
        raised = "flory_unfolded" in str(e)
    check("coil_edges_enabled raises on --coil_edges alone", raised)
    raised2 = False
    try:
        CT.validate_coil_edges_flags(True, False)
    except ValueError:
        raised2 = True
    check("validate_coil_edges_flags raises at argparse time", raised2)
    try:
        CT.validate_coil_edges_flags(True, True)
        CT.validate_coil_edges_flags(False, False)
        ok_pairs = True
    except ValueError:
        ok_pairs = False
    check("valid flag pairs do not raise", ok_pairs)

    print("\n[5] G2-U6  unfolded half's CA coords differ from the folded half's, coil on")
    ca_f = torch.randn(n_f, N, 3) * 8.0
    ca_b = torch.cat([ca_f, ca_f], dim=0)          # today's leak: folded coords in both halves
    out_ca = CT.per_half_ca_coords(ca_b, B, N, n_f, on)
    check("shape preserved [B,N,3]", tuple(out_ca.shape) == (B, N, 3))
    check("folded half byte-identical to input", torch.equal(out_ca[:n_f], ca_b[:n_f]))
    differ = all(not torch.equal(out_ca[n_f + r], out_ca[r]) for r in range(n_f))
    check("UNFOLDED half differs from FOLDED half (G2-U6)", differ)
    check("without U6 the two halves WERE identical (the defect this fixes)",
          torch.equal(ca_b[:n_f], ca_b[n_f:]))

    print("\n[6] U6 distances: the exact analytic path")
    d = CT.per_half_ca_distances(ca_b, B, N, n_f, on)
    check("distance shape [B,N,N]", tuple(d.shape) == (B, N, N))
    check("folded-half distances == cdist of folded coords",
          torch.equal(d[:n_f], torch.cdist(ca_b[:n_f], ca_b[:n_f])))
    nu = on.flory_nu
    for r in range(n_f):
        b = CT.coil_bond_length(ca_b[r])
        expect = CT.coil_ca_distances(N, b, nu, ca_b.device, ca_b.dtype)
        if not torch.equal(d[n_f + r], expect):
            check("unfolded row %d == analytic b*|i-j|^nu" % r, False)
            break
    else:
        check("every unfolded row == analytic b*|i-j|^nu (torch.equal)", True)
    check("unfolded distances differ from folded distances",
          not torch.equal(d[n_f:], d[:n_f]))
    # the coil must be monotone in separation -- a coil where d(0,5) < d(0,1) is not a coil
    b0 = CT.coil_bond_length(ca_b[0])
    dc = CT.coil_ca_distances(N, b0, nu, ca_b.device, ca_b.dtype)
    check("coil distance monotone in |i-j|  (d[0,1] < d[0,5] < d[0,10])",
          float(dc[0, 1]) < float(dc[0, 5]) < float(dc[0, N - 1]))

    print("\n[7] b matches _flory_unfolded_graph's definition exactly")
    # _flory_unfolded_graph: ca = x[:,1,:]; b = norm(ca[1:]-ca[:-1]).mean(); clamp(min=1e-3)
    x = torch.randn(N, 4, 3) * 5.0
    ca_from_x = x[:, 1, :]
    b_ref = torch.clamp(torch.linalg.norm(ca_from_x[1:] - ca_from_x[:-1], dim=-1).mean(),
                        min=1e-3)
    check("coil_bond_length reproduces the coil's own b (torch.equal)",
          torch.equal(CT.coil_bond_length(ca_from_x), b_ref))
    # The N==1 fallback. This assertion USED to be float(...) == 3.8 and it FAILED --
    # not because the code was wrong but because the ASSERTION was. torch.tensor(3.8) is
    # float32, i.e. 3.799999952316284, and comparing it to the float64 Python literal 3.8
    # with == is never true. The contract that actually matters is byte-identity with
    # train_utils._coil_bond_length's own fallback (`torch.tensor(3.8, device=dev)`), which
    # is what this now asserts, with torch.equal, against the same expression.
    # NOTE the source's `fitted` fallback is a bare literal 3.8 while U4's `fixed` arm uses
    # _COIL_B_FIXED = 5.82 * 0.1 in MODEL units. That inconsistency is real and is
    # train_utils' to own, not U6's: coil_bond_length must track whatever the node-feature
    # coil does, so it reproduces the literal rather than quietly converting it.
    check("N==1 falls back to the coil's own 3.8 constant, byte-identically",
          torch.equal(CT.coil_bond_length(torch.zeros(1, 3)), torch.tensor(3.8)))

    print("\n[8] CACHE key separates the cases the edge set actually depends on")
    k_off = CT.edge_cache_key(B, N, off, n_f)
    k_on = CT.edge_cache_key(B, N, on, n_f)
    k_on2 = CT.edge_cache_key(B, N, on, n_f + 1)
    check("coil on/off give different keys", k_off != k_on)
    check("different n_folded gives a different key", k_on != k_on2)
    check("same inputs give the same key", CT.edge_cache_key(B, N, on, n_f) == k_on)

    print("\n[9] EDGE CASES")
    e1 = CT.chain_local_edge_index(2, 1, dev)      # N == 1: a chain of one has no bonds
    check("N==1 yields zero coil edges", e1.numel() == 0 and tuple(e1.shape) == (2, 0))
    check("N==1 dtype is long", e1.dtype == torch.long)
    e2 = CT.chain_local_edge_index(1, 2, dev)      # N == 2: exactly 2 edges
    check("N==2 yields exactly 2 edges", e2.shape[1] == 2)
    raised3 = False
    try:
        CT.split_folded_rows(4, 9)
    except ValueError:
        raised3 = True
    check("n_folded > B raises", raised3)
    # odd split (not B//2) must still work -- nothing may assume the halves are equal
    out_odd = CT.apply_coil_edges(base, B, N, 3, dev, on)
    row_odd = torch.div(out_odd[0], N, rounding_mode='floor')
    check("asymmetric split honoured (n_folded=3 of B=%d)" % B,
          int((row_odd >= 3).sum()) == (B - 3) * 2 * (N - 1))

    print("\n[10] INTEGRATION  the REAL PEM.get_edge_index, not a paraphrase of it")
    # A module that passes in isolation proves nothing about the wiring. These checks import
    # the actual model and drive the actual code path, with CFG mutated the way train.py
    # mutates it.
    import os as _os, sys as _sys
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from model.model_cfg import CFG as _CFG
    from model.hydro_net import PEM as _PEM

    _saved = (getattr(_CFG, 'coil_edges', False), getattr(_CFG, 'flory_unfolded', False))
    try:
        # constructed exactly as gate_g4_cpu constructs it, and fed REAL graphs so the
        # feature width is whatever the tree actually produces, not a hardcoded 1092.
        from train_utils import get_graph as _get_graph
        m = _PEM(layers=_CFG.num_layers, gaussian_coef=_CFG.gaussian_coef,
                 dropout_rate=_CFG.dropout_rate, light_attention=True, readout=False)
        m.eval()
        _xyz = torch.randn(N, 4, 3) * 7.0
        _oh = torch.eye(20)[torch.randint(0, 20, (N,))]
        _emb = torch.randn(N, int(_CFG.emb_input_dim))
        _msk = torch.ones(N)
        xin = _get_graph(_xyz, _oh, _emb, _msk).unsqueeze(0).repeat(B, 1, 1)

        # -- OFF path: byte-identical edge sets, and the flag is genuinely absent by default
        _CFG.coil_edges, _CFG.flory_unfolded = False, False
        m._edge_cache_key, m._edge_cache = None, None
        g_off, a_off = m.get_edge_index(xin, n_folded=n_f)
        m._edge_cache_key, m._edge_cache = None, None
        g_base, a_base = m.get_edge_index(xin)          # exactly today's call, no n_folded
        check("PEM GAT edges byte-identical with coil off (torch.equal)",
              torch.equal(a_off, a_base))
        check("PEM GCN edges byte-identical with coil off (torch.equal)",
              torch.equal(g_off, g_base))
        check("PEM GAT edge count is the fully-connected baseline",
              a_base.shape[1] == B * N * (N - 1))

        # -- ON path: the unfolded rows are chain-local, the folded rows are not
        _CFG.coil_edges, _CFG.flory_unfolded = True, True
        m._edge_cache_key, m._edge_cache = None, None
        g_on, a_on = m.get_edge_index(xin, n_folded=n_f)
        r = torch.div(a_on[0], N, rounding_mode='floor')
        check("PEM unfolded GAT edges == 2(N-1) per row  [%d]" % int((r >= n_f).sum()),
              int((r >= n_f).sum()) == (B - n_f) * 2 * (N - 1))
        check("PEM folded GAT edges still fully connected",
              int((r < n_f).sum()) == n_f * N * (N - 1))
        check("PEM GCN edges UNCHANGED by --coil_edges (U5 is GAT-only)",
              torch.equal(g_on, g_base))

        # -- the cache bug this patch exists to prevent
        m._edge_cache_key, m._edge_cache = None, None
        _CFG.coil_edges = False
        _, a1 = m.get_edge_index(xin, n_folded=n_f)     # populates the cache, coil off
        _CFG.coil_edges = True
        _, a2 = m.get_edge_index(xin, n_folded=n_f)     # same (B,N): must NOT hit that entry
        check("cache does NOT serve the folded topology to a coil batch",
              a2.shape[1] != a1.shape[1])
        # and the folded-only path must never be split even with the coil on
        m._edge_cache_key, m._edge_cache = None, None
        _, a_feat = m.get_edge_index(xin)               # n_folded=None, f_type='features'
        check("folded-only path (n_folded=None) untouched with coil ON",
              torch.equal(a_feat, a_base))

        # -- forward() end-to-end, both halves, off vs on
        _CFG.coil_edges, _CFG.flory_unfolded = False, False
        m._edge_cache_key, m._edge_cache = None, None
        with torch.no_grad():
            e_plain = m(xin)
            m._edge_cache_key, m._edge_cache = None, None
            e_nf = m(xin, n_folded=n_f)
        check("PEM.forward byte-identical with/without n_folded when coil off",
              torch.equal(e_plain, e_nf))
        _CFG.coil_edges, _CFG.flory_unfolded = True, True
        m._edge_cache_key, m._edge_cache = None, None
        with torch.no_grad():
            e_on = m(xin, n_folded=n_f)
        check("PEM.forward folded half unchanged by the coil",
              torch.equal(e_on[:n_f], e_plain[:n_f]))
        check("PEM.forward unfolded half CHANGED by the coil",
              not torch.equal(e_on[n_f:], e_plain[n_f:]))
        check("PEM.forward output finite with the coil on",
              bool(torch.isfinite(e_on).all()))

        # -- RAISE through the real model, not just the helper
        _CFG.coil_edges, _CFG.flory_unfolded = True, False
        m._edge_cache_key, m._edge_cache = None, None
        raised4 = False
        try:
            m.get_edge_index(xin, n_folded=n_f)
        except ValueError as e:
            raised4 = "flory_unfolded" in str(e)
        check("PEM raises on --coil_edges without --flory_unfolded", raised4)
    finally:
        _CFG.coil_edges, _CFG.flory_unfolded = _saved

    print("")
    if _FAILURES:
        print("GATE FAILED: %d check(s) -> %s" % (len(_FAILURES), _FAILURES))
        return 1
    print("GATE PASSED: U5 + U6 ready. Byte-identical when off; G2-U5 and G2-U6 hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
