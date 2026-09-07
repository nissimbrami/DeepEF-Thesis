"""U5 / U6 -- coil-consistent edge topology and per-half CA coordinates.

WHY THIS FILE EXISTS
====================
dG = E_unfolded - E_folded. In this tree the two states are NOT two graph objects.
train.py builds two feature tensors and concatenates them along the BATCH dimension,
then makes ONE model call:

    all_graph = torch.cat([folded_graph, unfolded_graph], dim=0)   # [2n, N, F]
    energy    = self.model(all_graph)
    folded_energy, unfolded_energy = energy[:n], energy[n:]

Edges are then built once, inside PEM.get_edge_index, from SHAPE alone. Every row of the
batch -- folded and unfolded alike -- gets the same topology: fully connected for the GAT
(or a CA-distance cutoff when ca_coords is supplied and gat_cutoff is set).

That is defect A5: the "unfolded" state is handed the folded contact topology even after
its node features have been replaced by a coil. The coil is therefore HALF APPLIED, and a
weak coil result measured without this fix is confounded.

Defect A6 is the same leak through edge ATTRIBUTES: if ca_coords is broadcast over both
halves, the true folded CA distances become edge features in the unfolded pass, and Phase 7
and the coil then cancel each other exactly.

MEASURED CONTEXT -- READ BEFORE SPENDING GPU HOURS ON THIS
==========================================================
The W0 channel ablation (28 proteins, no training, calib_ctrl_repro2 e14) found the offset
lives in the ProtT5 channel, not the geometry: zeroing the unfolded ProtT5 block drops
var(E_u) to 0.331 of baseline and corr(E_u, wt_err) from 0.420 to 0.119, while the Flory
coil RAISES var(E_u) to 1.175 of baseline. The coil is measured HARMFUL. U5/U6 are
therefore not on the factorial's critical path; their value is that they make a coil
re-test FAIR rather than confounded. See REPORT.md.

WHAT IS AND IS NOT TRUE OF THE SOURCE TODAY
===========================================
ca_coords is declared as a parameter in four places in hydro_net.py and is passed by NOBODY.
train.py calls self.model(x) and self.model(x, f_type='features'). train_utils.py does not
contain the string at all. So A6 does not currently fire -- it is LATENT, and it fires the
moment Phase 7 or any --gat_cutoff run wires coordinates through. This module is written so
it is correct whether ca_coords arrives or not.

EVERY FUNCTION HERE DEFAULTS TO CURRENT BEHAVIOUR AND IS BYTE-IDENTICAL WHEN THE FLAGS ARE
OFF. Verified with torch.equal, never allclose -- see gate_u5u6.py.
"""

import torch


# ---------------------------------------------------------------------------
# Flag plumbing
# ---------------------------------------------------------------------------

def coil_edges_enabled(cfg):
    """True when U5 is active. Reads CFG at call time, like every other lever in this tree.

    --coil_edges without --flory_unfolded is a HARD ERROR, not a silent no-op: rebuilding
    the unfolded topology while the unfolded NODE features are still the folded tridiagonal
    would produce a third state that is neither the baseline nor the coil, and any number
    measured from it would be uninterpretable. The check lives here so it fires on the first
    forward pass even if the argparse-time validation in train.py were ever removed.
    """
    on = bool(getattr(cfg, 'coil_edges', False))
    if on and not bool(getattr(cfg, 'flory_unfolded', False)):
        raise ValueError(
            "--coil_edges requires --flory_unfolded. U5 rebuilds the UNFOLDED half's edge "
            "topology to be chain-local, which is only coherent when the unfolded NODE "
            "features are the Flory coil. With the tridiagonal baseline node features this "
            "would create a third, uninterpretable reference state. Pass --flory_unfolded, "
            "or drop --coil_edges."
        )
    return on


def validate_coil_edges_flags(coil_edges, flory_unfolded):
    """Argparse-time twin of the check above, so the run dies in the first second and not
    six hours in. Call from train.py immediately after parse_known_args()."""
    if coil_edges and not flory_unfolded:
        raise ValueError(
            "--coil_edges requires --flory_unfolded (U5 is only defined against the coil "
            "unfolded reference state; without it the unfolded half would have chain-local "
            "edges but tridiagonal-mask node features -- a third state that is neither arm)."
        )


# ---------------------------------------------------------------------------
# U5 -- the chain-local ("coil") edge set
# ---------------------------------------------------------------------------

def chain_local_edge_index(n_rows, N, device, offset_rows=0):
    """The bidirectional chain edge set for n_rows batch rows of length N.

    A random coil has bonded neighbours and nothing else. It has no defined long-range
    contacts, so the only edges it can justify are (i, i+1) and (i+1, i).

    Returns a [2, n_rows * 2 * (N-1)] LongTensor of FLAT node indices, where row r of the
    batch occupies flat nodes [(offset_rows + r) * N, (offset_rows + r + 1) * N).

    GATE G2-U5: per row this is exactly 2*(N-1) edges -- NOT the fully-connected N*(N-1) and
    NOT a k-NN count. gate_u5u6.py asserts that equality.

    Note this is BIDIRECTIONAL, unlike the GCN baseline chain set which is directed forward
    (N-1 edges at span 1). The two are different objects on purpose: the GCN set is the
    historical baseline and U5 must not perturb it; this set is the coil's GAT topology, and
    an undirected contact graph is the right analogue of the fully-connected set it replaces.
    """
    if N <= 1 or n_rows <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    idx = torch.arange(N - 1, device=device, dtype=torch.long)
    local_src = torch.cat([idx, idx + 1])       # [2(N-1)]
    local_dst = torch.cat([idx + 1, idx])       # [2(N-1)]
    rows = torch.arange(n_rows, device=device, dtype=torch.long) + int(offset_rows)
    row_off = (rows * N).unsqueeze(1)           # [n_rows, 1]
    src = (local_src.unsqueeze(0) + row_off).reshape(-1)
    dst = (local_dst.unsqueeze(0) + row_off).reshape(-1)
    return torch.stack([src, dst])


def split_folded_rows(B, n_folded):
    """Normalise the folded/unfolded row split.

    n_folded is passed EXPLICITLY by the caller and is never inferred as B//2 inside the
    model. Reason: get_ddg_head calls the model with a FOLDED-ONLY batch under
    f_type='features'. Inferring B//2 there would declare half of a folded batch unfolded
    and silently corrupt it -- a wrong-number bug with no exception to catch it.

    n_folded=None means "every row is folded", which is exactly today's behaviour.
    """
    if n_folded is None:
        return B, 0
    n_folded = int(n_folded)
    if not (0 <= n_folded <= B):
        raise ValueError("n_folded=%d out of range for batch size %d" % (n_folded, B))
    return n_folded, B - n_folded


def restrict_edges_to_rows(edge_index, N, row_lo, row_hi):
    """Keep only edges whose SOURCE node lies in batch rows [row_lo, row_hi).

    Used to carve the folded half's edges out of a whole-batch edge set built by the existing
    code, so U5 can leave that code path untouched and merely replace the unfolded half.
    Edges never cross batch rows in this tree (both builders offset by arange(B)*N), so
    filtering on the source is sufficient and filtering on the destination is redundant.
    """
    if edge_index.numel() == 0:
        return edge_index
    row = torch.div(edge_index[0], N, rounding_mode='floor')
    keep = (row >= row_lo) & (row < row_hi)
    return edge_index[:, keep]


def apply_coil_edges(edge_index_full, B, N, n_folded, device, cfg):
    """U5 entry point for the GAT edge set.

    Given the edge set the baseline code already built for the WHOLE batch, return the same
    set for the folded rows and the chain-local coil set for the unfolded rows.

    When U5 is off, or there are no unfolded rows, edge_index_full is returned UNCHANGED --
    the identical object, so the off-path is byte-identical by construction rather than by
    reconstruction.

    Why the GAT set and not the GCN set: the GCN set is already chain-local by construction
    ((i, i+1), or |i-j| <= span at --gcn_span > 1). It carries no folded CONTACT topology, so
    there is nothing for U5 to remove there, and touching it would confound U5 with W7/U10.
    The A5 defect is specifically the GAT's fully-connected / CA-cutoff topology.
    """
    if not coil_edges_enabled(cfg):
        return edge_index_full
    n_f, n_u = split_folded_rows(B, n_folded)
    if n_u <= 0:
        return edge_index_full
    folded_part = restrict_edges_to_rows(edge_index_full, N, 0, n_f)
    coil_part = chain_local_edge_index(n_u, N, device, offset_rows=n_f)
    return torch.cat([folded_part, coil_part], dim=1)


def edge_cache_key(B, N, cfg, n_folded):
    """The cache in PEM.get_edge_index is keyed on (B, N) alone.

    With U5 the edge set additionally depends on WHERE the folded/unfolded boundary is and on
    whether the coil is on, so the key must carry both or a coil batch will be served a
    folded-topology edge set out of the cache. This is a real correctness bug, not a
    performance nicety. gcn_span / gcn_bidir are folded into the key as well: they are set
    once at startup today, but they are read off CFG at call time, so a key that omits them
    is one refactor away from being wrong.
    """
    return (B, N, int(bool(getattr(cfg, 'coil_edges', False))),
            int(getattr(cfg, 'gcn_span', 1)), bool(getattr(cfg, 'gcn_bidir', False)),
            -1 if n_folded is None else int(n_folded))


# ---------------------------------------------------------------------------
# U6 -- per-half CA coordinates / per-half distances
# ---------------------------------------------------------------------------
#
# HONEST STATEMENT OF WHAT CAN AND CANNOT BE BUILT
# ------------------------------------------------
# _flory_unfolded_graph is a DISTANCE model: d(i,j) = b * |i-j|^nu. For nu < 1 that is a
# statistical average over an ensemble of conformations, NOT the distance matrix of any
# single 3-D conformation, and it is not in general embeddable in R^3. So NO coordinate set
# reproduces the coil distances exactly for nu != 1.
#
# Therefore U6 provides BOTH, and the caller must know which it is using:
#   * coil_ca_distances  -- EXACT. The analytic d(i,j) the coil actually specifies. This is
#                           what edge attributes should consume, and it is the correct fix.
#   * coil_ca_coords     -- an EXTENDED-CHAIN realisation along +x with spacing b. Exact for
#                           nu == 1; for nu < 1 it reproduces the |i-j|=1 distances and
#                           OVERSTATES longer ones. Supplied only for code paths that demand
#                           a [N,3] tensor and cannot take a distance matrix. Never silently
#                           substituted for the exact path.

def coil_bond_length(ca_folded, mask=None):
    """The effective bond length b, computed EXACTLY as _flory_unfolded_graph computes it:
    the mean CA-CA neighbour distance of THIS protein, clamped at 1e-3.

    Keeping this identical matters -- if U6 used a different b than the node-feature coil,
    the edge distances and the node distances would describe two different coils.

    NOTE this inherits defect A4: b is fitted from the FOLDED structure, which re-injects
    folded geometry into the reference state. A4's fix (b = 5.82 A fixed) is a separate arm
    and is deliberately NOT applied here, so that U6 tracks whatever the node-feature coil
    does. If the A4 arm lands, _flory_unfolded_graph and this function must change TOGETHER.
    """
    if ca_folded.shape[0] <= 1:
        return torch.tensor(3.8, device=ca_folded.device, dtype=ca_folded.dtype)
    b = torch.linalg.norm(ca_folded[1:] - ca_folded[:-1], dim=-1).mean()
    return torch.clamp(b, min=1e-3)


def coil_ca_distances(N, b, nu, device, dtype=torch.float32):
    """EXACT analytic coil distance matrix [N, N]: d(i,j) = b * |i-j|^nu.

    Matches _flory_unfolded_graph including its + 1e-6 guard on the separation, so the
    diagonal is ~0.004*b rather than 0 -- see audit A8, a known and deliberate asymmetry
    that is reproduced here rather than quietly fixed, so U6 cannot become a second silent
    lever.
    """
    idx = torch.arange(N, device=device, dtype=dtype)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return b * torch.pow(sep + 1e-6, float(nu))


def coil_ca_coords(N, b, device, dtype=torch.float32):
    """An extended-chain realisation: residue i at (i*b, 0, 0).

    EXACT for nu == 1 only. For nu < 1 this overstates long-range distances (a real coil is
    compact; a straight line is not). Provided for API compatibility with code that requires
    [N, 3]; prefer coil_ca_distances wherever a distance matrix is accepted.

    GATE G2-U6: with the coil on, the unfolded half's CA coordinates returned by
    per_half_ca_coords must DIFFER from the folded half's. A straight line differs from any
    real fold, so torch.equal(folded, unfolded) is False -- asserted in gate_u5u6.py.
    """
    idx = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
    out = torch.zeros((N, 3), device=device, dtype=dtype)
    out[:, 0:1] = idx * b
    return out


def _normalise_ca_batch(ca_coords, B, N, n_f, n_u):
    """Bring ca_coords to [B, N, 3]. Accepts [N,3], [B,N,3], or folded-only [n_f,N,3]."""
    c = ca_coords
    if c.dim() == 2:                       # [N, 3] -> one protein, broadcast to the batch
        c = c.unsqueeze(0).expand(B, N, 3)
    elif c.shape[0] == n_f and n_f != B:   # folded-only coords were supplied
        c = torch.cat([c, c[-1:].expand(n_u, N, 3)], dim=0)
    if c.shape[0] != B:
        raise ValueError(
            "ca_coords batch %d matches neither B=%d nor n_folded=%d" % (c.shape[0], B, n_f))
    return c


def per_half_ca_coords(ca_coords, B, N, n_folded, cfg, nu=None):
    """U6: build ca_coords PER HALF instead of broadcasting the folded coordinates.

    ca_coords is [B, N, 3] (or [n_folded, N, 3], or [N, 3] for a single protein) and is
    consumed by PEM.get_edge_index for the CA cutoff and by
    PEMGraphTransformer.get_edge_attr for the RBF edge features. If the folded coordinates
    are used for the unfolded rows, the TRUE FOLDED CA DISTANCES become edge attributes in
    the unfolded pass -- and Phase 7 and the coil then cancel each other exactly (audit A6).

    Returns [B, N, 3] where folded rows keep their real coordinates and unfolded rows carry
    the coil realisation.

    Returns ca_coords UNCHANGED (identical object) when U5/U6 is off, when ca_coords is None,
    or when there are no unfolded rows -- so the off-path is byte-identical by construction.
    """
    if ca_coords is None:
        return None
    if not coil_edges_enabled(cfg):
        return ca_coords
    n_f, n_u = split_folded_rows(B, n_folded)
    if n_u <= 0:
        return ca_coords

    c = _normalise_ca_batch(ca_coords, B, N, n_f, n_u)
    out = c.clone()
    for r in range(n_f, B):
        # row r of the unfolded half corresponds to row (r - n_f) of the folded half when the
        # two halves are the same length, which is how both call sites build the batch.
        src_row = c[(r - n_f) % n_f] if n_f > 0 else c[r]
        b = coil_bond_length(src_row)
        out[r] = coil_ca_coords(N, b, c.device, c.dtype)
    return out


def per_half_ca_distances(ca_coords, B, N, n_folded, cfg, nu=None):
    """The EXACT counterpart of per_half_ca_coords: [B, N, N] pairwise CA distances where
    folded rows use real coordinates and unfolded rows use the analytic coil d = b*|i-j|^nu.

    THIS is the correct object for edge attributes -- it is exact for every nu, whereas the
    coordinate version is exact only at nu == 1. Any Phase 7 edge-feature work should consume
    this and not torch.cdist(per_half_ca_coords(...)).
    """
    if ca_coords is None:
        return None
    n_f, n_u = split_folded_rows(B, n_folded)
    c = _normalise_ca_batch(ca_coords, B, N, n_f, max(n_u, 0))
    dists = torch.cdist(c, c)
    if not coil_edges_enabled(cfg):
        return dists
    if n_u <= 0:
        return dists
    nu = float(getattr(cfg, 'flory_nu', 0.5)) if nu is None else float(nu)
    out = dists.clone()
    for r in range(n_f, B):
        src_row = c[(r - n_f) % n_f] if n_f > 0 else c[r]
        b = coil_bond_length(src_row)
        out[r] = coil_ca_distances(N, b, nu, c.device, c.dtype)
    return out
