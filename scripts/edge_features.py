"""W7 / ITEM 22 -- pairwise EDGE ATTRIBUTES for the GAT branch.

The half of W7 that was missing. --gcn_span and --gcn_bidir (the edge SET) already
exist and are gated by scripts/gate_w7.py, 11/11. rbf_expand already exists in
train_utils and already concatenates. What did NOT exist is the edge ATTRIBUTE
path: 20 (source one-hot) + 20 (destination one-hot) + 16 (RBF over the CA-CA
distance) = 56 dims, handed to GATv2Conv as edge_dim.

This module is standalone and imports nothing from the DeepEF tree except torch,
so gate_w7_edge.py can exercise every arithmetic claim on a laptop with no
cluster, no dataset and no checkpoint.

---------------------------------------------------------------------------
THE TWO MEASURED FAILURES THIS FILE EXISTS TO NOT REPEAT
---------------------------------------------------------------------------
U8  The RBF bank must be CONCATENATED, never summed. A previous implementation
    summed the M responses back to the original width; a 5 A contact and a 15 A
    non-contact both came back 4.649 -- flat past 5 A, the model blind to
    distance. build_edge_attr therefore returns width 56 and NEVER reduces the
    bank. The runnable assertion k(2A) > k(8A) > k(15A) in gate_w7_edge.py exits
    non-zero if that ever regresses.

U6  The unfolded pass must NOT receive folded coordinates. In this tree the
    folded and unfolded graphs are stacked along the BATCH dimension
    (train.py get_deltaG: all_graph_minibatch = torch.cat([folded, unfolded],
    dim=0)), so a single ca_coords tensor .expand()ed over that batch would give
    the unfolded rows the folded protein's geometry -- exactly the failure the
    coil lever exists to remove. unfolded_reference_coords() below builds the
    unfolded half's geometry from SEQUENCE SEPARATION ONLY, and
    stack_half_coords() is the only supported way to assemble the [2n, N, 3]
    tensor. U6 proper (the per-half fix inside train.py) is owned by another
    worker; assert_u6_compatible() RAISES rather than silently mixing the states.
"""

import torch


# --------------------------------------------------------------------------
# 1. The RBF bank.  Signature and constants match train_utils.rbf_expand
#    EXACTLY so the two can never drift.  train_utils.rbf_expand is the one
#    the model calls; this copy exists so the gate runs standalone.
# --------------------------------------------------------------------------
RBF_DIM_DEFAULT = 16
RBF_LO_DEFAULT = 0.0
RBF_HI_DEFAULT = 20.0


def rbf_expand(d, n=RBF_DIM_DEFAULT, lo=RBF_LO_DEFAULT, hi=RBF_HI_DEFAULT):
    """Expand a distance into a bank of n radial basis functions. CONCATENATED.

    d    : [...]      any shape of distances, Angstroms
    ->     [..., n]   one response per centre.  The trailing dim is NEW; nothing
                      is summed, meaned or otherwise reduced back to the input
                      width.  That reduction is U8 and it is why this docstring
                      is longer than the function.

    Byte-identical to train_utils.rbf_expand as of srclive: same linspace, same
    width = (hi-lo)/n, same 2*width**2 denominator.  gate_w7_edge.py asserts the
    two agree with torch.equal when train_utils is importable.
    """
    centers = torch.linspace(lo, hi, n, device=d.device, dtype=d.dtype)
    width = (hi - lo) / n
    return torch.exp(-((d.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))


# --------------------------------------------------------------------------
# 2. Edge attributes: src one-hot (20) + dst one-hot (20) + RBF (16) = 56
# --------------------------------------------------------------------------
EDGE_ONE_HOT_DIM = 20
EDGE_ATTR_DIM = 2 * EDGE_ONE_HOT_DIM + RBF_DIM_DEFAULT   # 56


def edge_attr_dim(rbf_dim=RBF_DIM_DEFAULT):
    """The edge_dim to hand GATv2Conv. 20 + 20 + rbf_dim."""
    return 2 * EDGE_ONE_HOT_DIM + int(rbf_dim)


def build_edge_attr(edge_index, one_hot_flat, ca_coords, N,
                    rbf_dim=RBF_DIM_DEFAULT, rbf_lo=RBF_LO_DEFAULT,
                    rbf_hi=RBF_HI_DEFAULT):
    """Build the [E, 56] edge attribute matrix.

    edge_index   : [2, E] long, FLAT indices into the B*N node axis, exactly the
                   form hydro_net.PEM.get_edge_index already returns.
    one_hot_flat : [B*N, 20] the residue one-hot block, read from the node tensor
                   at the RIGHT-ANCHORED index x[:, -20:].  Right-anchored, so it
                   is immune to every left-anchored block W5/W6 may add.
    ca_coords    : [B, N, 3] CA coordinates, ALREADY PER-HALF.  Pass None to get
                   a zero distance channel (the RBF of 0, not a zero vector) --
                   only correct when there is genuinely no geometry.
    N            : residues per graph, needed to invert the flat index.

    Returns [E, 2*20 + rbf_dim] float, dtype and device following one_hot_flat.

    Ordering is (src_one_hot, dst_one_hot, rbf).  It is DIRECTED and deliberately
    NOT symmetrised: GATv2Conv aggregates at dst, so the pair (src=LEU, dst=ASP)
    must be distinguishable from (src=ASP, dst=LEU).  Symmetrising here would
    throw away exactly the (identity, identity, distance) triple that the phase
    document names as the reason to build this at all.
    """
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be [2,E]; got %s" % (tuple(edge_index.shape),))
    if one_hot_flat.shape[-1] != EDGE_ONE_HOT_DIM:
        raise ValueError(
            "one_hot_flat must be [B*N,%d]; got %s. Read it right-anchored: x[:, -20:]."
            % (EDGE_ONE_HOT_DIM, tuple(one_hot_flat.shape)))

    src = edge_index[0]
    dst = edge_index[1]
    oh_src = one_hot_flat[src]                       # [E,20]
    oh_dst = one_hot_flat[dst]                       # [E,20]

    if ca_coords is None:
        dist = torch.zeros(src.shape[0], device=one_hot_flat.device,
                           dtype=one_hot_flat.dtype)
    else:
        b = torch.div(src, N, rounding_mode='floor')
        i = src % N
        j = dst % N
        delta = ca_coords[b, i] - ca_coords[b, j]
        dist = delta.norm(dim=-1).to(one_hot_flat.dtype)

    rbf = rbf_expand(dist, n=rbf_dim, lo=rbf_lo, hi=rbf_hi)   # [E,rbf_dim]
    return torch.cat([oh_src, oh_dst, rbf], dim=-1)


# --------------------------------------------------------------------------
# 3. U6 -- per-half coordinates.  The unfolded half never sees folded geometry.
# --------------------------------------------------------------------------
CA_ATOM_INDEX = 1          # atom order in this dataset is (N, CA, C, CB)
CB_ATOM_INDEX = 3          # glycine stores CA here; the standard substitute
DEFAULT_BOND_LENGTH = 3.8  # A, CA-CA


def folded_ca(x):
    """CA coordinates of the FOLDED state from the raw coordinate tensor.

    x : [N,4,3] (one protein) or [B,N,4,3].  Returns [N,3] / [B,N,3].
    This is the only function permitted to touch real coordinates for the
    folded half.
    """
    if x.dim() == 3:
        return x[:, CA_ATOM_INDEX, :]
    if x.dim() == 4:
        return x[:, :, CA_ATOM_INDEX, :]
    raise ValueError("expected [N,4,3] or [B,N,4,3]; got %s" % (tuple(x.shape),))


def unfolded_reference_coords(N, device=None, dtype=torch.float32, b=None,
                              nu=None, flory=False):
    """Geometry for the UNFOLDED half, built from SEQUENCE SEPARATION ALONE.

    This is the U6 guarantee in one function: nothing here reads a folded
    coordinate, so no folded geometry can leak into the unfolded pass.

    Two modes, matching the two unfolded references the tree already has:

      flory=False (default, matches the tridiagonal baseline
                   train_utils.get_unfolded_graph)
          A straight extended chain on the x axis with spacing b.  Then
          |r_i - r_j| = b*|i-j| exactly, so the RBF sees a monotone function of
          sequence separation and NOTHING else -- which is precisely what the
          tridiagonal reference state asserts about the unfolded chain.

      flory=True (matches train_utils._flory_unfolded_graph)
          |r_i - r_j| = b*|i-j|^nu.  Placing residue k at x = b*k^nu on a line
          reproduces that separation law exactly for every pair measured from
          residue 0, and is monotone in |i-j| for all pairs -- which is all the
          RBF channel can express anyway.  It is an ANALYTIC reference, not a
          sampled conformer: no randomness, so the unfolded pass stays
          deterministic across epochs exactly as the node-feature coil path is.

    b defaults to 3.8 A (CA-CA).  Pass the protein's own mean CA-CA bond length
    to match _flory_unfolded_graph, which scales its coil to the protein.
    """
    if b is None:
        b = DEFAULT_BOND_LENGTH
    idx = torch.arange(int(N), device=device, dtype=dtype)
    if flory:
        nu = 0.5 if nu is None else float(nu)
        if not (0.0 < nu <= 1.0):
            raise ValueError(
                "flory nu must be in (0,1]; got %s. (0.5 = ideal chain, "
                "~0.588 = self-avoiding walk.)" % (nu,))
        pos = float(b) * torch.pow(idx + 1e-6, nu)
    else:
        pos = float(b) * idx
    out = torch.zeros(int(N), 3, device=device, dtype=dtype)
    out[:, 0] = pos
    return out


def mean_ca_bond_length(x):
    """Protein's own mean CA-CA neighbour distance, matching
    _flory_unfolded_graph's choice of b.  x : [N,4,3]. Returns a 0-d tensor.

    NOTE this DOES read folded coordinates.  It is a single SCALAR summary of
    bond length, not per-residue geometry -- the same scalar
    _flory_unfolded_graph already takes -- so it carries no fold information.
    Using it keeps the edge path's coil consistent with the node path's coil.
    If a reviewer objects even to the scalar, pass b=3.8 and the two paths
    differ only in overall chain scale.
    """
    ca = folded_ca(x)
    if ca.shape[0] < 2:
        return torch.tensor(DEFAULT_BOND_LENGTH, device=ca.device, dtype=ca.dtype)
    return torch.linalg.norm(ca[1:] - ca[:-1], dim=-1).mean().clamp(min=1e-3)


def stack_half_coords(folded_coords, n_folded, n_unfolded, flory=False, nu=None,
                      b=None):
    """Assemble the [n_folded + n_unfolded, N, 3] ca_coords for the stacked batch.

    THE ONLY SUPPORTED WAY to build ca_coords for train.py's
        all_graph_minibatch = torch.cat([folded_graph, unfolded_graph], dim=0)

    folded_coords : [N,4,3] raw coords for THIS protein (all variants share the
                    backbone in this dataset), or [N,3] CA already extracted.
    n_folded      : number of folded rows (the mini-batch size)
    n_unfolded    : number of unfolded rows (equal to n_folded in get_deltaG)

    The folded rows get the real CA.  The unfolded rows get
    unfolded_reference_coords -- built from sequence separation only.  A single
    .expand() over the whole 2n batch is what U6 forbids, and is what this
    function exists to make unnecessary.
    """
    if folded_coords.dim() == 3:
        ca = folded_ca(folded_coords)                 # [N,3]
    elif folded_coords.dim() == 2 and folded_coords.shape[-1] == 3:
        ca = folded_coords
    else:
        raise ValueError(
            "folded_coords must be [N,4,3] or [N,3]; got %s"
            % (tuple(folded_coords.shape),))
    N = ca.shape[0]
    if b is None and flory and folded_coords.dim() == 3:
        b = float(mean_ca_bond_length(folded_coords))
    ref = unfolded_reference_coords(N, device=ca.device, dtype=ca.dtype,
                                    b=b, nu=nu, flory=flory)
    top = ca.unsqueeze(0).expand(int(n_folded), N, 3)
    bot = ref.unsqueeze(0).expand(int(n_unfolded), N, 3)
    return torch.cat([top, bot], dim=0).contiguous()


def assert_u6_compatible(edge_features, flory_unfolded, per_half_coords_available):
    """G3 -- no silent interaction.  RAISE, never warn.

    --edge_features writes CA distances into the edge channel.  If the unfolded
    half is handed folded coordinates, that channel says the unfolded chain has
    the folded protein's contact map -- which directly contradicts the coil
    lever, so the two flags together would measure nothing.  Combining them
    without the per-half fix is a silent scientific error, therefore it is a
    hard error.
    """
    if edge_features and flory_unfolded and not per_half_coords_available:
        raise RuntimeError(
            "[W7/U6] --edge_features with --flory_unfolded requires the per-half "
            "ca_coords fix (U6). Without it the unfolded pass receives FOLDED CA "
            "coordinates, so the edge RBF channel reports the folded contact map "
            "for the unfolded state and directly contradicts the coil lever. "
            "Use edge_features.stack_half_coords() to build ca_coords per half, "
            "or run the two flags separately.")
    return True


# --------------------------------------------------------------------------
# 4. The assertion, importable so nothing has to re-derive it
# --------------------------------------------------------------------------
def rbf_monotone_ok(rbf_fn=None, dists=(2.0, 8.0, 15.0)):
    """k(2A) > k(8A) > k(15A), strictly, where k is the response in the channel
    whose centre is nearest 2 A -- i.e. read the FIRST-CONTACT channel and check
    it decays with distance.

    This is the check that catches the summed-bank failure completely: a summed
    bank returns one number per distance and that number went FLAT (4.649 at
    both 5 A and 15 A).  A single channel of a concatenated bank cannot be flat
    unless the bank collapsed.

    Returns (ok: bool, values: list[float]).
    """
    fn = rbf_expand if rbf_fn is None else rbf_fn
    d = torch.tensor(list(dists), dtype=torch.float32)
    K = fn(d)
    if K.dim() != 2 or K.shape[0] != len(dists):
        return False, []
    if K.shape[1] == 1:
        return False, []                    # width 1 == the summed-bank failure
    ch = int(torch.argmax(K[0]))            # channel nearest the shortest distance
    vals = [float(K[k, ch]) for k in range(len(dists))]
    ok = all(vals[k] > vals[k + 1] for k in range(len(vals) - 1))
    return ok, vals
