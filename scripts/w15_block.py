"""W15 -- the side-chain packing block, computed PER VARIANT from shared coordinates.

THE PROBLEM I FOUND BEFORE WIRING ANYTHING. Coordinates are shared: one protein has ONE
[62,4,3] backbone and 3,949 variants that differ only in one_hot. So a feature computed from
the packed WT structure alone is IDENTICAL for every variant and CANCELS EXACTLY in
ddG = dG_mut - dG_wt. It would be inert -- the W5 bug, again.

THE FIX. Two of the four quantities are environment-only and one is residue-only, so split them:

    env_density   [N] : how crowded position i is. From COORDINATES only, so shared across
                        variants -- but it MULTIPLIES the residue term below, and the product
                        does vary with the mutation.
    reach(aa)     [N] : how far the side chain of THIS residue extends from CB. From one_hot,
                        so it changes with the mutation.

    col0 = reach                      varies with the mutant (one_hot)
    col1 = env_density                shared, but see col2/col3
    col2 = reach * env_density        THE INTERACTION -- a big residue in a crowded pocket
    col3 = clash proxy: relu(reach - free_space)   nonzero only when the mutant does not fit

col2 and col3 are the load-bearing ones: they are LARGE when a bulky residue is placed in a
tight pocket and SMALL when it is placed on the surface, so they differ between WT and mutant
at the mutated position and survive ddG.

WHY THIS IS NOT one_hot @ T. col2 and col3 are products of a per-residue term with a
per-POSITION geometric term. Two tryptophans at different positions get different values.
Measured on the real reconstructions: one-hot alone explains only 0.443/0.488/0.245 of
sc_sasa/contacts/clash, so 55-76% of that signal is geometry.

STATE DEPENDENCE. env_density comes from the folded coordinates, so all four columns are ZERO
in the unfolded pass -- an extended chain has no packing. The folded-minus-unfolded difference
IS the packing term, which is what makes this a real state-dependent lever.
"""
import torch

# per-residue side-chain reach from CB, Angstrom, in the canonical 20-letter order ACDEFGHIKLMNPQRSTVWY
_REACH = torch.tensor([
    0.00,  # A  (CB is the whole side chain)
    1.90,  # C
    1.60,  # D
    2.80,  # E
    3.40,  # F
    0.00,  # G  (no CB)
    3.10,  # H
    1.90,  # I
    3.50,  # K
    2.10,  # L
    3.20,  # M
    1.60,  # N
    1.20,  # P
    2.70,  # Q
    4.50,  # R
    0.90,  # S
    1.10,  # T
    1.60,  # V
    4.20,  # W
    4.00,  # Y
], dtype=torch.float32)

_R_ENV = 10.0     # neighbour radius for the density term, Angstrom (model units: 1.0)
_CAP   = 20.0     # normaliser; a fixed constant, never N -- normalising by N would inject length


def w15_block(x, one_hot, mask=None, folded=True, device=None, dtype=torch.float32):
    """Return [N,4]: reach, env_density, reach*env_density, clash.

    x        [N,4,3] coordinates, atom order (N, CA, C, CB)
    one_hot  [N,>=20]
    folded   False -> all four columns are exactly zero (an extended chain has no packing)
    """
    dev = device if device is not None else one_hot.device
    N = one_hot.shape[0]
    if not folded:
        return torch.zeros(N, 4, device=dev, dtype=dtype)

    oh = one_hot[:, :20].to(dev, dtype)
    reach = (oh @ _REACH.to(dev, dtype)).unsqueeze(1)          # [N,1] depends on the MUTANT

    cb = x[:, 3, :].to(dev, dtype)
    d = torch.cdist(cb, cb)                                     # [N,N]
    v = (mask.to(dev, dtype).reshape(-1) if mask is not None else torch.ones(N, device=dev, dtype=dtype))
    near = ((d < _R_ENV) & (d > 0)).to(dtype) * v.unsqueeze(0) * v.unsqueeze(1)
    env = (near.sum(1, keepdim=True) / _CAP).clamp(0.0, 2.0)    # [N,1] shared across variants

    inter = reach * env                                          # [N,1] the interaction term
    # free space: distance to the nearest neighbour, minus a contact radius
    dm = d + torch.eye(N, device=dev, dtype=dtype) * 1e6
    nearest = dm.min(1, keepdim=True).values.clamp(max=12.0)
    clash = torch.relu(reach - (nearest - 3.8)) / 4.0            # [N,1] only when it does not fit

    out = torch.cat([reach / 4.5, env, inter / 4.5, clash], dim=1)
    return out * v.unsqueeze(1)
