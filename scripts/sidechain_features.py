"""Side-chain feature block (W12) — the fix for the hydrophobic ranking deficit.

WHY THIS EXISTS. Measured over 10 checkpoints (scripts/k13_muttype.py):
    slope to hydrophobic destinations   0.27-0.31
    slope to charged/polar destinations 0.53-0.57
    corr(slope, Kyte-Doolittle)    = -0.734   sign-consistent 10/10
    corr(spearman, Kyte-Doolittle) = -0.740
Rank accuracy degrades in LOCKSTEP with the slope, so this is LOST INFORMATION, not a
rescalable calibration error. No affine correction can recover it (results/02_findings/
SIDECHAIN_DESIGN.md, PER_MUTATION.md).

MECHANISM — CORRECTED. The original design blamed side-chain BULK. That is refuted by its own
shuffle test: corr(slope, volume) = -0.372 fails at P=0.069 while corr(slope, KD) = -0.765 passes
at P=0.0000, and cross-validated R^2 is volume 0.031 vs hydropathy 0.666 vs TRANSFER FREE ENERGY
0.753. So the driver is the hydrophobic effect -- the free energy of moving a side chain from water
into the protein core -- not steric packing. This block encodes that, and volume enters only as a
secondary term because burial is an area-scaled quantity.

WHY THE MODEL CANNOT SEE IT. The dataset stores 4 backbone atoms per residue (N, CA, C, CB) and
NO side chains. Burying a large hydrophobic is dominated by side-chain solvation, which CB-only
geometry cannot represent. This block supplies the per-residue chemistry the coordinates lack.

THE METRIC RULE. dG_transfer is a per-residue-TYPE constant, so at any unmutated position it is
identical between wild type and mutant and cancels exactly in ddG. What does NOT cancel is
(a) the mutated position's delta, and (b) the burial-weighted product, because burial is a
FOLDED-state quantity that is zero in the unfolded state. So the block acts on BOTH channels and
must be reported on ddG (for the mutation delta) AND on dG/b_p (for the burial-weighted term).

BLOCK LAYOUT, 4 columns inserted at offset 48 (between Fb and emb, so the right-anchored emb and
one_hot slices never move):
    0  dG_transfer            per-residue-type, kcal/mol, water -> octanol (Fauchere-Pliska)
    1  burial * dG_transfer   the actual hydrophobic driving force; ZERO in the unfolded state
    2  sidechain_volume       van der Waals volume, z-scored (secondary, area-scaling)
    3  burial * volume        packing proxy; ZERO in the unfolded state
Only fc1_gcn / fc1_gat grow. fc2_*, inst_norm1, inst_norm2 and fc_in_dim must NOT.

NOTE the GCN branch reads x[:, :32] only, so this block reaches the GAT branch alone -- the same
constraint that applies to W5/W6/W9/W11.
"""
import torch

AA = 'ACDEFGHIKLMNPQRSTVWY'

# Fauchere & Pliska (1983) water->octanol transfer free energies, kcal/mol.
# POSITIVE = hydrophobic = favours burial. This is the quantity that cross-validated at R^2 0.753
# against the per-residue slope, versus 0.031 for volume.
DG_TRANSFER = {
    'A':  0.31, 'C':  1.54, 'D': -0.77, 'E': -0.64, 'F':  1.79,
    'G':  0.00, 'H':  0.13, 'I':  1.80, 'K': -0.99, 'L':  1.70,
    'M':  1.23, 'N': -0.60, 'P':  0.72, 'Q': -0.22, 'R': -1.01,
    'S': -0.04, 'T':  0.26, 'V':  1.22, 'W':  2.25, 'Y':  0.96,
}

# Side-chain van der Waals volume, A^3 (Creighton). Secondary term.
VOLUME = {
    'A':  88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G':  60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S':  89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
}

SIDECHAIN_DIM = 4


def _z(d):
    v = torch.tensor([d[a] for a in AA], dtype=torch.float32)
    return (v - v.mean()) / v.std(unbiased=False)


_DGT = _z(DG_TRANSFER)     # z-scored so the block cannot dominate one-hot by scale.
_VOL = _z(VOLUME)          # The LORO descriptor arm COLLAPSED because a table with sd=1.0 and
                           # max|v|=6.0 carried ~16.8x the energy of one-hot. Both tables here are
                           # z-scored to sd=1 across only 20 residues and are 4 columns, not 726,
                           # so the energy ratio is ~4/20 of one-hot rather than 16.8x.


def sidechain_dim(cfg):
    return SIDECHAIN_DIM if getattr(cfg, 'sidechain_features', False) else 0


def sidechain_block(one_hot, burial=None, folded=True, device=None, dtype=torch.float32):
    """[N, 4] side-chain chemistry block.

    one_hot : [N, 20] residue identity (the 21st stored column is dropped upstream at train.py:317)
    burial  : [N] or [N,1] burial fraction, or None. MUST be None/zero for the unfolded pass --
              an unfolded chain has no core, so the burial-weighted columns are the folded-minus-
              unfolded difference and therefore the hydrophobic driving force itself.
    folded  : when False the burial-weighted columns are forced to zero regardless of `burial`,
              so a caller cannot accidentally leak folded geometry into the reference state.
    """
    dev = device or one_hot.device
    oh = one_hot[:, :20].to(dtype)
    dgt = (oh @ _DGT.to(dev, dtype)).unsqueeze(1)      # [N,1]
    vol = (oh @ _VOL.to(dev, dtype)).unsqueeze(1)      # [N,1]
    if folded and burial is not None:
        b = burial.to(dev, dtype).reshape(-1, 1)
    else:
        b = torch.zeros_like(dgt)
    return torch.cat([dgt, b * dgt, vol, b * vol], dim=1)
