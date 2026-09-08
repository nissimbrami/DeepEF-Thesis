#!/usr/bin/env python3
"""W15 -- side-chain features from FASPR-reconstructed structures.

WHY THIS IS NOT ANOTHER DESCRIPTOR TABLE. The decisive arithmetic: any per-residue descriptor
table is one_hot @ T, a rank-1 projection of a block fc1 already receives, so it cannot add
information. That killed LORO, W6, and two of W12's four columns.

These four columns are NOT functions of residue type. Two tryptophans at different positions get
different values because the GEOMETRY differs -- and the geometry comes from a real repacked
side chain, not from a lookup.

STATE DEPENDENCE IS THE WHOLE POINT. All four are computed on the FOLDED structure and are ZERO
in the unfolded state, so they appear in E_folded - E_unfolded rather than cancelling. An earlier
burial implementation was inert precisely because it was computed from the same coordinates in
both states and cancelled exactly.

COLUMNS
  0 sc_sasa      side-chain solvent-accessible surface area (Shrake-Rupley), A^2, normalised
  1 delta_asa    folded minus extended-reference ASA -- how much surface burial costs
  2 contacts     heavy-atom neighbours within 5 A of any side-chain atom
  3 clash        sum of van der Waals overlap with non-bonded neighbours

freesasa is not installed; Bio.PDB.SASA.ShrakeRupley is and is the same algorithm.
"""
import argparse, glob, os, sys, json
import numpy as np

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
except Exception as e:
    print("Biopython missing:", e); sys.exit(2)

BACKBONE = {'N','CA','C','O','OXT'}
# Tien et al. 2013 theoretical maximum ASA per residue (A^2), for the extended reference
MAXASA = {'ALA':129,'ARG':274,'ASN':195,'ASP':193,'CYS':167,'GLN':225,'GLU':223,'GLY':104,
          'HIS':224,'ILE':197,'LEU':201,'LYS':236,'MET':224,'PHE':240,'PRO':159,'SER':155,
          'THR':172,'TRP':285,'TYR':263,'VAL':174}
VDW = {'C':1.70,'N':1.55,'O':1.52,'S':1.80}

def one_protein(path, sr, parser):
    st = parser.get_structure('x', path)
    try: sr.compute(st, level='A')
    except Exception: return None
    res = [r for r in st.get_residues() if r.get_resname() in MAXASA]
    if not res: return None
    # all heavy atoms once, for neighbour queries
    allat = [a for r in res for a in r if a.element != 'H']
    P = np.array([a.get_coord() for a in allat])
    owner = np.array([i for i, r in enumerate(res) for a in r if a.element != 'H'])
    rad = np.array([VDW.get(a.element, 1.7) for a in allat])
    out = np.zeros((len(res), 4), dtype=np.float32)
    for i, r in enumerate(res):
        sc = [a for a in r if a.get_name() not in BACKBONE and a.element != 'H']
        if not sc:                                   # glycine
            out[i] = [0.0, 0.0, 0.0, 0.0]; continue
        sasa = float(sum(getattr(a, 'sasa', 0.0) for a in sc))
        mx = MAXASA[r.get_resname()]
        out[i, 0] = sasa / mx                                    # relative side-chain SASA
        out[i, 1] = max(0.0, (mx - sasa) / mx)                   # buried fraction = delta ASA
        C = np.array([a.get_coord() for a in sc])
        d = np.linalg.norm(P[None, :, :] - C[:, None, :], axis=-1)
        other = owner[None, :] != i
        out[i, 2] = float(((d < 5.0) & other).any(0).sum()) / 50.0   # contacts, normalised
        rsum = (np.array([VDW.get(a.element, 1.7) for a in sc])[:, None] + rad[None, :])
        ov = np.clip(rsum - d, 0, None) * other
        out[i, 3] = float(ov.sum()) / 10.0                        # steric clash
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packed', default='/home/nissimb/tools/w15_packed')
    ap.add_argument('--out', default='/home/nissimb/tools/w15_features')
    ap.add_argument('--limit', type=int, default=0)
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    sr, parser = ShrakeRupley(), PDBParser(QUIET=True)
    fs = sorted(glob.glob(A.packed + '/*.pdb'))
    if A.limit: fs = fs[:A.limit]
    ok = fail = 0; stats = []
    for f in fs:
        name = os.path.basename(f)[:-4]
        try:
            F = one_protein(f, sr, parser)
            if F is None: fail += 1; continue
            np.save(os.path.join(A.out, name + '.npy'), F)
            stats.append(F); ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3: print('  FAIL', name, type(e).__name__, e)
    print(f"wrote {ok} feature files, {fail} failed -> {A.out}")
    if stats:
        S = np.concatenate(stats, 0)
        names = ['sc_sasa', 'buried_frac', 'contacts', 'clash']
        print(f"\n{'column':<14}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
        for j, n in enumerate(names):
            print(f"{n:<14}{S[:,j].mean():>9.4f}{S[:,j].std():>9.4f}{S[:,j].min():>9.4f}{S[:,j].max():>9.4f}")
        print(f"\nresidues total: {len(S)}")
        # a column with zero variance carries nothing
        for j, n in enumerate(names):
            if S[:, j].std() < 1e-6: print(f"  WARNING {n} has ZERO variance -- it cannot help")
    return 0

if __name__ == '__main__':
    sys.exit(main())
