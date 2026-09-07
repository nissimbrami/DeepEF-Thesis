"""Is the descriptor block already inside the ProtT5 embedding? CPU only.

WHY THIS GATES THE GPU SPEND
------------------------------------------------------------------------------------
The LORO arm costs GPU. It is only worth it if the descriptor block carries information
the feature vector does not already have. ProtT5 is a protein language model; its
per-residue embedding is known to encode physicochemical character. If a linear map from
the ProtT5 embedding reproduces the descriptor vector for a residue TYPE the map never
saw, then the descriptor block is largely REDUNDANT, and the LORO contrast would be
measuring a re-encoding of information the vector already carries rather than a new
channel. That is a real finding either way, and it costs seconds.

WHY THE SPLIT MUST BE RESIDUE-DISJOINT
------------------------------------------------------------------------------------
This is the whole methodological point and it is easy to get wrong.

Every occurrence of tryptophan in the corpus shares ONE descriptor vector -- the W row of
the 20xK table. So a ROW-RANDOM train/test split puts thousands of W rows in train and
more W rows in test with an IDENTICAL target. The regression then only has to recognise
"this is a W" from the embedding (trivial -- ProtT5 certainly encodes residue identity)
and emit the memorised W vector. R^2 would be near 1.0 and would mean NOTHING about
interpolation: it would be a residue-identity classifier scored as a regressor.

The honest split holds out a residue TYPE: fit on 19 types, predict the 20th, whose
descriptor vector was never seen at any point. That asks the actual question -- can the
descriptor vector be INTERPOLATED from the embedding -- and it is exactly the question
the LORO experiment asks of the model.

Both splits are reported side by side, because the gap between them IS the demonstration
that the contamination is real and large.

RESIDUE IDENTITY COMES FROM one_hot_encodings.pt, NOT from parsing aa_seq. Descriptor row
i == one-hot index i for every i < 20 -- enforced by construction in aa_descriptors.py
(_load_table_obj reorders by AA_MAP and never trusts file row order), so the join is exact
and needs no letter mapping that could silently mis-align.

WHAT R^2 MEANS HERE
------------------------------------------------------------------------------------
R^2 is computed against the mean of the TRAINING targets (the 19 seen residues). A single
held-out residue has one target vector, so its own variance is zero and an R^2 against
itself is undefined. The baseline being beaten is therefore "predict the average amino
acid" -- which is precisely the one-hot arm's situation for an unseen residue.

  R^2 near 1.0  ->  descriptors are redundant given ProtT5; report it, do not spend GPU.
  R^2 near 0.0  ->  ProtT5 cannot linearly place an unseen residue; the block is new
                    information and LORO is worth running.
  R^2 negative  ->  worse than predicting the training mean; strongly not redundant.
"""
from __future__ import print_function

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())


def per_residue_embeddings(tensor_root, max_proteins, rng, n_variants):
    """Mean ProtT5 embedding per residue TYPE (one-hot index), pooled over the corpus.

    One vector per residue type is what the question needs: "given how ProtT5 represents
    this residue, can its descriptor vector be recovered". Pooling over many proteins and
    positions averages out context, which is right here -- the descriptor table is
    context-free, so a context-carrying embedding must be reduced to a context-free
    summary before the two are comparable at all.
    """
    import torch

    def load(p):
        return torch.load(p, map_location='cpu', weights_only=True)

    sums = {}
    counts = {}
    prots = sorted(os.listdir(tensor_root))
    rng.shuffle(prots)
    used = 0
    for p in prots:
        if used >= max_proteins:
            break
        pdir = os.path.join(tensor_root, p)
        if not os.path.isdir(pdir):
            continue
        files = sorted(glob.glob(os.path.join(pdir, 'prott5_embeddings',
                                              'prott5_embedding_*.pt')),
                       key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
        ohp = os.path.join(pdir, 'one_hot_encodings.pt')
        mkp = os.path.join(pdir, 'mask_tensor.pt')
        if not files or not os.path.isfile(ohp):
            continue
        try:
            E = load(files[0]).float()          # [V, L, D]
            OH = load(ohp).float()              # [V, L, 21]
            M = load(mkp).float() if os.path.isfile(mkp) else None
        except Exception:
            continue
        if E.dim() != 3 or OH.dim() != 3:
            continue
        V = min(E.shape[0], OH.shape[0], n_variants)
        L = min(E.shape[1], OH.shape[1])
        if M is not None and M.numel() >= L:
            valid = (M[:L] > 0).numpy()
        else:
            valid = np.ones(L, dtype=bool)
        Ev = E[:V, :L].numpy()
        Ov = OH[:V, :L].numpy()
        aa = Ov.argmax(-1)                      # [V, L] one-hot index
        has = Ov.max(-1) > 0                    # padded positions are all-zero
        for v in range(V):
            for i in range(L):
                if not valid[i] or not has[v, i]:
                    continue
                a = int(aa[v, i])
                if a >= 20:                     # index 20 is the padding/unknown slot
                    continue
                if a not in sums:
                    sums[a] = np.zeros(Ev.shape[2], dtype=np.float64)
                    counts[a] = 0
                sums[a] += Ev[v, i]
                counts[a] += 1
        used += 1

    idxs = sorted(sums.keys())
    X = np.stack([sums[a] / counts[a] for a in idxs])
    return idxs, X, [counts[a] for a in idxs], used


def ridge_fit_predict(Xtr, Ytr, Xte, lam):
    """Ridge with an intercept, standardised on TRAIN statistics only."""
    mu, sg = Xtr.mean(0), Xtr.std(0)
    sg = np.where(sg == 0, 1.0, sg)
    Ztr = np.hstack([(Xtr - mu) / sg, np.ones((Xtr.shape[0], 1))])
    Zte = np.hstack([(Xte - mu) / sg, np.ones((Xte.shape[0], 1))])
    A = Ztr.T.dot(Ztr) + lam * np.eye(Ztr.shape[1])
    A[-1, -1] -= lam                       # never penalise the intercept
    W = np.linalg.solve(A, Ztr.T.dot(Ytr))
    return Zte.dot(W)


def r2_vs_train_mean(y_true, y_pred, train_mean):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - train_mean) ** 2))
    if ss_tot == 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tensor_root',
                    default='data/Processed_K50_dG_datasets/training_data')
    ap.add_argument('--csv', default='data/aa_descriptors_pca16.csv',
                    help='descriptor table. PCA-16 is the block that enters the model '
                         'under --aa_descriptors pca16.')
    ap.add_argument('--max_proteins', type=int, default=60)
    ap.add_argument('--n_variants', type=int, default=4,
                    help='variants per protein to pool (positions dominate the count)')
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    A = ap.parse_args()

    rng = np.random.RandomState(A.seed)
    print('=' * 78)
    print('DESCRIPTOR / ProtT5 REDUNDANCY PROBE  (CPU only)')
    print('  question: can the descriptor block be linearly predicted from ProtT5,')
    print('            for a residue type the map never saw?')
    print('  descriptor table: %s' % A.csv)
    print('=' * 78)

    from aa_descriptors import load_descriptor_table, descriptor_labels
    dlabels = descriptor_labels(A.csv)
    table = load_descriptor_table(A.csv, canonical_only=True).numpy().astype(float)
    print('  descriptor table shape : %s   (canonical 20 rows)' % (table.shape,))

    idxs, X, counts, used = per_residue_embeddings(
        A.tensor_root, A.max_proteins, rng, A.n_variants)
    labels = [dlabels[i] for i in idxs]
    print('  proteins pooled        : %d' % used)
    print('  residue types recovered: %d  %s' % (len(labels), ''.join(labels)))
    print('  ProtT5 embedding dim   : %d' % X.shape[1])
    print('  residue occurrences    : min %d  median %d  max %d'
          % (min(counts), int(np.median(counts)), max(counts)))
    if len(labels) < 15:
        sys.stderr.write('too few residue types recovered; aborting\n')
        return 2

    Y = np.stack([table[i] for i in idxs])
    # standardise the TARGET per column so R^2 is comparable across descriptor columns
    Ys = Y.std(0)
    Y = (Y - Y.mean(0)) / np.where(Ys == 0, 1.0, Ys)
    print('  aligned residue types  : %d   descriptor dim: %d' % (Y.shape[0], Y.shape[1]))

    # ------------------------------------------------------- residue-disjoint (honest)
    print('\nRESIDUE-DISJOINT leave-one-residue-out  (the HONEST split)')
    print('  fit on %d residue types, predict the held-out one, whose descriptor'
          % (len(labels) - 1))
    print('  vector was never seen at any point during the fit.')
    print('  %-6s %12s' % ('held', 'R^2'))
    scores = {}
    for k in range(len(labels)):
        tr = [i for i in range(len(labels)) if i != k]
        P = ridge_fit_predict(X[tr], Y[tr], X[k:k + 1], A.lam)
        scores[labels[k]] = r2_vs_train_mean(Y[k], P[0], Y[tr].mean(0))
        print('  %-6s %12.4f' % (labels[k], scores[labels[k]]))
    vals = np.array([scores[a] for a in labels], dtype=float)
    disj_mean = float(np.mean(vals))
    disj_med = float(np.median(vals))
    print('  ' + '-' * 24)
    print('  MEAN   R^2 (residue-disjoint) : %+.4f' % disj_mean)
    print('  MEDIAN R^2 (residue-disjoint) : %+.4f' % disj_med)
    for r in ('W', 'P'):
        if r in scores:
            print('  %s  (a LORO arm)               : %+.4f' % (r, scores[r]))

    # --------------------------------------- row-random (contaminated, shown for contrast)
    print('\nROW-RANDOM split  (CONTAMINATED -- shown for contrast, NOT as a result)')
    print('  every occurrence of a residue shares ONE descriptor vector, so this split')
    print('  puts the same target in train and test and measures memorisation.')
    reps = 400
    ridx = rng.randint(0, len(labels), size=reps)
    Xr, Yr = X[ridx], Y[ridx]
    perm = rng.permutation(reps)
    cut = int(0.7 * reps)
    tr, te = perm[:cut], perm[cut:]
    Pr = ridge_fit_predict(Xr[tr], Yr[tr], Xr[te], A.lam)
    row_r2 = r2_vs_train_mean(Yr[te], Pr, Yr[tr].mean(0))
    print('  R^2 (row-random, %d rows)     : %+.4f' % (reps, row_r2))

    print('\n' + '=' * 78)
    print('READING')
    print('  row-random       R^2 = %+.4f   <- inflated by shared targets; NOT a result'
          % row_r2)
    print('  residue-disjoint R^2 = %+.4f mean / %+.4f median   <- the real number'
          % (disj_mean, disj_med))
    print('  contamination gap    = %+.4f' % (row_r2 - disj_mean))
    print('')
    if disj_mean > 0.80:
        print('  VERDICT: descriptors are LARGELY REDUNDANT given ProtT5. A linear map')
        print('  from the embedding recovers an UNSEEN residue\'s descriptor vector, so')
        print('  the LORO contrast would re-encode information the vector already has.')
        print('  Report this; do not spend GPU.')
    elif disj_mean > 0.40:
        print('  VERDICT: PARTIALLY redundant. ProtT5 carries much of the descriptor')
        print('  signal but not all of it. LORO is defensible, but the expected effect')
        print('  is bounded by the unexplained fraction.')
    else:
        print('  VERDICT: NOT redundant. ProtT5 cannot linearly place an unseen residue')
        print('  in descriptor space. The descriptor block is a genuinely new channel')
        print('  and LORO is worth the GPU.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
