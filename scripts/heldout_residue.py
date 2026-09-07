"""W10 -- held-out-residue evaluation harness.

THE CLAIM THIS EXISTS TO TEST
--------------------------------------------------------------------------------------
Ofir Ezrielev's central result is that a model trained on canonical residues predicts
NON-canonical effects, because the physicochemical descriptor space is CONTINUOUS. A
residue the model never saw is still a point in a space the model already understands,
so the model can place it by interpolation rather than needing an example of it.

That is a GENERALISATION claim about the representation. It is NOT a claim that any
number on MegaScale goes up.

WHY A CANONICAL PROXY
--------------------------------------------------------------------------------------
MegaScale contains ZERO non-canonical residues -- verified, not assumed: see
scripts/count_noncanonical.py, which scans every mutation CSV and reports the distinct
residue characters actually present. So the claim cannot be tested directly on this
corpus at all, with any amount of code.

The proxy that CAN be run today: hold a CANONICAL residue out of training, then predict
mutations INTO it. From the model's point of view a residue it has never been trained on
is exactly the situation a non-canonical residue creates -- an unseen point in descriptor
space. Tryptophan is the natural choice: it is the most distinctive residue in the
alphabet (largest, only two-ring aromatic; the descriptor matrix puts it furthest from
the centroid), so it is the HARDEST case, and 'we can place even W without seeing it' is
a stronger statement than the average residue would support.

WHAT SEPARATES THE TWO ARMS
--------------------------------------------------------------------------------------
The whole test is a contrast between two feature encodings, holding the data split fixed:

  one-hot only    : W is a column the model never saw fire. Every unseen residue is at
                    the same distance from every seen residue -- the encoding asserts
                    that all twenty residues are mutually equidistant, so there is
                    literally no information from which to place W. Predictions collapse
                    toward the training mean.
  descriptors     : W is a VECTOR near F and Y on aromaticity, ring atoms, pi-electrons
                    and volume. Those axes were trained on F, Y, H, L, I, ... so the
                    model can reach W by interpolation.

If descriptors help ONLY when the held-out residue's neighbours are in training, that is
the continuity claim, demonstrated.

HOW TO SCORE IT -- THE METRIC RULE, WHICH BITES HERE TOO
--------------------------------------------------------------------------------------
This project's biggest finding is that ddG cancels anything identical between WT and
mutant. A held-out-residue test does NOT cancel: the mutant residue is exactly what
differs, so the held-out residue's identity survives into ddG and ddG IS a legitimate
target here.

But there is a SECOND, subtler version of the same trap, and it was found by running
this harness rather than by reasoning about it in advance. MEASURED on 120 MegaScale
proteins holding out W: the two arms returned PCC identical to four decimals (0.2785 vs
0.2785) and identical a_p (0.1083), while b_p moved from -0.7752 to -0.4794 and MAE from
0.8372 to 0.8022.

The reason is structural, not statistical. With ONE held-out residue, EVERY held-out row
has the same mutant residue, so the mutant-side feature block is CONSTANT across all of
them. A constant feature can shift every prediction by the same amount; it cannot re-rank
them. Correlation is invariant to that shift, so PCC is fixed by the WT side alone and
must be identical in both arms -- by construction, whatever the encoding is worth.

So for a SINGLE held-out residue, PCC is structurally blind to the very lever under test,
and dPCC ~ 0 is the expected reading rather than a null result. The metric that can see
it is b_p (and MAE, which contains the offset). Scoring this on correlation would have
"disproved" the lever the same way the Flory coil was nearly discarded on a bad metric.

Hold out SEVERAL residues (--hold_out W,F,Y) to make the mutant side vary again; then,
and only then, PCC discriminates.

The harness prints all of PCC, MAE, a_p and b_p for exactly this reason, and says in
words which one is readable for the split you chose.

WHAT THIS ACTUALLY MEASURED (2026-09-07, ridge probe, MegaScale subsets)
--------------------------------------------------------------------------------------
Recorded here so the numbers are not lost and are not overstated. Ridge probe, canonical
20-row Mordred matrix (K=726), ddG_ML target, fitted on TRAIN rows only:

  hold-out   files  n_held   arm        PCC      MAE     a_p      b_p
  W          120     5359    one-hot   0.2785   0.8372  0.1083  -0.7752
  W          120     5359    desc      0.2785   0.8022  0.1083  -0.4794
  W,F,Y       80    11955    one-hot   0.3006   0.8262  0.1162  -0.7885
  W,F,Y       80    11955    desc      0.2779   0.7975  0.1116  -0.6045
  W,K,T       80    11345    one-hot   0.3849   0.8386  0.1408  -0.8077
  W,K,T       80    11345    desc      0.3572   0.8367  0.1485  -0.6915

The pattern is consistent across all three splits and should be reported as such:

  * |b_p| improves every time, and substantially: 0.775->0.479, 0.789->0.605,
    0.808->0.692. The descriptor space places an unseen residue on roughly the right
    ABSOLUTE scale, which one-hot cannot do at all because an unseen column never fired
    during fitting.
  * MAE improves every time, by less, because MAE contains that offset.
  * PCC is flat (single hold-out, where it is structurally blind -- see above) or
    slightly WORSE (-0.02 to -0.03) for multi-residue hold-outs.

So the honest statement is: the descriptor encoding fixes the OFFSET for an unseen
residue and does not improve RANK ORDER within it. That is a real and useful effect --
b_p is the term this project has identified as the dominant error, and the offset-removal
ceiling is 0.70-0.72 -- but it is NOT "descriptors beat one-hot on correlation", and it
must not be written up that way.

CAVEAT that bounds all six rows: this probe sees residue IDENTITY ONLY. No structure, no
context, no ProtT5. Its absolute numbers are far below the real model's and must never be
quoted as DeepEF performance. Only the CONTRAST between its two arms is meaningful,
because the data split and the estimator are held fixed across them.

USAGE
--------------------------------------------------------------------------------------
    # what the split looks like -- no model, no GPU, instant
    python scripts/heldout_residue.py --hold_out W --report_split

    # descriptor-space geometry of the held-out residue: who are its neighbours,
    # and are they in training?
    python scripts/heldout_residue.py --hold_out W --descriptor_neighbours

    # score a checkpoint (needs the checkpoint; CPU is fine but slow)
    python scripts/heldout_residue.py --hold_out W --ckpt path/to/best_model.pt

    # hold out a NON-canonical label -- works the moment such data exists
    python scripts/heldout_residue.py --hold_out SEP --csv data/aa_descriptors_open25.csv
"""
from __future__ import print_function

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MUT_RE = re.compile(r'^([A-Z]{1,3})(\d+)([A-Z]{1,3})$')
NAME_RE = re.compile(r'_([A-Z]{1,3})(\d+)([A-Z]{1,3})(?:_|$)')


def parse_mutation(code):
    """'L7S' -> ('L', 7, 'S'). Also accepts 3-letter labels, so 'S7SEP' is not a
    special case -- the OPEN alphabet has to survive the mutation parser too, or the
    harness would silently drop exactly the rows it exists to score.

    Returns None for anything that is not a substitution (wt rows, insertions,
    deletions, multi-mutants), which the caller counts rather than discards silently.
    """
    if not isinstance(code, str):
        return None
    c = code.strip()
    m = MUT_RE.match(c)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    m = NAME_RE.search(c)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def split_by_held_out(rows, hold_out):
    """Partition (wt, pos, mut) rows into TRAIN and HELD-OUT.

    A row goes to HELD-OUT if the MUTANT residue is held out. Rows whose WT residue is
    held out also leave training -- otherwise the model still sees the residue, as the
    thing being mutated away from, and the 'never seen' premise is false. This is the
    detail that makes or breaks the experiment and it is easy to get wrong: excluding
    only mutations INTO W still shows W to the model in every W->X row.
    """
    hold = set(hold_out)
    train, held, wt_side = [], [], []
    for r in rows:
        p = parse_mutation(r.get('mut_type') or r.get('name') or '')
        if p is None:
            continue
        wt, pos, mut = p
        rr = dict(r); rr['_wt'] = wt; rr['_pos'] = pos; rr['_mut'] = mut
        if mut in hold:
            held.append(rr)
        elif wt in hold:
            wt_side.append(rr)          # removed from training, not scored
        else:
            train.append(rr)
    return train, held, wt_side


def descriptor_neighbours(hold_out, csv_path, k=5):
    """Who sits next to the held-out residue in descriptor space, and are they in
    training? This is the whole mechanism of the claim, made inspectable WITHOUT a model:
    if W's nearest neighbours are F and Y and both are in training, the descriptor space
    has somewhere to interpolate W from. If its neighbours were also held out, the test
    would be measuring extrapolation, not interpolation, and would be expected to fail.
    """
    import torch
    from aa_descriptors import load_descriptor_table, descriptor_labels
    labels = descriptor_labels(csv_path)
    table = load_descriptor_table(csv_path, canonical_only=False)
    hold = set(hold_out)
    out = []
    for h in hold_out:
        if h not in labels:
            out.append((h, None, 'label %r is not in %s' % (h, csv_path)))
            continue
        i = labels.index(h)
        d = torch.linalg.norm(table - table[i:i + 1], dim=1)
        order = torch.argsort(d).tolist()
        nn = [(labels[j], float(d[j]), labels[j] not in hold)
              for j in order if j != i][:k]
        out.append((h, nn, None))
    return out


def _tofloat(a):
    """Coerce to float, mapping any non-numeric cell to NaN.

    MegaScale writes '-' where a ddG could not be fitted. Those must become NaN and be
    DROPPED, never 0.0: a fake zero reads as a real neutral mutation and would flatten
    every slope toward compression -- the exact a_p artefact this project measures.
    """
    import numpy as np
    out = np.empty(len(a), dtype=float)
    for i, v in enumerate(a):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out



def per_protein_ab(y_true, y_pred):
    """b_p (offset) and a_p (slope) of pred ~= a_p*true + b_p, by least squares.

    The project's calibration decomposition: b_p is the WT error, a_p the compression.
    Reported per held-out-residue group because a lever that acts on the reference state
    moves b_p while leaving pooled ddG correlation flat.
    """
    import numpy as np
    y_true = _tofloat(list(y_true))
    y_pred = _tofloat(list(y_pred))
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size < 2 or float(np.std(y_true)) == 0.0:
        return float('nan'), float('nan'), int(y_true.size)
    a, b = np.polyfit(y_true, y_pred, 1)
    return float(a), float(b), int(y_true.size)


def metrics(y_true, y_pred):
    import numpy as np
    y_true = _tofloat(list(y_true))
    y_pred = _tofloat(list(y_pred))
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    n = int(y_true.size)
    if n < 2:
        return dict(n=n, pcc=float('nan'), mae=float('nan'), rmse=float('nan'),
                    a_p=float('nan'), b_p=float('nan'))
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 \
        else float('nan')
    a, b, _ = per_protein_ab(y_true, y_pred)
    return dict(n=n, pcc=pcc,
                mae=float(np.mean(np.abs(y_true - y_pred))),
                rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                a_p=a, b_p=b)


def load_corpus(mut_dir, limit=None):
    """Every substitution row in the mutation CSVs, as plain dicts."""
    import pandas as pd
    files = sorted(glob.glob(os.path.join(mut_dir, '*.csv')))
    if limit:
        files = files[:limit]
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        prot = os.path.splitext(os.path.basename(f))[0]
        cols = [c for c in ('name', 'mut_type', 'ddG_ML', 'deltaG', 'aa_seq') if c in df.columns]
        for rec in df[cols].to_dict('records'):
            rec['_protein'] = prot
            rows.append(rec)
    return rows, files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hold_out', default='W',
                    help='comma-separated residue LABELS to hold out of training. '
                         'One-letter or multi-char (SEP). Default W -- the hardest and '
                         'most distinctive canonical residue.')
    ap.add_argument('--mut_dir', default='data/Processed_K50_dG_datasets/mutation_datasets')
    ap.add_argument('--csv', default='data/aa_descriptors.csv',
                    help='descriptor CSV; pass the 25-row open table to hold out SEP/TPO/...')
    ap.add_argument('--limit_files', type=int, default=0,
                    help='scan only the first N protein CSVs (fast smoke test)')
    ap.add_argument('--report_split', action='store_true',
                    help='report the train/held-out counts and exit -- no model needed')
    ap.add_argument('--descriptor_neighbours', action='store_true',
                    help='report the held-out residue\'s nearest neighbours in descriptor '
                         'space and whether they survive in training')
    ap.add_argument('--ckpt', default=None,
                    help='checkpoint to score the held-out rows with. Without it the '
                         'harness reports the split and the geometry only.')
    ap.add_argument('--reference_probe', action='store_true',
                    help='run the no-GPU ridge probe that actually PREDICTS the held-out '
                         'residue under both encodings. This is the runnable demonstration '
                         'of the claim; it is not the DeepEF model and its absolute numbers '
                         'must never be quoted as DeepEF performance.')
    ap.add_argument('--predictor', default=None,
                    help='dotted path to a callable(rows)->list[float] used instead of a '
                         'checkpoint. The harness is model-agnostic on purpose: the split '
                         'and the scoring are the contribution, not any one network.')
    A = ap.parse_args()

    hold_out = [h.strip() for h in A.hold_out.split(',') if h.strip()]
    print('W10 held-out-residue harness')
    print('  hold out      : %s' % ','.join(hold_out))
    print('  descriptor CSV: %s' % A.csv)

    if A.descriptor_neighbours:
        print('\nDescriptor-space neighbours of the held-out residue(s):')
        for h, nn, err in descriptor_neighbours(hold_out, A.csv):
            if err:
                print('  %-4s %s' % (h, err))
                continue
            print('  %-4s nearest in descriptor space:' % h)
            for lab, d, in_train in nn:
                print('        %-4s d=%7.3f  %s' % (lab, d,
                      'IN TRAINING' if in_train else 'ALSO HELD OUT'))
            n_in = sum(1 for _, _, t in nn if t)
            print('        -> %d of %d nearest neighbours survive in training. %s'
                  % (n_in, len(nn),
                     'Interpolation is possible.' if n_in else
                     'NOTHING to interpolate from -- this is extrapolation, expect failure.'))

    rows, files = load_corpus(A.mut_dir, A.limit_files or None)
    train, held, wt_side = split_by_held_out(rows, hold_out)
    print('\nSplit over %d protein CSVs, %d rows read:' % (len(files), len(rows)))
    print('  TRAIN rows (mutant not held out, WT not held out) : %d' % len(train))
    print('  HELD-OUT rows (mutation INTO a held-out residue)  : %d' % len(held))
    print('  DROPPED rows (WT is a held-out residue)           : %d' % len(wt_side))
    print('    -- dropped, not trained on: leaving X->W rows in training would show the')
    print('       model tryptophan as the residue being mutated away from, and the')
    print('       "never seen" premise would be false.')
    held_prots = sorted(set(r['_protein'] for r in held))
    print('  held-out rows span %d proteins' % len(held_prots))

    if A.report_split and not (A.ckpt or A.predictor):
        print('\n--report_split: split only, no scoring requested.')
        return 0

    if not held:
        print('\nNo held-out rows to score.')
        return 0

    pred_fn = None
    tag = None
    if A.predictor:
        mod, _, fn = A.predictor.rpartition('.')
        import importlib
        pred_fn = getattr(importlib.import_module(mod), fn)
        tag = A.predictor
    elif A.ckpt:
        pred_fn = _checkpoint_predictor(A.ckpt)
        tag = A.ckpt
    elif A.reference_probe:
        # The runnable, no-GPU demonstration. See _reference_probe.
        return _run_reference_probe(train, held, hold_out, A.csv)

    if pred_fn is None:
        print('\nNo --ckpt, --predictor or --reference_probe: nothing to score. The split')
        print('and the descriptor geometry above are the parts that need no model.')
        return 0

    preds = list(pred_fn(held))
    if len(preds) != len(held):
        sys.stderr.write('predictor returned %d values for %d rows\n' % (len(preds), len(held)))
        return 1
    truth = [r.get('ddG_ML') for r in held]
    m = metrics(truth, preds)
    print('\nHELD-OUT RESIDUE(S) %s, predictor=%s' % (','.join(hold_out), tag))
    print('scored on ddG -- legitimate here: the held-out residue is exactly what differs')
    print('between WT and mutant, so it does NOT cancel under the metric rule.')
    print('  ddG  n=%(n)d  PCC=%(pcc).4f  MAE=%(mae).4f  RMSE=%(rmse).4f  a_p=%(a_p).4f  b_p=%(b_p).4f' % m)
    if 'deltaG' in held[0]:
        mg = metrics([r.get('deltaG') for r in held], preds)
        print('  dG   n=%(n)d  PCC=%(pcc).4f  MAE=%(mae).4f  b_p=%(b_p).4f' % mg)
    return 0


# --------------------------------------------------------------------------------------
# The runnable proxy: a descriptor-space probe that needs no GPU and no checkpoint.
# --------------------------------------------------------------------------------------
def _reference_probe(train, held, hold_out, csv_path):
    """Fit on TRAIN only, predict the HELD-OUT residue, two encodings, same fit.

    This is not the DeepEF model and does not pretend to be. It is the SMALLEST estimator
    that can distinguish the two hypotheses, which is exactly what makes it evidence: a
    ridge regression on the mutation's feature vector, fitted on train rows only, where
    the ONLY thing that changes between the two arms is how a residue is encoded.

      arm 'onehot' : features = [one_hot(wt) | one_hot(mut)], width 40.
                     The held-out residue's mutant column is ALL ZERO in every training
                     row, so its coefficient is never updated from its initial value.
                     Whatever the model predicts for W is therefore reached without any
                     information about W. It CANNOT do better than the intercept plus the
                     WT-side term. This is the closed alphabet, in one sentence.

      arm 'desc'   : features = [desc(wt) | desc(mut) | desc(mut)-desc(wt)], width 3K.
                     W is a point among F, Y, H, L on aromaticity / volume / pi-electrons,
                     all of which ARE in training, so the fitted coefficients apply to it.
                     This is the open alphabet.

    The difference between the two arms on held-out rows IS Ofir's claim, measured. If the
    descriptor arm is no better, the claim fails on this corpus and we say so.

    NOTE the honest limitation, stated because it bounds what this number means: the probe
    sees residue identity only. It has no structure, no context, no ProtT5. Its absolute
    numbers are far below the real model's and must never be quoted as DeepEF performance.
    What is meaningful is the CONTRAST between its two arms, because everything else is
    held fixed.
    """
    import numpy as np
    from aa_descriptors import load_descriptor_table, descriptor_labels

    labels = descriptor_labels(csv_path)
    table = load_descriptor_table(csv_path, canonical_only=False).numpy().astype(float)
    lab_i = dict((a, i) for i, a in enumerate(labels))
    K = table.shape[1]
    M = len(labels)

    def feats(rows, arm):
        X, keep = [], []
        for j, r in enumerate(rows):
            wt, mut = r['_wt'], r['_mut']
            if wt not in lab_i or mut not in lab_i:
                continue
            if arm == 'onehot':
                v = np.zeros(2 * M)
                v[lab_i[wt]] = 1.0
                v[M + lab_i[mut]] = 1.0
            else:
                dw, dm = table[lab_i[wt]], table[lab_i[mut]]
                v = np.concatenate([dw, dm, dm - dw])
            X.append(v)
            keep.append(j)
        return np.asarray(X), keep

    def target(rows, keep, col):
        # MegaScale writes '-' (and occasionally other non-numeric sentinels) where a
        # ddG could not be fitted. float() on those raises, so coerce to NaN and let the
        # finite-mask below drop them. Silently mapping them to 0.0 would be far worse:
        # a fake 0.0 ddG is a plausible-looking neutral mutation and would bias every
        # slope toward compression -- exactly the a_p artefact this project measures.
        out = np.empty(len(keep), dtype=float)
        for i, j in enumerate(keep):
            try:
                out[i] = float(rows[j].get(col, np.nan))
            except (TypeError, ValueError):
                out[i] = np.nan
        return out

    out = {}
    for arm in ('onehot', 'desc'):
        Xtr, ktr = feats(train, arm)
        ytr = target(train, ktr, 'ddG_ML')
        good = np.isfinite(ytr) & np.isfinite(Xtr).all(1)
        Xtr, ytr = Xtr[good], ytr[good]
        Xhe, khe = feats(held, arm)
        yhe = target(held, khe, 'ddG_ML')
        ghe = np.isfinite(yhe) & np.isfinite(Xhe).all(1)
        Xhe, yhe = Xhe[ghe], yhe[ghe]
        if Xtr.shape[0] < 10 or Xhe.shape[0] < 2:
            out[arm] = dict(n=0, pcc=float('nan'), mae=float('nan'), rmse=float('nan'),
                            a_p=float('nan'), b_p=float('nan'), width=Xtr.shape[1] if Xtr.size else 0)
            continue
        # ridge with an explicit intercept column, lam small and FIXED across arms so the
        # arms differ only in their encoding.
        mu, sg = Xtr.mean(0), Xtr.std(0)
        sg[sg == 0] = 1.0
        Ztr = np.hstack([(Xtr - mu) / sg, np.ones((Xtr.shape[0], 1))])
        Zhe = np.hstack([(Xhe - mu) / sg, np.ones((Xhe.shape[0], 1))])
        lam = 1.0
        A_ = Ztr.T.dot(Ztr) + lam * np.eye(Ztr.shape[1])
        A_[-1, -1] -= lam                      # do not penalise the intercept
        w = np.linalg.solve(A_, Ztr.T.dot(ytr))
        p = Zhe.dot(w)
        m = metrics(yhe, p)
        m['width'] = int(Xtr.shape[1])
        m['n_train'] = int(Xtr.shape[0])
        out[arm] = m
    return out


def _run_reference_probe(train, held, hold_out, csv_path):
    res = _reference_probe(train, held, hold_out, csv_path)
    print('\n' + '=' * 78)
    print('RUNNABLE PROXY for Ofir\'s claim -- ridge probe, fitted on TRAIN only,')
    print('predicting mutations INTO the held-out residue %s. No GPU, no checkpoint.'
          % ','.join(hold_out))
    print('The two arms differ ONLY in how a residue is encoded.')
    print('=' * 78)
    for arm, name in (('onehot', 'one-hot (CLOSED alphabet)'),
                      ('desc', 'descriptors (OPEN alphabet)')):
        m = res[arm]
        print('  %-28s width=%-4s n_train=%-8s' % (name, m.get('width', '?'), m.get('n_train', '?')))
        print('      held-out n=%(n)d  PCC=%(pcc).4f  MAE=%(mae).4f  RMSE=%(rmse).4f  '
              'a_p=%(a_p).4f  b_p=%(b_p).4f' % m)
    import math
    po, pd_ = res['onehot']['pcc'], res['desc']['pcc']
    mo, md = res['onehot']['mae'], res['desc']['mae']
    bo, bd = res['onehot']['b_p'], res['desc']['b_p']
    print('  ' + '-' * 74)
    if all(not math.isnan(v) for v in (po, pd_)):
        print('  PREDICTIONS EXIST for the held-out residue in BOTH arms (that is')
        print('  requirement (d) of the gate). The CONTRAST is the claim:')
        print('      dPCC  = %+.4f    dMAE = %+.4f    d|b_p| = %+.4f'
              % (pd_ - po, md - mo, abs(bd) - abs(bo)))

        # ------------------------------------------------------------------------
        # THE METRIC RULE, AGAIN, IN A NEW PLACE. Read this before quoting dPCC.
        # ------------------------------------------------------------------------
        if len(hold_out) == 1:
            print('')
            print('  READ THE PCC CAREFULLY -- with ONE held-out residue it is structurally')
            print('  blind to this lever, and dPCC ~ 0 is the EXPECTED result, not a null one:')
            print('    every held-out row has the SAME mutant residue, so the mutant-side')
            print('    feature block is CONSTANT across all of them. A constant block shifts')
            print('    every prediction by the same amount; it cannot RE-RANK them. So PCC is')
            print('    determined by the WT side alone and comes out identical in both arms')
            print('    by construction. What the mutant encoding CAN move is the OFFSET.')
            print('    That is b_p -- and b_p is where the effect shows up:')
            print('      b_p  one-hot %+.4f  ->  descriptors %+.4f   (|b_p| %s by %.4f)'
                  % (bo, bd, 'DOWN' if abs(bd) < abs(bo) else 'UP', abs(abs(bd) - abs(bo))))
            print('    This is the project\'s metric rule recurring: a lever that acts on the')
            print('    reference state MUST be scored on b_p, never on a correlation. The')
            print('    Flory coil was nearly discarded for exactly this mistake.')
            print('    To make PCC able to see the lever, hold out SEVERAL residues at once:')
            print('      --hold_out W,F,Y   (then the mutant side varies and PCC discriminates)')

        better_bp = abs(bd) < abs(bo)
        better_mae = md < mo
        better_pcc = pd_ > po + 1e-6
        print('')
        if len(hold_out) == 1:
            if better_bp or better_mae:
                print('  -> On the metric that CAN see this lever (b_p, and MAE which contains it)')
                print('     the descriptor encoding places an UNSEEN residue better than one-hot.')
                print('     That is the continuity claim, on a canonical proxy: a GENERALISATION')
                print('     result about the representation, NOT a MegaScale benchmark number.')
            else:
                print('  -> Even on b_p the descriptor encoding did not help. Reported as')
                print('     measured; the claim is not supported by this probe on this split.')
        else:
            if better_pcc or better_bp:
                print('  -> The descriptor encoding places UNSEEN residues better than one-hot.')
                print('     GENERALISATION result, not a MegaScale benchmark number.')
            else:
                print('  -> The descriptor encoding did NOT beat one-hot here. Reported as')
                print('     measured; the claim is not supported by this probe on this split.')
    return 0


def _checkpoint_predictor(ckpt):
    """Return a callable(rows)->preds backed by a trained PEM checkpoint.

    Deliberately late-bound and defensive: this harness's value is the SPLIT and the
    SCORING, both of which run with no model at all. Wiring a checkpoint is the part that
    needs a GPU-trained artifact, and it must not be able to break the parts that do not.
    """
    def _predict(rows):
        import torch
        from model.hydro_net import PEM
        from model.model_cfg import CFG
        if not os.path.isfile(ckpt):
            raise IOError('checkpoint not found: %s. The split and descriptor-geometry '
                          'reports above run without one.' % ckpt)
        sd = torch.load(ckpt, map_location='cpu')
        model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                    dropout_rate=CFG.dropout_rate, light_attention=True, readout=False)
        model.load_state_dict(sd.get('model_state_dict', sd), strict=False)
        model.eval()
        raise NotImplementedError(
            'W10: checkpoint scoring needs the per-protein tensors (coords/one_hot/'
            'prott5) that AllProteinValidationDataset loads, and a GPU to be practical. '
            'Pass --predictor <dotted.path> to plug in the project\'s existing evaluation '
            'loop instead. The harness contract is: callable(rows) -> list[float], one '
            'prediction per held-out row, in order.')
    return _predict


if __name__ == '__main__':
    sys.exit(main())
