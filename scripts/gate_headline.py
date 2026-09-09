"""P8 GATE: ONE canonical basis for the DeepEF headline.

THE CANONICAL BASIS
-------------------
    27 test proteins (2K5H excluded -- its reference row is a mutant background, P7)
    ddG metric, pooled over all mutations of those 27 proteins
    the 9 ORIGINAL-POPULATION eval CSVs
    the val-selected epoch only (exactly one CSV per run)

    pooled 0.5772 +/- 0.0316   oracle 0.7156 +/- 0.0474   gain +0.1384   (n=9 runs)

WHAT "ORIGINAL POPULATION" MEANS, precisely: the runs whose SLURM job name begins DeepEF_ --
the control, the five seed replicates, and the three anchor arms. They share ONE training set
and ONE recipe, differing only in seed or in the single lever named in the tag. That shared
training set is what makes averaging over them mean anything.

WHAT IS EXCLUDED, and why each exclusion is forced:
  * p3_* factorial cells      -- a crossed design, not a population; 8 of its 17 cells are D1.
  * gld_* golden-lane arms    -- a different lane, a different lever per arm.
  * abl_loroW_onehot_s42_e12  -- its filename lacks the gld_ prefix, so a glob over
                                 eval_results/ picks it up as if it were original-population.
                                 It is not: its SLURM job is gld_loroW_onehot and line 1 of
                                 its log reads "[LORO] holding out of TRAINING (both
                                 directions): W". A DIFFERENT TRAINING SET. This is the one
                                 file a naive glob gets wrong, so membership below is listed
                                 BY HAND rather than matched by pattern.
  * every non-val-selected epoch of the two trajectory runs (P0).

WHY THE RECORD DISAGREED WITH ITSELF
------------------------------------
0.5772 was ALREADY the canonical number. The defect was never that it was wrong -- it was that
the record never wrote down WHICH NINE CSVs produced it, so two figures computed on other
populations stood beside it as apparent equals:
    0.4899  = mean over 22 mixed CSVs, INCLUDING D1 arms scoring as low as -0.074
    0.5646  = mean over 20 non-D1 checkpoints spanning three different lanes
Neither is wrong arithmetic; both are the wrong population. A correlation averaged over a mixed
population is not a measurement of anything.

THE RULE THIS GATE ENCODES
--------------------------
NEVER average across runs that differ in factor D (--unfolded_emb zero). D0 and D1 are different
models -- D1 pooled ddG PCC runs -0.074 to 0.317 against D0's 0.536 to 0.638, a separation of
14.4 sigma. Report D0 and D1 SEPARATELY, always.

Exit 0 = the canonical numbers recompute, and no document quotes a retired population's figure
         for a canonical quantity without marking it as retired.
Exit 1 = otherwise.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, 'eval_results')
DOCS = os.path.join(ROOT, 'results', '01_start_here')
EXCLUDE_PROTEIN = '2K5H'
N_PROT_EXPECT = 27

CANONICAL = [
    'abl_calib_ctrl_repro2_e14.csv',
    'abl_sigma_seed1_e13.csv',
    'abl_sigma_seed2_e10.csv',
    'abl_sigma_seed3_e13.csv',
    'abl_sigma_seed4_e14.csv',
    'abl_sigma_seed42_e9.csv',
    'abl_anchor_w0.3_s42_e14.csv',
    'abl_anchor_w1.0_s42_e14.csv',
    'abl_anchor_w3.0_s42_e13.csv',
]

# The values this gate defends, to 4 dp.
EXPECT = {'pooled': 0.5772, 'oracle': 0.7156, 'gain': 0.1384}
TOL = 5e-5

# Figures produced by a RETIRED population for a canonical quantity. A line may carry one only
# if it also marks it as retired -- that is how the correction paragraphs stay legal.
RETIRED = {
    '0.4899': 'pooled ddG PCC over 22 mixed CSVs including D1 arms',
    '0.6443': 'offset-removal oracle over those same 22 mixed CSVs',
    '0.1544': 'offset-removal gain over those same 22 mixed CSVs',
    '0.5646': 'pooled ddG PCC over 20 non-D1 checkpoints spanning three lanes',
    '0.6347': 'offset-removal oracle over those same 20 checkpoints',
    '0.0702': 'offset-removal gain over those same 20 checkpoints',
}
OKMARK = re.compile(
    r'retired|superseded|was quoted|previously|earlier|wrong population|mixed population|'
    r'not the canonical|do not quote|NOT canonical|mis-populated|non-canonical',
    re.I)

# A bare number can collide with an unrelated quantity -- 0.7156 is also the a_p of a p3 cell.
# So a retired value only counts when the line is TALKING ABOUT a canonical quantity.
QUANT = re.compile(r'pooled|oracle|gain|headline|corrected', re.I)


def pooled_and_oracle(path):
    d = pd.read_csv(path)
    d = d[d['protein'] != EXCLUDE_PROTEIN]
    t = d['ddG'].values.astype(float)
    p = d['pred_ddG'].values.astype(float)
    pooled = float(np.corrcoef(t, p)[0, 1])
    q = p.copy()
    for _, idx in d.groupby('protein').indices.items():
        q[idx] = p[idx] + np.mean(t[idx] - p[idx])       # remove this protein's own offset
    oracle = float(np.corrcoef(t, q)[0, 1])
    return pooled, oracle, d['protein'].nunique()


def main():
    fail = []
    rows = []
    for f in CANONICAL:
        p = os.path.join(EVAL, f)
        if not os.path.exists(p):
            print('FAIL: canonical CSV missing: %s' % f)
            sys.exit(1)
        po, orc, nprot = pooled_and_oracle(p)
        if nprot != N_PROT_EXPECT:
            fail.append('%s has %d proteins, expected %d' % (f, nprot, N_PROT_EXPECT))
        rows.append((f, po, orc))

    po = np.array([r[1] for r in rows])
    orc = np.array([r[2] for r in rows])
    got = {'pooled': po.mean(), 'oracle': orc.mean(), 'gain': orc.mean() - po.mean()}

    print('CANONICAL BASIS')
    print('  27 proteins (2K5H excluded) | ddG | 9 original-population CSVs | val-selected epoch')
    for f, a, b in rows:
        print('   %-34s pooled %.4f  oracle %.4f' % (f, a, b))
    print('  n=%d  pooled %.4f +/- %.4f   oracle %.4f +/- %.4f   gain %+.4f'
          % (len(rows), got['pooled'], po.std(ddof=1), got['oracle'], orc.std(ddof=1),
             got['gain']))
    print('  best single run: pooled %.4f (%s)' % (po.max(), rows[int(po.argmax())][0]))

    for k, v in EXPECT.items():
        if abs(got[k] - v) > TOL:
            fail.append('%s recomputed as %.4f, expected %.4f' % (k, got[k], v))

    # ---- document consistency ----
    bad = []
    for name in sorted(os.listdir(DOCS)):
        if not name.endswith('.md'):
            continue
        path = os.path.join(DOCS, name)
        for i, line in enumerate(open(path, encoding='utf-8', errors='replace'), 1):
            if not QUANT.search(line) or OKMARK.search(line):
                continue
            for v, why in RETIRED.items():
                if v in line:
                    bad.append((name, i, v, why, line.strip()[:96]))
    if bad:
        print('\nretired-population figures quoted for a canonical quantity, unmarked:')
        for d, i, v, why, ln in bad:
            print('  %s:%d  %s = %s\n      %s' % (d, i, v, why, ln))
        fail.append('%d unmarked retired figure(s) in %s' % (len(bad), DOCS))

    # the canonical figures must actually appear in the record
    txt = ''.join(open(os.path.join(DOCS, n), encoding='utf-8', errors='replace').read()
                  for n in os.listdir(DOCS) if n.endswith('.md'))
    for k, v in EXPECT.items():
        if ('%.4f' % v) not in txt:
            fail.append('canonical %s=%.4f appears in no document in %s' % (k, v, DOCS))

    if fail:
        print('\nFAIL')
        for f in fail:
            print('  - %s' % f)
        sys.exit(1)
    print('\nPASS: canonical numbers recompute; every document agrees.')
    sys.exit(0)


if __name__ == '__main__':
    main()
