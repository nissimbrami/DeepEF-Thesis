"""SEVERING EXPERIMENT -- can the reference state be told apart from a per-protein network bias?

WHAT IS ACTUALLY IN DOUBT
-------------------------
The project's headline is that 77-88% of across-protein dG variance sits in E_u, with
corr(E_u, wt_err) = 0.865, and it concludes from this that THE REFERENCE STATE IS WRONG.
That inference does not follow, for one reason that has never been checked:

    wt_err = (E_u - E_f) - dG_true        <- a DIFFERENCE
    var(E_u - E_f) = var(E_u) + var(E_f) - 2*cov(E_u, E_f)

cov(E_u, E_f) HAS NEVER BEEN REPORTED anywhere in this repo (verified: no cov_Eu_Ef /
corr(E_u,E_f) string exists in any .py/.md/.json). E_u and E_f are produced by the SAME
frozen network on the SAME coordinates and the SAME sequence. A shared per-protein bias
term c_p -- the network simply reading "this is protein p" and adding an offset to BOTH
energies -- would:
    * inflate var(E_u)                                  (matches the 77-88% split)
    * inflate corr(E_u, wt_err)                         (matches the 0.865)
    * contribute EXACTLY ZERO to var(E_u - E_f)         (cancels in the difference)
So the existing evidence cannot distinguish "the reference state is miscalibrated" from
"the network has a per-protein bias". Those two diagnoses point at completely different
fixes, and ~150 GPU-hours of reference-state work is about to be spent on the first one.

THE TEST
--------
Freeze one baseline checkpoint. 28 test proteins, WILD TYPE ONLY, one forward pass per
protein per condition, and log E_u and E_f SEPARATELY (the thing the original analysis
never did).

    C0  base                                                 in-distribution reference
    C1  --unfolded_emb zero                                  fold-blind reference state
    C2  --flory_unfolded --coil_b fixed                      analytic ideal-coil geometry
    C3  C1 + C2 + --coil_edges                               all three, coil-consistent edges

    f_resid = var(wt_err | C3) / var(wt_err | C0)

If the across-protein spread really is manufactured by the reference state, severing the
reference state should collapse it: f_resid small. If instead it is a shared per-protein
bias, it cancels in the difference and survives severing: f_resid near 1.

THE OBJECTION THIS SCRIPT IS BUILT AROUND (read before believing any number it prints)
--------------------------------------------------------------------------------------
C1/C2/C3 push a FROZEN checkpoint out of the distribution it was trained on. This
checkpoint (p3_a0_d0_s0_D0) was trained with all four levers OFF, so C1/C2/C3 are all OOD
for it by construction. A network fed a zeroed unfolded embedding does not hand you "the
reference state, severed" -- it hands you an OOD E_u whose variance can collapse for
reasons that have nothing to do with protein physics. Degenerate outputs have low variance.
So var(wt_err) shrinks, f_resid looks wonderful, and the number is measuring OOD collapse.

f_resid IS THEREFORE NOT INTERPRETABLE ON ITS OWN. This script refuses to let it be read
alone. Per condition it also reports mean and dynamic range (std, min, max, p95-p5) of E_u
and E_f, and the dG MAE, and it prints an explicit WARNING when the degeneracy signature
fires. If C3's MAE explodes or its energy range collapses, f_resid must NOT be used to
cancel anything -- the run has measured the checkpoint leaving its training distribution.

WHY E_f IS RECOMPUTED IN EVERY CONDITION
----------------------------------------
scripts/w0_dg.py hoists E_f out of the condition loop, on the assumption that the unfolded
levers cannot touch the folded pass. For C0/C1/C2 that assumption does hold. For C3 IT IS
FALSE: --coil_edges is applied inside PEM.get_edge_index, which also owns the edge CACHE.
Recomputing E_f per condition costs 28 extra forwards and removes the assumption entirely.
The script ASSERTS that E_f is in fact unchanged in C1/C2 (where theory says it must be),
so if that invariant ever breaks it is reported rather than silently absorbed.

WHY ONE BATCHED FORWARD AND NOT TWO
-----------------------------------
--coil_edges (U5) rebuilds the UNFOLDED rows' GAT topology. coil_topology.apply_coil_edges
locates those rows by the n_folded split of the batch: rows [0, n_folded) are folded, rows
[n_folded, B) are unfolded. Scoring the two halves in two separate single-row forward
passes means n_folded == B in each, apply_coil_edges returns the input UNCHANGED, and
--coil_edges silently does nothing. C3 would then be a mislabelled C1+C2, the experiment
would report a null it never actually tested, and nothing would crash.
This is the site's SIGNATURE FAILURE MODE: code runs, completes, reports a number, feature
never read. So the folded and unfolded graphs go through ONE batched call
    model(torch.cat([folded, unfolded], dim=0), n_folded=1)
which is exactly the convention in Megascale-fineTuning/train.py:718 (get_wt_deltaG).
--assert_levers (default ON) proves at runtime that C3's unfolded edge set really did
change; if it did not, the run ABORTS instead of reporting.

Dry run (no GPU, no checkpoint, no data -- validates the whole pipeline on random tensors):
    python scripts/severing.py --dry_run
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
import argparse, json, sys, math

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from model.model_cfg import CFG
from model.hydro_net import PEM
from train_utils import get_graph, get_unfolded_graph, load_checkpoint
import coil_topology as _CT


# ---------------------------------------------------------------------------
# Conditions. Each is the exact CFG state, written out in full so that no
# condition inherits a stray flag from the one before it (a real hazard: CFG is
# a module-level singleton and every lever is read off it at call time).
# ---------------------------------------------------------------------------
BASE_CFG = dict(flory_unfolded=False, coil_b='fitted', coil_channels='broadcast',
                unfolded_emb='full', coil_edges=False)

CONDITIONS = [
    ('C0', 'base',                       dict(BASE_CFG)),
    ('C1', 'unfolded_emb=zero',          dict(BASE_CFG, unfolded_emb='zero')),
    ('C2', 'flory + coil_b=fixed',       dict(BASE_CFG, flory_unfolded=True, coil_b='fixed')),
    ('C3', 'C1+C2+coil_edges',           dict(BASE_CFG, unfolded_emb='zero',
                                              flory_unfolded=True, coil_b='fixed',
                                              coil_edges=True)),
]


def apply_cfg(d):
    """Set every lever explicitly. Never leaves a flag at whatever the last condition left."""
    for k, v in d.items():
        setattr(CFG, k, v)


def reset_cfg():
    apply_cfg(BASE_CFG)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def summarise(E_u, E_f, true_dg):
    """Everything the critic demanded, per condition, in one place.

    var_* use ddof=1 (sample variance) because these are 28 draws, not a population.
    f_resid is a RATIO of these, so the ddof convention cancels; it is stated anyway so
    the numbers can be reproduced from the per-protein CSV.
    """
    E_u = np.asarray(E_u, float); E_f = np.asarray(E_f, float)
    true_dg = np.asarray(true_dg, float)
    pred = E_u - E_f
    wt_err = pred - true_dg

    def rng(x):
        return dict(mean=float(x.mean()), std=float(x.std(ddof=1)),
                    min=float(x.min()), max=float(x.max()),
                    p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                    range=float(x.max() - x.min()),
                    iqr90=float(np.percentile(x, 95) - np.percentile(x, 5)))

    cov = float(np.cov(E_u, E_f, ddof=1)[0, 1])
    var_u = float(np.var(E_u, ddof=1))
    var_f = float(np.var(E_f, ddof=1))
    var_d = float(np.var(pred, ddof=1))

    return dict(
        n=int(E_u.size),
        # --- the quantity the whole experiment exists to produce ---
        var_wt_err=float(np.var(wt_err, ddof=1)),
        mae=float(np.abs(wt_err).mean()),
        rmse=float(np.sqrt((wt_err ** 2).mean())),
        bias=float(wt_err.mean()),
        corr_pred_true=_pearson(pred, true_dg),
        # --- THE NUMBER THAT WAS NEVER REPORTED ---
        cov_Eu_Ef=cov,
        corr_Eu_Ef=_pearson(E_u, E_f),
        # --- the variance decomposition of the DIFFERENCE, written out ---
        var_Eu=var_u, var_Ef=var_f, var_dG=var_d,
        var_identity_residual=float(var_d - (var_u + var_f - 2.0 * cov)),
        shared_bias_share=float(2.0 * cov / (var_u + var_f)) if (var_u + var_f) > 0 else float('nan'),
        frac_var_in_Eu=float(var_u / (var_u + var_f)) if (var_u + var_f) > 0 else float('nan'),
        corr_Eu_wt_err=_pearson(E_u, wt_err),
        corr_Ef_wt_err=_pearson(E_f, wt_err),
        # --- the OOD / degeneracy guard channels ---
        E_u=rng(E_u), E_f=rng(E_f), dG_pred=rng(pred),
    )


# ---------------------------------------------------------------------------
# Scoring one protein under one condition
# ---------------------------------------------------------------------------
def score(model, coords, oh0, emb0, mask, cfg_d, want_edges=False):
    """ONE batched forward: rows = [folded, unfolded], n_folded=1.

    Mirrors Megascale-fineTuning/train.py:718. Returns (E_f, E_u) and, if asked, the
    unfolded GAT edge count actually used -- the proof that --coil_edges was read.
    """
    apply_cfg(cfg_d)

    folded = get_graph(coords, oh0, emb0, mask).unsqueeze(0)
    unfolded = get_unfolded_graph(coords, oh0, emb0, mask).unsqueeze(0)
    batch = torch.cat([folded, unfolded], dim=0)

    n_edges = None
    if want_edges:
        # Read the edge set the model will actually build for THIS batch, through the
        # model's own code path (including its cache), so this is not a re-derivation.
        model._edge_cache = None
        model._edge_cache_key = None
        _gcn, gat = model.get_edge_index(batch, n_folded=1)
        N = batch.shape[1]
        n_edges = int((gat[0] >= N).sum())   # edges whose source is in the unfolded row
        model._edge_cache = None
        model._edge_cache_key = None

    energy = model(batch, n_folded=1)
    e = energy.reshape(-1)
    E_f = float(e[0])
    E_u = float(e[1])
    return E_f, E_u, n_edges


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_proteins(tensor_root, tm_path, device):
    import pandas as pd
    tm = pd.read_csv(tm_path)['name'].apply(lambda x: str(x).split('.')[0]).unique().tolist()
    names = sorted([p for p in os.listdir(tensor_root) if p in tm])
    out = []
    for name in names:
        d = os.path.join(tensor_root, name)
        try:
            coords = torch.load(os.path.join(d, 'coords_tensor.pt'),
                                map_location=device, weights_only=False).squeeze()
            dg = torch.load(os.path.join(d, 'deltaG.pt'),
                            map_location='cpu', weights_only=False)
            mask = torch.load(os.path.join(d, 'mask_tensor.pt'),
                              map_location=device, weights_only=False).squeeze()
            oh = torch.load(os.path.join(d, 'one_hot_encodings.pt'),
                            map_location='cpu', weights_only=False)
            ed = os.path.join(d, 'prott5_embeddings')
            emb = (torch.load(os.path.join(ed, sorted(os.listdir(ed))[0]),
                              map_location='cpu', weights_only=False)
                   if os.path.isdir(ed) else
                   torch.load(ed + '.pt', map_location='cpu', weights_only=False))
        except Exception as ex:
            print('  skip %s: %s' % (name, ex)); continue
        # Row 0 is the WILD TYPE. Verified on disk: for 1GYZ, one_hot[0] and one_hot[1]
        # differ in 0 positions, and deltaG[0] is the WT dG. WT ONLY -- no mutants here.
        out.append(dict(
            name=name,
            coords=coords.to(device).float(),
            oh0=oh[0].squeeze().to(device).float(),
            emb0=emb[0].squeeze().to(device).float(),
            mask=mask.to(device).float(),
            true=float(dg.reshape(-1)[0]),
            length=float((mask > 0).float().sum()),
        ))
    return out


# ---------------------------------------------------------------------------
# Dry run: the whole pipeline on random tensors. No GPU, no checkpoint, no data.
# ---------------------------------------------------------------------------
def dry_run(args):
    print('=' * 78)
    print('DRY RUN -- random tensors, untrained weights, CPU. Validates PLUMBING ONLY.')
    print('Numbers here are meaningless by construction; what is being checked is that')
    print('every condition runs, every lever is actually READ, and every statistic is')
    print('computed without NaN. This is the guard against the signature failure mode.')
    print('=' * 78)
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device('cpu')

    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True, readout=False).to(dev)
    model.eval()

    n_prot = int(args.dry_n)
    prots = []
    for i in range(n_prot):
        N = int(np.random.randint(43, 73))       # the real test set is 43-72 aa
        prots.append(dict(
            name='FAKE%02d' % i,
            coords=torch.randn(N, 4, 3) * 7.0,
            oh0=torch.eye(21)[torch.randint(0, 20, (N,))],
            emb0=torch.randn(N, int(CFG.emb_input_dim)),
            mask=torch.ones(N),
            true=float(np.random.randn() * 2 + 4),
            length=float(N),
        ))

    ok = [True]
    def check(msg, cond, extra=''):
        print('  %-58s %s %s' % (msg, 'PASS' if cond else 'FAIL', extra))
        ok[0] = ok[0] and bool(cond)

    results, per_protein = run_all(model, prots, dev, assert_levers=True, verbose=True)

    print('\n[dry-run checks]')
    for cid, _lab, _d in CONDITIONS:
        s = results[cid]
        check('%s all statistics finite' % cid,
              all(math.isfinite(s[k]) for k in
                  ('var_wt_err', 'mae', 'cov_Eu_Ef', 'var_Eu', 'var_Ef', 'var_dG')))
        check('%s var(dG) identity holds to 1e-6' % cid,
              abs(s['var_identity_residual']) < 1e-6,
              'residual=%.2e' % s['var_identity_residual'])

    # The levers must CHANGE something. If a condition is bit-identical to C0 the flag
    # was not read, which is exactly the failure this project keeps hitting.
    for cid in ('C1', 'C2', 'C3'):
        d = max(abs(per_protein[i]['E_u_' + cid] - per_protein[i]['E_u_C0'])
                for i in range(len(per_protein)))
        check('%s changed E_u vs C0 (lever was READ)' % cid, d > 1e-9, 'max|dE_u|=%.3e' % d)

    # C1/C2 must NOT touch the folded pass; C3 legitimately may (coil_edges owns the cache).
    for cid in ('C1', 'C2'):
        d = max(abs(per_protein[i]['E_f_' + cid] - per_protein[i]['E_f_C0'])
                for i in range(len(per_protein)))
        check('%s left E_f untouched (unfolded-only lever)' % cid, d < 1e-6,
              'max|dE_f|=%.3e' % d)

    ne0 = per_protein[0].get('n_unfolded_edges_C0')
    ne3 = per_protein[0].get('n_unfolded_edges_C3')
    check('C3 rebuilt the unfolded GAT edge set (U5 active)',
          ne0 is not None and ne3 is not None and ne3 != ne0,
          'C0=%s -> C3=%s' % (ne0, ne3))
    N0 = int(prots[0]['mask'].numel())
    check('C3 unfolded edge count == 2(N-1) chain', ne3 == 2 * (N0 - 1),
          'got %s, expected %d' % (ne3, 2 * (N0 - 1)))

    f_resid = results['C3']['var_wt_err'] / results['C0']['var_wt_err']
    check('f_resid computable and finite', math.isfinite(f_resid), 'f_resid=%.4f' % f_resid)

    reset_cfg()
    print('\nDRY RUN: %s' % ('ALL PASS -- pipeline is sound, submit the GPU job'
                            if ok[0] else 'FAILURES -- do not submit'))
    return 0 if ok[0] else 1


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
def run_all(model, prots, dev, assert_levers=True, verbose=False):
    per_protein = [dict(protein=p['name'], true=p['true'], length=p['length']) for p in prots]
    results = {}

    with torch.no_grad():
        for cid, label, cfg_d in CONDITIONS:
            E_us, E_fs, trues = [], [], []
            for i, p in enumerate(prots):
                want = (i == 0)      # edge proof on the first protein only (cheap)
                E_f, E_u, n_edges = score(model, p['coords'], p['oh0'], p['emb0'],
                                          p['mask'], cfg_d, want_edges=want)
                per_protein[i]['E_f_' + cid] = E_f
                per_protein[i]['E_u_' + cid] = E_u
                per_protein[i]['dG_pred_' + cid] = E_u - E_f
                per_protein[i]['wt_err_' + cid] = (E_u - E_f) - p['true']
                if want:
                    per_protein[i]['n_unfolded_edges_' + cid] = n_edges
                E_us.append(E_u); E_fs.append(E_f); trues.append(p['true'])
            results[cid] = summarise(E_us, E_fs, trues)
            results[cid]['label'] = label
            if verbose:
                print('  %s %-24s  E_u std=%9.4f  E_f std=%9.4f  MAE=%8.4f'
                      % (cid, label, results[cid]['E_u']['std'],
                         results[cid]['E_f']['std'], results[cid]['mae']))

    reset_cfg()

    if assert_levers:
        # HARD ABORT, not a warning. A silently-inert lever makes every number below a lie.
        n0 = per_protein[0].get('n_unfolded_edges_C0')
        n3 = per_protein[0].get('n_unfolded_edges_C3')
        if n0 is None or n3 is None or n3 == n0:
            raise RuntimeError(
                'ABORT: --coil_edges did not change the unfolded GAT edge set '
                '(C0=%s, C3=%s). C3 is not what it claims to be. Refusing to report.'
                % (n0, n3))
        for cid in ('C1', 'C2', 'C3'):
            d = max(abs(pp['E_u_' + cid] - pp['E_u_C0']) for pp in per_protein)
            if d <= 1e-9:
                raise RuntimeError(
                    'ABORT: condition %s produced E_u identical to C0 (max|dE_u|=%.3e). '
                    'The lever was never read. Refusing to report.' % (cid, d))

    return results, per_protein


# ---------------------------------------------------------------------------
# Interpretation + the OOD guard
# ---------------------------------------------------------------------------
def report(results, per_protein, args, out_json):
    C0, C3 = results['C0'], results['C3']
    f_resid = C3['var_wt_err'] / C0['var_wt_err'] if C0['var_wt_err'] > 0 else float('nan')

    W = 78
    print('\n' + '=' * W)
    print('SEVERING EXPERIMENT -- RESULTS')
    print('=' * W)

    print('\n--- per condition: the headline + the guard channels ---')
    print('%-4s %-22s %9s %9s %9s %9s %9s' %
          ('cond', 'label', 'var(err)', 'MAE', 'std(E_u)', 'std(E_f)', 'rng(E_u)'))
    for cid, label, _ in CONDITIONS:
        s = results[cid]
        print('%-4s %-22s %9.4f %9.4f %9.4f %9.4f %9.3f' %
              (cid, label, s['var_wt_err'], s['mae'],
               s['E_u']['std'], s['E_f']['std'], s['E_u']['range']))

    print('\n--- THE NUMBER THAT WAS NEVER REPORTED: cov(E_u, E_f) ---')
    print('%-4s %12s %12s %12s %12s %12s' %
          ('cond', 'cov(Eu,Ef)', 'corr(Eu,Ef)', 'var(Eu)', 'var(Ef)', 'var(dG)'))
    for cid, _l, _ in CONDITIONS:
        s = results[cid]
        print('%-4s %12.4f %12.4f %12.4f %12.4f %12.4f' %
              (cid, s['cov_Eu_Ef'], s['corr_Eu_Ef'], s['var_Eu'], s['var_Ef'], s['var_dG']))
    print('\n  var(dG) = var(Eu) + var(Ef) - 2cov(Eu,Ef).  A large POSITIVE cov means E_u and')
    print('  E_f move together across proteins: a SHARED per-protein component that inflates')
    print('  var(E_u) and corr(E_u, wt_err) while cancelling in the difference. That is the')
    print('  alternative explanation the 77-88% / 0.865 headline cannot rule out.')
    s0 = results['C0']
    print('\n  C0: 2cov/(var_Eu+var_Ef) = %.4f   <- share of the two variances that is shared'
          % s0['shared_bias_share'])
    print('  C0: var_Eu/(var_Eu+var_Ef) = %.4f  <- the "77-88%% sits in E_u" style number'
          % s0['frac_var_in_Eu'])
    print('  C0: corr(E_u, wt_err)      = %.4f  <- the "0.865" style number' % s0['corr_Eu_wt_err'])

    print('\n' + '-' * W)
    print('f_resid = var(wt_err | C3) / var(wt_err | C0) = %.4f' % f_resid)
    print('-' * W)

    # ---------------- OOD / degeneracy guard ----------------
    warnings = []
    mae_ratio = C3['mae'] / C0['mae'] if C0['mae'] > 0 else float('inf')
    rng_ratio = (C3['E_u']['range'] / C0['E_u']['range']
                 if C0['E_u']['range'] > 0 else float('inf'))
    std_ratio = (C3['E_u']['std'] / C0['E_u']['std']
                 if C0['E_u']['std'] > 0 else float('inf'))

    if mae_ratio > args.mae_blowup:
        warnings.append('C3 dG MAE is %.2fx the C0 MAE (%.4f -> %.4f), above the %.1fx '
                        'threshold. The frozen checkpoint is out of its training '
                        'distribution under C3.' %
                        (mae_ratio, C0['mae'], C3['mae'], args.mae_blowup))
    if rng_ratio < args.range_collapse:
        warnings.append('C3 dynamic range of E_u collapsed to %.2fx of C0 (%.3f -> %.3f), '
                        'below the %.2f threshold. Low variance from a DEGENERATE output, '
                        'not from physics.' %
                        (rng_ratio, C0['E_u']['range'], C3['E_u']['range'], args.range_collapse))
    if std_ratio < args.range_collapse:
        warnings.append('C3 std(E_u) collapsed to %.2fx of C0 (%.4f -> %.4f).'
                        % (std_ratio, C0['E_u']['std'], C3['E_u']['std']))
    if abs(C3['bias']) > args.bias_blowup * max(abs(C0['bias']), 1e-9) and abs(C3['bias']) > 1.0:
        warnings.append('C3 mean signed error moved to %.3f (C0 %.3f): the severed model is '
                        'no longer on the same energy scale.' % (C3['bias'], C0['bias']))
    # f_resid > 1 means severing made the across-protein spread WORSE. That is not a
    # "the reference state was not the cause" result -- a severed reference state should
    # at worst leave the spread alone. Growth means the severing itself injected new
    # across-protein variance, i.e. OOD damage, so the band interpretation below does
    # not apply and must not be read as evidence about the per-protein bias.
    if f_resid > args.fresid_growth:
        warnings.append('f_resid = %.3f > %.2f: severing INCREASED the across-protein error '
                        'variance. A severed reference state cannot legitimately add '
                        'variance, so this is the severing injecting its own OOD spread, '
                        'not evidence about the per-protein bias.'
                        % (f_resid, args.fresid_growth))

    print('\n' + '=' * W)
    if warnings:
        print('*** OOD WARNING -- f_resid IS NOT INTERPRETABLE ***')
        print('=' * W)
        for w in warnings:
            print('  ! ' + w)
        print('\n  A frozen checkpoint fed a severed reference state can shrink var(wt_err)')
        print('  simply by degenerating. When this warning fires, the low f_resid above is')
        print('  evidence of OOD collapse, NOT evidence that the reference state manufactures')
        print('  the offset. DO NOT use this f_resid to cancel or justify the reference-state')
        print('  work. The honest next step is a RETRAINED severed model (the levers on from')
        print('  step 0), not a frozen-checkpoint ablation.')
        verdict = 'UNINTERPRETABLE_OOD'
    else:
        print('OOD guard: PASSED. C3 kept its energy scale and dynamic range, so f_resid')
        print('reflects the severing rather than the checkpoint falling apart.')
        print('=' * W)
        if f_resid < 0.3:
            verdict = 'REFERENCE_STATE_CONFIRMED'
            print('\n  f_resid = %.4f  < 0.30' % f_resid)
            print('  MEANING: severing the reference state removed most of the across-protein')
            print('  error variance. The offset really is manufactured in the unfolded state,')
            print('  the shared per-protein bias is NOT the main story, and the ~150 GPU-hours')
            print('  of reference-state work is justified. PROCEED.')
        elif f_resid <= 0.7:
            verdict = 'MIXED'
            print('\n  0.30 <= f_resid = %.4f <= 0.70' % f_resid)
            print('  MEANING: BOTH mechanisms are live. The reference state carries a real part')
            print('  of the offset, but a substantial remainder survives total severing and')
            print('  must therefore live in the shared network bias (check corr(E_u,E_f) above,')
            print('  which should be materially > 0). Reference-state work will help and will')
            print('  NOT be sufficient. Fund it at reduced scope, and budget separately for the')
            print('  per-protein bias. Do not promise the full offset will close.')
        else:
            verdict = 'BIAS_NOT_REFERENCE_STATE'
            print('\n  f_resid = %.4f  > 0.70' % f_resid)
            print('  MEANING: the across-protein error survived even after the reference state')
            print('  was severed three ways. It cannot be coming from the reference state. The')
            print('  77-88% / 0.865 headline was reading a SHARED per-protein network bias that')
            print('  cancels in E_u - E_f. The ~150 GPU-hours of reference-state work would')
            print('  NOT have fixed the offset. STOP and redirect to the per-protein bias.')

    print('\n' + '=' * W)
    print('VERDICT: %s   f_resid=%.4f' % (verdict, f_resid))
    print('=' * W)

    payload = dict(
        checkpoint=args.ckpt, n_proteins=len(per_protein),
        f_resid=f_resid, verdict=verdict, ood_warnings=warnings,
        thresholds=dict(mae_blowup=args.mae_blowup, range_collapse=args.range_collapse,
                        bias_blowup=args.bias_blowup, fresid_growth=args.fresid_growth),
        conditions={cid: results[cid] for cid, _l, _d in CONDITIONS},
        per_protein=per_protein,
    )
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print('wrote %s' % out_json)

    csv_path = out_json.rsplit('.', 1)[0] + '_per_protein.csv'
    keys = ['protein', 'true', 'length']
    for cid, _l, _d in CONDITIONS:
        keys += ['E_f_' + cid, 'E_u_' + cid, 'dG_pred_' + cid, 'wt_err_' + cid]
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for pp in per_protein:
            f.write(','.join(str(pp.get(k, '')) for k in keys) + '\n')
    print('wrote %s' % csv_path)
    return verdict


def main():
    ap = argparse.ArgumentParser(description='Severing experiment: reference state vs per-protein bias')
    ap.add_argument('--ckpt', default=None, help='frozen checkpoint (required unless --dry_run)')
    ap.add_argument('--out', default='results/severing.json')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tensor_root', default='./data/Processed_K50_dG_datasets/training_data')
    ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
    ap.add_argument('--dry_run', action='store_true',
                    help='CPU pipeline validation on random tensors; no ckpt, no data, no GPU')
    ap.add_argument('--dry_n', type=int, default=8)
    ap.add_argument('--no_assert_levers', dest='assert_levers', action='store_false',
                    help='(discouraged) do not abort when a lever proves inert')
    ap.add_argument('--mae_blowup', type=float, default=2.0,
                    help='warn if C3 MAE exceeds this multiple of C0 MAE')
    ap.add_argument('--range_collapse', type=float, default=0.25,
                    help='warn if C3 E_u range/std falls below this fraction of C0')
    ap.add_argument('--bias_blowup', type=float, default=3.0)
    ap.add_argument('--fresid_growth', type=float, default=1.25,
                    help='warn if f_resid exceeds this (severing ADDED variance -> OOD damage)')
    args = ap.parse_args()

    if args.dry_run:
        sys.exit(dry_run(args))

    if not args.ckpt:
        print('FATAL: --ckpt is required (or pass --dry_run)'); sys.exit(2)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('FATAL: --device cuda but CUDA is not available on this node.'); sys.exit(2)

    dev = torch.device(args.device)
    torch.manual_seed(CFG.seed); np.random.seed(CFG.seed)

    print('loading proteins ...')
    prots = load_proteins(args.tensor_root, args.tm_path, dev)
    print('test proteins: %d' % len(prots))
    if len(prots) < 20:
        print('FATAL: only %d proteins loaded, expected 28' % len(prots)); sys.exit(1)

    model = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True, readout=False).to(dev)
    try:
        model, _, ep, _, _ = load_checkpoint(args.ckpt, model, device=dev)
        print('checkpoint epoch: %s' % ep)
    except Exception:
        model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False))
        print('loaded raw state_dict')
    model.eval()
    # FROZEN. No training, no grad, weights identical across all four conditions.
    for p in model.parameters():
        p.requires_grad_(False)

    print('\nscoring %d proteins x %d conditions = %d forward passes ...'
          % (len(prots), len(CONDITIONS), len(prots) * len(CONDITIONS)))
    results, per_protein = run_all(model, prots, dev,
                                   assert_levers=args.assert_levers, verbose=True)
    report(results, per_protein, args, args.out)
    sys.exit(0)


if __name__ == '__main__':
    main()
