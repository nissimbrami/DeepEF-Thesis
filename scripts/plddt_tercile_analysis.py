"""W8 -- split an existing eval CSV's metrics by structure-confidence tercile.

WHAT QUESTION THIS ANSWERS
--------------------------
Not "does pLDDT improve the benchmark". It answers: WHERE does the model fail? If
per-protein correlation in the bottom-confidence third is materially worse than in the
top third, then some of the residual error is structure-quality error rather than model
error, and the W8 feature has something to bite on. If the three terciles are flat, W8
is honestly not worth training and this script says so with numbers.

This runs on eval CSVs that ALREADY EXIST. It does not need a trained W8 model, and it
does not need the annotation CSV -- confidence can be sourced straight from the
AlphaFold PDB B-factor column.

WHY IT REFUSES RATHER THAN GUESSES
----------------------------------
If no per-structure quality can be sourced, this script EXITS with a message naming
what it looked for. It never falls back to an arbitrary or random split. Inventing
terciles would produce a plot that looks exactly like a real finding and means nothing,
which is the specific failure mode the item calls out.

USAGE
-----
    python scripts/plddt_tercile_analysis.py --csv eval_results/<file>.csv
    python scripts/plddt_tercile_analysis.py --csv <file> --quality_csv plddt.csv
    python scripts/plddt_tercile_analysis.py --csv <file> --pdb_dir <dir>
    python scripts/plddt_tercile_analysis.py --csv a.csv --csv b.csv --out summary.csv
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import struct_quality as sq   # noqa: E402


# ---------------------------------------------------------------------------
# Statistics -- plain Python, no scipy dependency
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float('nan')
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 2:
        return float('nan')
    return pearson(rank(xs), rank(ys))


def rmse(xs, ys):
    if not xs:
        return float('nan')
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(xs, ys)) / len(xs))


def mae(xs, ys):
    if not xs:
        return float('nan')
    return sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs)


# ---------------------------------------------------------------------------
# Sourcing the per-structure quality scalar
# ---------------------------------------------------------------------------

def load_quality(quality_csv=None, pdb_dir=None, tensors=None):
    """{protein: mean confidence in [0,1]}, or {} if nothing could be sourced.

    Three sources, tried in order of explicitness:
      1. --quality_csv, a W8 annotation CSV (per-residue, averaged here)
      2. --quality_csv with a per-STRUCTURE 'quality'/'plddt'/'mean_plddt' column
      3. --pdb_dir, mean CA B-factor per PDB
    """
    if quality_csv:
        if not os.path.exists(quality_csv):
            raise FileNotFoundError(
                "W8 tercile: --quality_csv %s does not exist." % quality_csv)
        with open(quality_csv, newline='', encoding='utf-8-sig') as fh:
            cols = (csv.DictReader(fh).fieldnames) or []
        # Per-structure form: one row per protein, no resi column.
        if 'resi' not in cols:
            key = None
            for c in ('quality', 'mean_plddt', 'plddt', 'confidence'):
                if c in cols:
                    key = c
                    break
            if key is None:
                raise ValueError(
                    "W8 tercile: %s has no 'resi' column (so it is not a per-residue "
                    "annotation) and none of quality/mean_plddt/plddt/confidence "
                    "(so it is not a per-structure table). Columns: %s"
                    % (quality_csv, cols))
            if 'protein' not in cols:
                raise ValueError(
                    "W8 tercile: %s has no 'protein' column to join on. Columns: %s"
                    % (quality_csv, cols))
            out = {}
            with open(quality_csv, newline='', encoding='utf-8-sig') as fh:
                for row in csv.DictReader(fh):
                    p = (row['protein'] or '').strip()
                    if not p:
                        continue
                    v = float(row[key])
                    out[p] = sq._clip01(v / sq.PLDDT_SCALE if v > 1.0 else v)
            return out
        ann = sq.PlddtAnnotations.from_csv(quality_csv)
        return {p: ann.mean_confidence(p) for p in ann.proteins()}

    d = pdb_dir or sq.DEFAULT_PDB_DIR
    return sq.per_structure_quality(d, tensors)


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def read_eval_csv(path):
    """[(protein, deltaG, pred_deltaG, ddG, pred_ddG)] from an eval CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError("W8 tercile: eval CSV not found: %s" % path)
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        if 'protein' not in cols:
            raise ValueError(
                "W8 tercile: %s has no 'protein' column, so it cannot be joined to a "
                "per-structure confidence. Columns: %s" % (path, cols))
        has_dg = 'deltaG' in cols and 'pred_deltaG' in cols
        has_ddg = 'ddG' in cols and 'pred_ddG' in cols
        if not (has_dg or has_ddg):
            raise ValueError(
                "W8 tercile: %s has neither (deltaG, pred_deltaG) nor (ddG, pred_ddG). "
                "Columns: %s" % (path, cols))
        for row in reader:
            p = (row['protein'] or '').strip()
            if not p:
                continue

            def g(k):
                try:
                    return float(row[k])
                except (KeyError, TypeError, ValueError):
                    return None
            rows.append((p, g('deltaG'), g('pred_deltaG'), g('ddG'), g('pred_ddG')))
    return rows


def per_protein_metrics(rows):
    """{protein: {...}} -- per-protein PCC/Spearman/RMSE on ddG, and on dG."""
    by = {}
    for p, dg, pdg, ddg, pddg in rows:
        by.setdefault(p, []).append((dg, pdg, ddg, pddg))
    out = {}
    for p, rs in by.items():
        d = {'n': len(rs)}
        dgs = [(a, b) for a, b, _, _ in rs if a is not None and b is not None]
        ddgs = [(c, e) for _, _, c, e in rs if c is not None and e is not None]
        if len(dgs) >= 2:
            xs = [a for a, _ in dgs]
            ys = [b for _, b in dgs]
            d['dg_pcc'] = pearson(xs, ys)
            d['dg_rmse'] = rmse(xs, ys)
            d['dg_mae'] = mae(xs, ys)
        if len(ddgs) >= 2:
            xs = [a for a, _ in ddgs]
            ys = [b for _, b in ddgs]
            d['ddg_pcc'] = pearson(xs, ys)
            d['ddg_spearman'] = spearman(xs, ys)
            d['ddg_rmse'] = rmse(xs, ys)
            d['ddg_mae'] = mae(xs, ys)
        out[p] = d
    return out


def terciles(quality_by_prot, proteins):
    """Split proteins into three equal-count confidence bands.

    Split by RANK, not by value. Equal-width value bins would be near-empty at the
    bottom, because pLDDT is heavily left-skewed on this dataset; equal-count bands
    keep the comparison statistically meaningful.
    """
    have = [(quality_by_prot[p], p) for p in proteins if p in quality_by_prot]
    have.sort()
    n = len(have)
    if n < 3:
        return None
    a = n // 3
    b = (2 * n) // 3
    return {
        'low': [p for _, p in have[:a]],
        'mid': [p for _, p in have[a:b]],
        'high': [p for _, p in have[b:]],
    }, {p: q for q, p in have}


def _mean(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return float('nan')
    return sum(vals) / len(vals)


def analyse(csv_path, quality, metrics_keys=None):
    rows = read_eval_csv(csv_path)
    pm = per_protein_metrics(rows)
    proteins = sorted(pm)
    matched = [p for p in proteins if p in quality]

    if not matched:
        raise SystemExit(
            "W8 tercile: NONE of the %d proteins in %s could be matched to a "
            "confidence value.\n"
            "The join key is the 'protein' column; confidence keys look like: %s\n"
            "Refusing to invent terciles."
            % (len(proteins), csv_path, sorted(quality)[:5]))

    split = terciles(quality, proteins)
    if split is None:
        raise SystemExit(
            "W8 tercile: only %d proteins in %s have a confidence value; at least 3 "
            "are needed to form terciles. Refusing to invent them."
            % (len(matched), csv_path))
    bands, qmap = split

    keys = metrics_keys or ('ddg_pcc', 'ddg_spearman', 'ddg_rmse', 'dg_pcc', 'dg_rmse')
    out = {'csv': csv_path, 'n_proteins': len(proteins), 'n_matched': len(matched),
           'unmatched': [p for p in proteins if p not in quality], 'bands': {}}
    for name in ('low', 'mid', 'high'):
        ps = bands[name]
        band = {'n_proteins': len(ps),
                'n_rows': sum(pm[p]['n'] for p in ps),
                'conf_mean': _mean([qmap[p] for p in ps]),
                'conf_min': min([qmap[p] for p in ps]) if ps else float('nan'),
                'conf_max': max([qmap[p] for p in ps]) if ps else float('nan')}
        for k in keys:
            band[k] = _mean([pm[p].get(k) for p in ps])
        out['bands'][name] = band
    out['keys'] = keys
    return out


def report(res):
    print('=' * 78)
    print('W8 confidence-tercile split: %s' % res['csv'])
    print('  proteins in CSV: %d   matched to confidence: %d'
          % (res['n_proteins'], res['n_matched']))
    if res['n_matched'] < res['n_proteins']:
        n = res['n_proteins'] - res['n_matched']
        print('  NOTE: %d protein(s) had NO confidence value and are EXCLUDED.' % n)
        print('        Most are designed mini-proteins whose PDB B-factor column is a')
        print('        placeholder (all zeros), i.e. never scored by AlphaFold. They')
        print('        are dropped rather than treated as confidence 0, which would')
        print('        fabricate a bottom tercile. Excluded: %s'
              % (', '.join(res['unmatched'][:6])
                 + (' ...' if len(res['unmatched']) > 6 else '')))
    print('-' * 78)
    hdr = '%-6s %6s %8s %9s %9s' % ('band', 'nprot', 'nrows', 'conf_lo', 'conf_hi')
    for k in res['keys']:
        hdr += ' %11s' % k
    print(hdr)
    for name in ('low', 'mid', 'high'):
        b = res['bands'][name]
        line = '%-6s %6d %8d %9.4f %9.4f' % (
            name, b['n_proteins'], b['n_rows'], b['conf_min'], b['conf_max'])
        for k in res['keys']:
            line += ' %11.4f' % b[k]
        print(line)
    print('-' * 78)
    lo = res['bands']['low']
    hi = res['bands']['high']
    for k in res['keys']:
        d = hi[k] - lo[k]
        if not math.isnan(d):
            print('  delta high-low  %-14s %+0.4f' % (k, d))
    print('=' * 78)


def main():
    ap = argparse.ArgumentParser(
        description='Split eval-CSV metrics by structure-confidence tercile.')
    ap.add_argument('--csv', action='append', required=True,
                    help='Eval CSV. Repeatable.')
    ap.add_argument('--quality_csv',
                    help='W8 per-residue annotation CSV, or a per-structure table '
                         'with protein + quality/mean_plddt/plddt/confidence.')
    ap.add_argument('--pdb_dir', default=None,
                    help='Directory of PDBs whose CA B-factor is pLDDT. '
                         'Default: %s' % sq.DEFAULT_PDB_DIR)
    ap.add_argument('--tensors', default=sq.DEFAULT_TENSOR_DIR)
    ap.add_argument('--out', help='Write a tidy summary CSV of every band.')
    args = ap.parse_args()

    quality = load_quality(args.quality_csv, args.pdb_dir, args.tensors)
    if not quality:
        raise SystemExit(
            "W8 tercile: no per-structure confidence could be sourced.\n"
            "Tried:\n"
            "  --quality_csv  : %s\n"
            "  --pdb_dir      : %s (CA B-factor column)\n"
            "Provide one of them. Format for the annotation CSV: PLDDT_SPEC.md.\n"
            "Refusing to invent terciles -- a random split would look exactly like a "
            "real finding."
            % (args.quality_csv or '<not given>',
               args.pdb_dir or sq.DEFAULT_PDB_DIR))

    print('sourced confidence for %d structures (mean %.4f)'
          % (len(quality), sum(quality.values()) / len(quality)))

    results = []
    for c in args.csv:
        res = analyse(c, quality)
        report(res)
        results.append(res)

    if args.out:
        with open(args.out, 'w', newline='') as fh:
            w = csv.writer(fh)
            keys = results[0]['keys']
            w.writerow(['csv', 'band', 'n_proteins', 'n_rows', 'conf_mean',
                        'conf_min', 'conf_max'] + list(keys))
            for res in results:
                for name in ('low', 'mid', 'high'):
                    b = res['bands'][name]
                    w.writerow([res['csv'], name, b['n_proteins'], b['n_rows'],
                                '%.6f' % b['conf_mean'], '%.6f' % b['conf_min'],
                                '%.6f' % b['conf_max']]
                               + ['%.6f' % b[k] for k in keys])
        print('wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
