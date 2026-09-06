"""measure_coil_offsets.py -- measure the 16 per-channel offsets for the U3 'offset' arm.

The 'offset' arm needs, for each of the 16 atom-pair channels c = atom_i*4 + atom_j
(atom order N=0, CA=1, C=2, CB=3):

    offset[c] = < d(atom_i(res i), atom_j(res j)) - d(CA(res i), CA(res j)) >

averaged over the FOLDED training structures at matched |i-j|. That is precisely the
quantity the coil is missing: the coil knows one CA-CA distance per residue pair, and
offset[c] is how far the other 15 atom pairs sit from it.

WHY THIS SCRIPT EXISTS RATHER THAN A HARDCODED TABLE. train_utils ships a
rigid-geometry FALLBACK table so the arm runs out of the box, but a fallback is an
assumption. This script replaces the assumption with a measurement, and prints both a
JSON file (for --coil_offsets_path) and a drop-in Python literal (to replace
_COIL_OFFSETS_FALLBACK_ANGSTROM). Run it once; it needs no GPU and no checkpoint.

USAGE
    cd <repo root>
    python scripts/measure_coil_offsets.py \
        --tensor_root ./data/Processed_K50_dG_datasets/training_data \
        --out data/coil_offsets.json \
        --max_proteins 200

    then either
        --coil_offsets_path data/coil_offsets.json
    or paste the printed literal over _COIL_OFFSETS_FALLBACK_ANGSTROM in train_utils.py.

UNITS -- THE TRAP. coords_tensor.pt on disk is in Angstrom, but train.normalize_batch
multiplies coords by NANO_TO_ANGSTROM = 0.1 BEFORE any graph is built, so the coil sees
coordinates ten times smaller. The JSON "offsets" key is written PRE-SCALED into model
units, ready for --coil_offsets_path; "offsets_angstrom" holds the unscaled values for
the source literal, which train_utils scales itself. Do not mix the two.

WHAT TO LOOK AT IN THE OUTPUT
  * sep_dependence: offset[c] computed separately at |i-j| = 1, 2, 4, 8, 16. The
    'offset' arm assumes ONE constant per channel, i.e. that the offset does not depend
    on separation. If these columns disagree by more than ~0.15 A the assumption is
    weak and the REPORT says so -- see "What would falsify this".
  * n_pairs and n_proteins: how much the average rests on.
  * The diagonal channels 0, 5, 10, 15 (N-N, CA-CA, C-C, CB-CB) should give offsets
    near 0 at large |i-j|, since two parallel legs cancel. Channel 5 must be EXACTLY 0
    by construction; the script asserts it.
"""
import argparse
import json
import os
import sys

import torch

ATOMS = ['N', 'CA', 'C', 'CB']
CA = 1


def channel_name(c):
    return '%s-%s' % (ATOMS[c // 4], ATOMS[c % 4])


def offsets_for_protein(x, mask, seps):
    """x: [N,4,3] folded coords. Returns dict sep -> [16] sum, and dict sep -> count.

    For a given separation s we take every valid pair (i, i+s) and accumulate
    d(atom_a(i), atom_b(i+s)) - d(CA(i), CA(i+s)) for all 16 (a,b).
    """
    n = x.shape[0]
    v = (mask > 0)
    sums, counts = {}, {}
    for s in seps:
        if n <= s:
            continue
        i = torch.arange(0, n - s)
        j = i + s
        keep = v[i] & v[j]
        if keep.sum() == 0:
            continue
        i, j = i[keep], j[keep]
        d_ca = torch.linalg.norm(x[i, CA, :] - x[j, CA, :], dim=-1)   # [P]
        acc = torch.zeros(16, dtype=torch.float64)
        for c in range(16):
            a, b = c // 4, c % 4
            d = torch.linalg.norm(x[i, a, :] - x[j, b, :], dim=-1)
            acc[c] = (d - d_ca).double().sum()
        sums[s] = acc
        counts[s] = int(i.numel())
    return sums, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tensor_root', required=True,
                    help='./data/Processed_K50_dG_datasets/training_data')
    ap.add_argument('--coords_name', default='coords_tensor.pt')
    ap.add_argument('--masks_name', default='mask_tensor.pt',
                    help='train.MASKS; if the file is absent an all-ones mask is used')
    ap.add_argument('--coord_scale', type=float, default=0.1,
                    help='train.NANO_TO_ANGSTROM. normalize_batch multiplies coords by '
                         'this BEFORE any graph is built, so the raw coords_tensor.pt on '
                         'disk is in Angstrom while the coil sees coords 10x smaller. '
                         'The JSON offsets key is written in MODEL units (scaled); the '
                         'printed table is shown in BOTH.')
    ap.add_argument('--out', default='data/coil_offsets.json')
    ap.add_argument('--max_proteins', type=int, default=200)
    ap.add_argument('--seps', default='1,2,4,8,16',
                    help='separations |i-j| to report separately')
    ap.add_argument('--pool_seps', default='4,8,16',
                    help='separations averaged together to produce the FINAL table. '
                         'Short separations are dominated by rigid intra-residue '
                         'geometry that the coil model does not claim to describe; the '
                         'coil is a long-range model, so the pooled table uses the '
                         'longer separations. Change with care and record the choice.')
    a = ap.parse_args()

    seps = [int(s) for s in a.seps.split(',')]
    pool = [int(s) for s in a.pool_seps.split(',')]
    for s in pool:
        if s not in seps:
            seps.append(s)
    seps = sorted(set(seps))

    dirs = sorted(os.listdir(a.tensor_root))[:a.max_proteins]
    tot_sum = {s: torch.zeros(16, dtype=torch.float64) for s in seps}
    tot_cnt = {s: 0 for s in seps}
    n_prot = 0
    skipped = []

    for d in dirs:
        cp = os.path.join(a.tensor_root, d, a.coords_name)
        if not os.path.isfile(cp):
            continue
        try:
            x = torch.load(cp, weights_only=True)
        except Exception as e:
            skipped.append((d, repr(e)))
            continue
        if x.dim() != 3 or x.shape[1] != 4:
            skipped.append((d, 'coords shape %s, expected [N,4,3]' % (tuple(x.shape),)))
            continue
        x = x.float()
        mp = os.path.join(a.tensor_root, d, a.masks_name)
        if os.path.isfile(mp):
            try:
                mask = torch.load(mp, weights_only=True).float().reshape(-1)
            except Exception:
                mask = torch.ones(x.shape[0])
        else:
            mask = torch.ones(x.shape[0])
        if mask.numel() != x.shape[0]:
            mask = torch.ones(x.shape[0])
        s_, c_ = offsets_for_protein(x, mask, seps)
        for s in s_:
            tot_sum[s] += s_[s]
            tot_cnt[s] += c_[s]
        n_prot += 1

    if n_prot == 0:
        print('ERROR: no proteins read from %s -- check --tensor_root and --coords_name'
              % a.tensor_root)
        return 1

    per_sep = {}
    for s in seps:
        if tot_cnt[s] > 0:
            per_sep[s] = (tot_sum[s] / tot_cnt[s]).tolist()

    print('\nproteins used: %d   skipped: %d' % (n_prot, len(skipped)))
    for d, why in skipped[:5]:
        print('   skipped %s: %s' % (d, why))

    print('\nsep_dependence -- offset[c] in ANGSTROM at each |i-j| '
          '(raw on-disk units, before the 0.1 model scaling)')
    hdr = '  %-8s' % 'channel' + ''.join('%10s' % ('s=%d' % s) for s in seps if s in per_sep)
    print(hdr)
    max_drift = 0.0
    for c in range(16):
        row = '  %-8s' % channel_name(c)
        vals = []
        for s in seps:
            if s in per_sep:
                row += '%10.3f' % per_sep[s][c]
                vals.append(per_sep[s][c])
        if len(vals) > 1:
            max_drift = max(max_drift, max(vals) - min(vals))
        print(row)
    print('\n  n_pairs: ' + '  '.join('s=%d:%d' % (s, tot_cnt[s]) for s in seps
                                      if s in per_sep))
    print('  max spread of any channel across separations: %.3f A' % max_drift)
    if max_drift > 0.15:
        print('  WARNING: the offset is separation-DEPENDENT by more than 0.15 A. The'
              '\n  offset arm models it as ONE constant per channel, so it is an'
              '\n  approximation here. Record this in the write-up; it is a limitation'
              '\n  of the arm, not a bug.')

    pooled = torch.zeros(16, dtype=torch.float64)
    pooled_n = 0
    for s in pool:
        if s in per_sep and tot_cnt[s] > 0:
            pooled += torch.tensor(per_sep[s], dtype=torch.float64) * tot_cnt[s]
            pooled_n += tot_cnt[s]
    if pooled_n == 0:
        print('ERROR: no pairs at the pooled separations %s' % pool)
        return 1
    pooled = (pooled / pooled_n)
    # The CA-CA channel is the reference, so it MUST be exactly zero.
    assert abs(float(pooled[CA * 4 + CA])) < 1e-9, \
        'CA-CA offset is not zero -- the reference subtraction is wrong'
    pooled_ang = [round(float(v), 4) for v in pooled]
    # The JSON 'offsets' key is consumed by --coil_offsets_path, and
    # _coil_channel_offsets treats that path as ALREADY being in model units (it scales
    # only its own built-in Angstrom fallback). So scale here, once, and record both.
    pooled_model = [round(float(v) * a.coord_scale, 6) for v in pooled]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    payload = {
        'offsets': pooled_model,
        'units': 'model units (angstrom * coord_scale)',
        'offsets_angstrom': pooled_ang,
        'coord_scale': a.coord_scale,
        'coord_scale_note': ('train.normalize_batch multiplies coords by '
                             'NANO_TO_ANGSTROM=0.1 before any graph is built, so the '
                             'coil sees coordinates 10x smaller than Angstrom. The '
                             '"offsets" key is pre-scaled for direct use via '
                             '--coil_offsets_path.'),
        'channel_index': 'atom_i*4 + atom_j, atom order (N, CA, C, CB)',
        'reference_channel': CA * 4 + CA,
        'n_proteins': n_prot,
        'pooled_separations': pool,
        'n_pairs_pooled': pooled_n,
        'per_separation_angstrom': {str(s): per_sep[s] for s in per_sep},
        'max_spread_across_separations_angstrom': round(max_drift, 4),
        'tensor_root': a.tensor_root,
    }
    with open(a.out, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print('\nwrote %s' % a.out)
    print('  offsets (model units, for --coil_offsets_path): %s' % pooled_model)

    print('\n--- drop-in replacement for _COIL_OFFSETS_FALLBACK_ANGSTROM ---')
    print('# NOTE: this literal is in ANGSTROM. _coil_channel_offsets multiplies it by')
    print('# _COIL_COORD_SCALE. Do NOT paste the model-unit numbers here, or the scale')
    print('# is applied twice and the offsets become ~100x too small.')
    print('_COIL_OFFSETS_FALLBACK_ANGSTROM = [')
    for r in range(4):
        names = ', '.join(channel_name(r * 4 + k) for k in range(4))
        print('    # atom_i = %-2s (channels %2d..%2d): %s'
              % (ATOMS[r], r * 4, r * 4 + 3, names))
        print('    ' + ', '.join('%.4f' % pooled_ang[r * 4 + k] for k in range(4)) + ',')
    print(']')
    print('--- end ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
