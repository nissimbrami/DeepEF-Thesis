"""W9 gates -- must all pass before --metal_features is used in any run.

Usage
-----
    python scripts/gate_w9.py                      # G1 + block-arithmetic gates only
    python scripts/gate_w9.py --csv data/Processed_K50_dG_datasets/metal_sites.csv \
                              --tensors data/Processed_K50_dG_datasets/training_data

Gates
-----
G1  OFF is BIT-IDENTICAL. torch.equal, never allclose, on both graph builders.
G2a The folded block DIFFERS from the unfolded block for an annotated protein.
    If it does not, the U7 cancellation bug is back and the feature is inert:
    it would vanish exactly in dG = E_u - E_f.
G2b The block is EXACTLY zero for a protein with no annotation.
G2c Coordination number is right: a 4-residue site gives 4/6 on all four residues.
G3  Slot arithmetic. The block lands at metal_start(cfg) and the right-anchored
    indices (one_hot at -20, llm at -(E+20)) are unmoved, with and without W5.
G4  Annotation sanity, only when --csv is given:
      - every protein resolves to a tensor directory
      - every resi is inside that protein's residue count
      - >=90% of annotated residues are in {C,H,D,E,M,N,Q,S,T,Y}
        A failure here is an OFF-BY-ONE in the indexing, not bad biology, and the
        script searches offsets in [-5,5] and prints the one that would fix it.

Exit code 0 = all gates pass.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metal_features import (          # noqa: E402
    METAL_DIM, METAL_VOCAB, METAL_OTHER_INDEX, COORD_NUM_CAP, COORDINATING_AA,
    MetalAnnotations, metal_features, metal_or_none, metal_dim, metal_start,
    set_annotations, set_current_protein, clear_current_protein,
)

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'      # matches train_utils.AA_MAP

_FAILURES = []


class _Cfg(object):
    """Stand-in for model_cfg.CFG so the gates run without the training package."""
    def __init__(self, **kw):
        self.metal_features = False
        self.burial_features = False
        self.emb_input_dim = 1024
        for k, v in kw.items():
            setattr(self, k, v)


def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print('[%s] %s%s' % (status, name, ('  -- ' + detail) if detail else ''))
    if not ok:
        _FAILURES.append(name)
    return ok


# ---------------------------------------------------------------------------
# Synthetic fixture -- no dataset needed
# ---------------------------------------------------------------------------

def _fixture(n=24, emb=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 3, generator=g)
    one_hot = torch.zeros(n, 20)
    one_hot[torch.arange(n), torch.randint(0, 20, (n,), generator=g)] = 1.0
    e = torch.randn(n, emb, generator=g)
    mask = torch.ones(n)
    return x, one_hot, e, mask


def _toy_table():
    """One 4-residue Zn site and one 2-residue Ca site in protein P1. P2 has none."""
    rows = [('P1', 3, 'ZN', 'ZN1'), ('P1', 6, 'ZN', 'ZN1'),
            ('P1', 9, 'ZN', 'ZN1'), ('P1', 12, 'ZN', 'ZN1'),
            ('P1', 18, 'CA', 'CA1'), ('P1', 20, 'CA', 'CA1')]
    counts = {}
    for p, r, m, s in rows:
        counts[(p, s)] = counts.get((p, s), 0) + 1
    table = {}
    for p, r, m, s in rows:
        table.setdefault(p, {})[r] = (m, counts[(p, s)])
    return MetalAnnotations(table=table, path='<toy>', n_rows=len(rows))


# ---------------------------------------------------------------------------
# G1 -- OFF is bit-identical
# ---------------------------------------------------------------------------

def gate_g1():
    print('\n--- G1: OFF is bit-identical (torch.equal) ---')
    x, one_hot, emb, mask = _fixture()
    cfg_off = _Cfg(metal_features=False)

    # The OFF path must return None so the caller builds the ORIGINAL cat with no
    # extra tensor. A zero block would be numerically equal but is NOT the same
    # tensor shape, and would change fc1 widths.
    b_f = metal_or_none(x, mask, True, cfg_off)
    b_u = metal_or_none(x, mask, False, cfg_off)
    check('G1.1 metal_or_none returns None when off (folded)', b_f is None)
    check('G1.2 metal_or_none returns None when off (unfolded)', b_u is None)
    check('G1.3 metal_dim == 0 when off', metal_dim(cfg_off) == 0,
          'got %d' % metal_dim(cfg_off))

    # Simulate the two torch.cat branches exactly as get_graph writes them.
    D = torch.randn(x.shape[0], 16); Fb = torch.randn(x.shape[0], 32)
    ref = torch.cat([D, Fb, emb, one_hot], dim=1)
    S = None
    M = metal_or_none(x, mask, True, cfg_off)
    parts = [D, Fb] + ([S] if S is not None else []) + ([M] if M is not None else [])
    got = torch.cat(parts + [emb, one_hot], dim=1)
    check('G1.4 node tensor byte-identical when off (torch.equal)',
          torch.equal(ref, got),
          'shapes ref=%s got=%s' % (tuple(ref.shape), tuple(got.shape)))

    # And with no annotations loaded but the flag ON, the block must still be a
    # legal all-zero block, never a crash.
    set_annotations(None); clear_current_protein()
    cfg_on = _Cfg(metal_features=True)
    M = metal_or_none(x, mask, True, cfg_on)
    check('G1.5 flag on + no table -> all-zero [N,9], no crash',
          M is not None and M.shape == (x.shape[0], METAL_DIM)
          and torch.equal(M, torch.zeros_like(M)))


# ---------------------------------------------------------------------------
# G2 -- it does what it is named for
# ---------------------------------------------------------------------------

def gate_g2():
    print('\n--- G2: the feature expresses coordination ---')
    set_annotations(_toy_table())
    n = 24

    set_current_protein('P1')
    folded = metal_features(n, folded=True)
    unfolded = metal_features(n, folded=False)

    check('G2a folded block DIFFERS from unfolded (U7 cancellation check)',
          not torch.equal(folded, unfolded),
          'if equal, the column cancels exactly in E_u - E_f and the lever is inert')
    check('G2a.2 unfolded block is EXACTLY zero',
          torch.equal(unfolded, torch.zeros_like(unfolded)))
    check('G2a.3 folded block is NOT all zero for an annotated protein',
          folded.abs().sum().item() > 0)

    set_current_protein('P2')          # unannotated
    p2 = metal_features(n, folded=True)
    check('G2b unannotated protein -> exactly zero',
          torch.equal(p2, torch.zeros_like(p2)))

    clear_current_protein()
    none_p = metal_features(n, folded=True)
    check('G2b.2 unset current protein -> exactly zero',
          torch.equal(none_p, torch.zeros_like(none_p)))

    set_current_protein('P1')
    f = metal_features(n, folded=True)
    zn_rows = [3, 6, 9, 12]
    ca_rows = [18, 20]
    ok_flag = all(f[r, 0].item() == 1.0 for r in zn_rows + ca_rows)
    check('G2c.1 first-shell flag set on every annotated residue', ok_flag)
    zn_idx = 1 + METAL_VOCAB.index('ZN')
    ca_idx = 1 + METAL_VOCAB.index('CA')
    check('G2c.2 metal identity one-hot correct',
          all(f[r, zn_idx].item() == 1.0 for r in zn_rows) and
          all(f[r, ca_idx].item() == 1.0 for r in ca_rows))
    check('G2c.3 identity one-hot sums to exactly 1 on annotated rows',
          all(abs(f[r, 1:1 + 7].sum().item() - 1.0) < 1e-6 for r in zn_rows + ca_rows))
    exp_zn = 4.0 / COORD_NUM_CAP
    exp_ca = 2.0 / COORD_NUM_CAP
    check('G2c.4 coordination number = 4/6 on the four-residue Zn site',
          all(abs(f[r, 8].item() - exp_zn) < 1e-6 for r in zn_rows),
          'expected %.4f, got %s' % (exp_zn, [round(f[r, 8].item(), 4) for r in zn_rows]))
    check('G2c.5 coordination number = 2/6 on the two-residue Ca site',
          all(abs(f[r, 8].item() - exp_ca) < 1e-6 for r in ca_rows))
    unann = [r for r in range(n) if r not in zn_rows + ca_rows]
    check('G2c.6 non-coordinating residues are exactly zero',
          all(f[r].abs().sum().item() == 0.0 for r in unann))

    # An unknown symbol must land in OTHER, not be dropped.
    tbl = MetalAnnotations(table={'PX': {2: ('IR', 3)}}, path='<toy>', n_rows=1)
    set_annotations(tbl); set_current_protein('PX')
    fx = metal_features(8, folded=True)
    check('G2c.7 unknown metal symbol -> OTHER slot, not dropped',
          fx[2, 0].item() == 1.0 and fx[2, 1 + METAL_OTHER_INDEX].item() == 1.0)

    # Coordination number is clipped, never > 1.
    tbl = MetalAnnotations(table={'PY': {1: ('CA', 9)}}, path='<toy>', n_rows=1)
    set_annotations(tbl); set_current_protein('PY')
    fy = metal_features(8, folded=True)
    check('G2c.8 coordination number clipped to 1.0 for an over-large site',
          abs(fy[1, 8].item() - 1.0) < 1e-6)


# ---------------------------------------------------------------------------
# G3 -- slot arithmetic
# ---------------------------------------------------------------------------

def gate_g3():
    print('\n--- G3: slot arithmetic and right-anchored indices ---')
    E = 1024
    # (burial on/off) x (W6 descriptors off / 16-dim pca16). The metal block must land
    # at 48 + solv_dim + desc_dim in ALL FOUR combinations. This is the offset that is
    # dangerous: a wrong value does not crash, it silently feeds the wrong columns.
    for burial, desc_k in ((False, 0), (True, 0), (False, 16), (True, 16)):
        cfg = _Cfg(metal_features=True, burial_features=burial, emb_input_dim=E,
                   aa_descriptors=('pca16' if desc_k else 'none'),
                   aa_desc_dim=(desc_k or None))
        solv = 3 if burial else 0
        tag = 'burial=%s,desc=%d' % (burial, desc_k)
        start = metal_start(cfg)
        exp_start = 48 + solv + desc_k
        check('G3 metal_start == 48+solv+desc (%s)' % tag,
              start == exp_start, 'got %d, expected %d' % (start, exp_start))

        n = 12
        D = torch.randn(n, 16); Fb = torch.randn(n, 32)
        S = torch.randn(n, solv) if solv else None
        Dsc = torch.randn(n, desc_k) if desc_k else None
        M = torch.arange(n * METAL_DIM, dtype=torch.float32).reshape(n, METAL_DIM)
        emb = torch.randn(n, E); one_hot = torch.eye(20)[torch.arange(n) % 20]
        parts = ([D, Fb] + ([S] if S is not None else [])
                 + ([Dsc] if Dsc is not None else []) + [M, emb, one_hot])
        Fh = torch.cat(parts, dim=1)

        sliced = Fh[:, start:start + METAL_DIM]
        check('G3 the block is recoverable at metal_start (%s)' % tag,
              torch.equal(sliced, M))
        check('G3 one_hot still at -20 (%s)' % tag,
              torch.equal(Fh[:, -20:], one_hot))
        llm_index = -(E + 20)
        check('G3 emb still at -(E+20) (%s)' % tag,
              torch.equal(Fh[:, llm_index:-20], emb))
        exp_w = 16 + 32 + solv + desc_k + METAL_DIM + E + 20
        check('G3 total width = 16+32+solv+desc+9+E+20 (%s)' % tag,
              Fh.shape[1] == exp_w, 'got %d, expected %d' % (Fh.shape[1], exp_w))

        # BUG 1 guard: the GCN slice is x[:, :32] -- D(16) plus HALF of Fb -- NOT
        # x[:, :48]. Widening it to 48 feeds 48+solv+9+20 dims into a layer sized
        # for 52+solv+9. The slice stays 32 and the extra blocks are appended
        # explicitly, exactly as hydro_net does for W5.
        blocks = (([Fh[:, 48:48 + solv]] if solv else [])
                  + ([Fh[:, 48 + solv:48 + solv + desc_k]] if desc_k else [])
                  + [sliced])
        gcn_in = torch.cat([Fh[:, :32]] + blocks + [Fh[:, -20:]], dim=-1)
        exp_gcn = 32 + solv + desc_k + METAL_DIM + 20
        check('G3 GCN width = 32+solv+desc+9+20, slice stays x[:, :32] (%s)' % tag,
              gcn_in.shape[1] == exp_gcn,
              'got %d, expected %d' % (gcn_in.shape[1], exp_gcn))
        gat_in = torch.cat([Fh[:, :16]] + blocks + [Fh[:, -20:]], dim=-1)
        exp_gat = 16 + solv + desc_k + METAL_DIM + 20
        check('G3 GAT width = 16+solv+desc+9+20 (%s)' % tag,
              gat_in.shape[1] == exp_gat,
              'got %d, expected %d' % (gat_in.shape[1], exp_gat))


# ---------------------------------------------------------------------------
# G4 -- annotation file sanity, only with --csv
# ---------------------------------------------------------------------------

def _residue_letters(tensor_dir, protein):
    """Decode one_hot_encodings.pt row 0 (wild type) into a residue string."""
    p = os.path.join(tensor_dir, protein, 'one_hot_encodings.pt')
    if not os.path.exists(p):
        return None
    oh = torch.load(p, weights_only=True)
    # [n_variants, N, 21] in the stored form; train.py drops the last column.
    while oh.dim() > 3:
        oh = oh.squeeze(0)
    wt = oh[0] if oh.dim() == 3 else oh
    wt = wt[:, :20]
    idx = wt.argmax(dim=-1)
    active = wt.sum(dim=-1) > 0
    return ''.join(AA_ORDER[i] if a else '-' for i, a in zip(idx.tolist(), active.tolist()))


def gate_g4(csv_path, tensor_dir):
    print('\n--- G4: annotation file sanity ---')
    ann = MetalAnnotations.from_csv(csv_path)
    print('    %r' % (ann,))
    check('G4.0 file parsed', True, '%d proteins, %d rows' % (len(ann.table), ann.n_rows))

    if ann.n_rows == 0:
        print('    NOTE: the annotation file is EMPTY. That is a legitimate delivery '
              '(see ANNOTATION_SPEC.md section 8) and means the dataset has no metal '
              'sites. The W9 arm should then be REPORTED, not RUN.')
        return

    if not tensor_dir or not os.path.isdir(tensor_dir):
        print('    SKIP G4.1-G4.3: --tensors not given or not a directory.')
        return

    available = set(os.listdir(tensor_dir))
    missing = [p for p in ann.proteins() if p not in available]
    check('G4.1 every annotated protein has a tensor directory',
          not missing,
          'missing: %s' % (missing[:10],) if missing else
          'all %d resolve' % len(ann.table))

    # G4.2 in-range, G4.3 coordinating-residue identity + offset search
    total = 0
    coordinating = 0
    out_of_range = []
    per_offset = {o: 0 for o in range(-5, 6)}
    for prot in ann.proteins():
        if prot not in available:
            continue
        letters = _residue_letters(tensor_dir, prot)
        if letters is None:
            continue
        n = len(letters)
        for resi, (metal, cn) in ann.table[prot].items():
            if resi >= n:
                out_of_range.append((prot, resi, n))
                continue
            total += 1
            if letters[resi] in COORDINATING_AA:
                coordinating += 1
            for o in per_offset:
                j = resi + o
                if 0 <= j < n and letters[j] in COORDINATING_AA:
                    per_offset[o] += 1

    check('G4.2 every resi is inside its protein residue count',
          not out_of_range,
          'out of range: %s' % (out_of_range[:10],) if out_of_range else 'ok')

    if total:
        frac = coordinating / float(total)
        ok = frac >= 0.90
        check('G4.3 >=90%% of annotated residues can coordinate a metal', ok,
              '%d/%d = %.1f%%' % (coordinating, total, 100 * frac))
        if not ok:
            best = max(per_offset, key=lambda o: per_offset[o])
            print('    OFF-BY-ONE DIAGNOSIS: offset %+d would give %d/%d = %.1f%%.' %
                  (best, per_offset[best], total, 100 * per_offset[best] / float(total)))
            print('    If offset 0 is not the best, the annotation is in PDB resSeq '
                  'numbering, not 0-based tensor numbering. See ANNOTATION_SPEC.md '
                  'section 4 -- do NOT proceed until this is resolved.')
    else:
        print('    SKIP G4.3: no annotated residue resolved to a stored tensor.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='path to metal_sites.csv (enables G4)')
    ap.add_argument('--tensors', default=None, help='tensor_root_dir (enables G4.1-G4.3)')
    a = ap.parse_args()

    gate_g1()
    gate_g2()
    gate_g3()
    if a.csv:
        gate_g4(a.csv, a.tensors)
    else:
        print('\n--- G4 SKIPPED: no --csv. The annotation file does not exist yet; '
              'see ANNOTATION_SPEC.md. G1-G3 are complete without it. ---')

    print('\n' + '=' * 62)
    if _FAILURES:
        print('W9 GATES FAILED (%d): %s' % (len(_FAILURES), _FAILURES))
        return 1
    print('W9 GATES PASSED. --metal_features is safe to enable '
          '(and is inert without an annotation file).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
