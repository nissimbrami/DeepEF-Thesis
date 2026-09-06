"""gate_u3u4.py -- verification gate for ITEM 15 (U3 coil channels x U4 coil b).

RUN THIS AFTER APPLYING PATCH.md AND BEFORE LAUNCHING ANY U3/U4 RUN.
If any test fails, the arms are not measuring what their names say and the four
cells are wasted.

    cd <repo root>            # the dir containing train_utils.py and model/
    python scripts/gate_u3u4.py

Exit 0 = all pass. Exit 1 = at least one failure, printed with its reason.

No GPU, no checkpoint, no dataset. Runs in seconds on CPU.

WHAT EACH TEST ESTABLISHES
  1  channel layout        the 16-wide last dim is atom_i*4 + atom_j, so CA-CA is 5.
                           If this fails, ca_only zeroes the WRONG channel and every
                           U3 conclusion is void.
  2  byte-identity, coil   coil_channels=broadcast + coil_b=fitted reproduces the
                           pre-U3/U4 coil EXACTLY (torch.equal, not allclose).
  3  byte-identity, base   with flory_unfolded=False the new code is unreachable and
                           the tridiagonal baseline is untouched (torch.equal).
  4  ca_only post-kernel   the 15 non-CA channels are EXACTLY 0 after the kernel, not
                           1.0. This is the trap described in PATCH.md.
  5  ca_only CA survives   the CA-CA channel is NOT zero -- i.e. we did not zero all 16.
  6  offset differentiates the 16 channels are NOT all identical under offset, which is
                           the entire point of U3.
  7  broadcast IS degenerate  demonstrates the defect the arms exist to fix: under
                           broadcast all 16 summed channels are equal, so the state is
                           trivially separable. If this test ever FAILS, the premise of
                           ITEM 15 is wrong and the arms need re-motivating.
  8  fixed b is protein-independent  coil_b=fixed gives the same b for two proteins with
                           different folded geometry; fitted does not.
  9  shape invariance      all six (channels x b) combinations return the identical
                           shape as the baseline, so hydro_net needs no edit.
 10  guard rails           bad enum values raise, and a 16-entry offset table is enforced.
"""
import os
import sys
import json
import tempfile
import traceback

import torch

# Import the patched code from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import train_utils as TU
from model.model_cfg import CFG

FAILURES = []
PASSES = []


def check(name, cond, detail=''):
    if cond:
        PASSES.append(name)
        print('  PASS  %s' % name)
    else:
        FAILURES.append((name, detail))
        print('  FAIL  %s   %s' % (name, detail))


COORD_SCALE = 0.1   # train.NANO_TO_ANGSTROM, applied by normalize_batch before get_graph


def fake_protein(n=24, seed=0):
    """A plausible [N,4,3] backbone: a coarse alpha helix with per-atom offsets, plus
    mask, one_hot and a small embedding.

    IMPORTANT -- the coordinates are returned in MODEL UNITS, i.e. already multiplied by
    COORD_SCALE, because train.normalize_batch does exactly that before any graph is
    built. Building this fixture in Angstrom would make test 8b (the units trap) pass
    for the wrong reason and would hide the very bug that test exists to catch.
    Angstrom geometry: 2.3 A helix radius, 1.5 A rise, ~3.8 A CA-CA neighbour distance.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32)
    ca = torch.stack([2.3 * torch.cos(t * 1.75),
                      2.3 * torch.sin(t * 1.75),
                      1.5 * t], dim=1)
    # N, CA, C, CB as small fixed displacements off CA, so intra-residue legs are
    # near-constant -- the same rigid-geometry regime add_cb produces.
    off = torch.tensor([[-1.20, 0.30, -0.35],
                        [0.00, 0.00, 0.00],
                        [1.25, 0.20, 0.35],
                        [-0.30, -1.45, 0.20]])
    x = ca.unsqueeze(1) + off.unsqueeze(0)
    x = x + 0.02 * torch.randn(x.shape, generator=g)
    x = x * COORD_SCALE                    # -> model units, as normalize_batch delivers
    mask = torch.ones(n)
    mask[-2] = 0                      # exercise the mask path
    one_hot = torch.zeros(n, 20)
    one_hot[torch.arange(n), torch.randint(0, 20, (n,), generator=g)] = 1.0
    emb = torch.randn(n, CFG.emb_input_dim if hasattr(CFG, 'emb_input_dim') else 1024,
                      generator=g)
    return x, one_hot, emb, mask


def set_cfg(**kw):
    for k, v in kw.items():
        setattr(CFG, k, v)


def reset_cfg():
    set_cfg(flory_unfolded=False, flory_nu=0.5, coil_channels='broadcast',
            coil_b='fitted', coil_offsets_path=None, unfolded_emb='full',
            burial_features=False)
    if hasattr(CFG, 'coil_offsets'):
        delattr(CFG, 'coil_offsets')
    TU._COIL_OFFSETS_CACHE.clear()


# ---------------------------------------------------------------------------
# Reference implementations: the code EXACTLY as it stood before this patch.
# Test 2 and test 3 compare against these, so they are the definition of
# "byte-identical to current behaviour".
# ---------------------------------------------------------------------------

def ref_flory_pre_u3u4(x, one_hot, emb, mask, gaussian_coef):
    """_flory_unfolded_graph as it stood BEFORE ITEM 15."""
    nu = float(getattr(CFG, 'flory_nu', 0.5))
    if not (0.0 < nu <= 1.0):
        raise ValueError('flory_nu out of range')
    N, N_atoms, _ = x.shape
    n_atom_dist = N_atoms * N_atoms
    dev = x.device
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    ca = x[:, 1, :]
    if N > 1:
        b = torch.linalg.norm(ca[1:] - ca[:-1], dim=-1).mean()
        b = torch.clamp(b, min=1e-3)
    else:
        b = torch.tensor(3.8, device=dev)
    d_coil = b * torch.pow(sep + 1e-6, nu)
    D = d_coil.unsqueeze(-1).expand(N, N, n_atom_dist).contiguous()
    D = torch.relu(torch.exp(gaussian_coef * D ** 2))
    mask_index = torch.where(mask == 0)
    D[mask_index[0], :, :] = 0
    D[:, mask_index[0], :] = 0
    Fb = TU.get_bonded_features(D)
    D = D.sum(dim=1)
    D = TU.F.normalize(D, p=2, dim=0)
    emb = TU.F.normalize(TU._unfolded_emb(emb), p=2, dim=0)
    _S = TU._solv_or_none(x, one_hot, mask, folded=False)
    return (torch.cat([D, Fb, emb, one_hot], dim=1) if _S is None
            else torch.cat([D, Fb, _S, emb, one_hot], dim=1))


def ref_tridiagonal(x, one_hot, emb, mask, gaussian_coef):
    """The tridiagonal baseline unfolded graph, unchanged by this patch."""
    D = TU.get_dist_matrix(x)
    D = torch.relu(torch.exp(gaussian_coef * D ** 2))
    mask_index = torch.where(mask == 0)
    D[mask_index[0], :, :] = 0
    D[:, mask_index[0], :] = 0
    D = TU.zero_except_udiagonal(D)
    Fb = TU.get_bonded_features(D)
    D = D.sum(dim=1)
    D = TU.F.normalize(D, p=2, dim=0)
    emb = TU.F.normalize(TU._unfolded_emb(emb), p=2, dim=0)
    _S = TU._solv_or_none(x, one_hot, mask, folded=False)
    return (torch.cat([D, Fb, emb, one_hot], dim=1) if _S is None
            else torch.cat([D, Fb, _S, emb, one_hot], dim=1))


# ---------------------------------------------------------------------------

def raw_coil_D(x, mask, gaussian_coef):
    """Re-run the coil's D pipeline up to just after the kernel + mask, so tests can
    inspect the per-channel values that get_unfolded_graph then reduces away.
    Mirrors the patched _flory_unfolded_graph exactly."""
    nu = float(getattr(CFG, 'flory_nu', 0.5))
    N, N_atoms, _ = x.shape
    n_atom_dist = N_atoms * N_atoms
    dev = x.device
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    ca = x[:, 1, :]
    b = TU._coil_bond_length(ca, N, dev, x.dtype)
    d_coil = b * torch.pow(sep + 1e-6, nu)
    D = TU._coil_expand_channels(d_coil, n_atom_dist)
    D = torch.relu(torch.exp(gaussian_coef * D ** 2))
    if getattr(CFG, 'coil_channels', 'broadcast') == 'ca_only':
        nonca = [c for c in range(D.shape[-1]) if c != TU._CA_CA_CHANNEL]
        D[:, :, nonca] = 0
    mask_index = torch.where(mask == 0)
    D[mask_index[0], :, :] = 0
    D[:, mask_index[0], :] = 0
    return D, b


def main():
    gc = CFG.gaussian_coef
    x, one_hot, emb, mask = fake_protein()

    print('\n[1] channel layout: last dim index == atom_i*4 + atom_j, CA-CA == 5')
    reset_cfg()
    D16 = TU.get_dist_matrix(x)
    layout_ok = True
    for c in range(16):
        ai, aj = c // 4, c % 4
        ref = torch.cdist(x[:, ai, :], x[:, aj, :], p=2)
        if not torch.equal(D16[:, :, c], ref):
            layout_ok = False
    check('channel index == atom_i*4 + atom_j', layout_ok,
          'get_dist_matrix layout changed -- ca_only would zero the WRONG channel')
    check('_CA_CA_CHANNEL == 5', TU._CA_CA_CHANNEL == 5,
          'got %r' % (TU._CA_CA_CHANNEL,))
    check('D[:,:,5] is the CA-CA matrix',
          torch.equal(D16[:, :, TU._CA_CA_CHANNEL], torch.cdist(x[:, 1, :], x[:, 1, :], p=2)))

    print('\n[2] byte-identity: coil default (broadcast + fitted) == pre-U3/U4 coil')
    for nu in (0.5, 0.588, 1.0):
        reset_cfg()
        set_cfg(flory_unfolded=True, flory_nu=nu)
        got = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
        set_cfg(flory_nu=nu)
        want = ref_flory_pre_u3u4(x, one_hot, emb, mask, gc)
        check('coil default == pre-patch coil, nu=%s (torch.equal)' % nu,
              torch.equal(got, want),
              'max abs diff %s' % (got - want).abs().max().item())

    print('\n[3] byte-identity: flory_unfolded=False leaves the tridiagonal baseline alone')
    for cc in ('broadcast',):
        reset_cfg()
        set_cfg(flory_unfolded=False, coil_channels=cc)
        got = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
        want = ref_tridiagonal(x, one_hot, emb, mask, gc)
        check('tridiagonal baseline unchanged (torch.equal)', torch.equal(got, want))
    # And the folded pass must be untouched no matter what the coil flags say.
    reset_cfg()
    f_base = TU.get_graph(x, one_hot, emb, mask, gc)
    set_cfg(flory_unfolded=True, coil_channels='ca_only', coil_b='fixed')
    f_arm = TU.get_graph(x, one_hot, emb, mask, gc)
    check('folded get_graph unaffected by U3/U4 (torch.equal)', torch.equal(f_base, f_arm))

    print('\n[4] ca_only: the 15 non-CA channels are EXACTLY 0 AFTER the kernel')
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_channels='ca_only')
    D, _ = raw_coil_D(x, mask, gc)
    nonca = [c for c in range(16) if c != TU._CA_CA_CHANNEL]
    mx = D[:, :, nonca].abs().max().item()
    check('non-CA channels are exactly zero post-kernel', mx == 0.0,
          'max |value| = %g -- if this is ~1.0 the post-kernel re-zero of EDIT 2c is '
          'MISSING and ca_only is feeding a constant 1.0 into 15 channels' % mx)

    print('\n[5] ca_only: the CA-CA channel is NOT zero (we did not zero all 16)')
    check('CA-CA channel survives', D[:, :, TU._CA_CA_CHANNEL].abs().max().item() > 0.0)

    print('\n[6] offset: the 16 channels are NOT all identical')
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_channels='offset')
    Do, _ = raw_coil_D(x, mask, gc)
    ch_spread = Do[0, 1, :].std().item()
    check('offset makes channels differ', ch_spread > 1e-9,
          'std across channels at (0,1) = %g -- offsets are all equal, the arm is a '
          'no-op relative to broadcast' % ch_spread)
    # And the summed feature vector must not be 16 equal entries.
    go = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
    check('offset: summed D block is not 16 equal entries',
          go[0, :16].std().item() > 1e-9)

    print('\n[7] the DEFECT is real: broadcast makes all 16 summed channels equal')
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_channels='broadcast')
    gb = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
    row_spread = gb[0, :16].std().item()
    check('broadcast D block IS degenerate (all 16 equal)', row_spread < 1e-6,
          'std = %g. If this is large the U3 premise (trivial separability) is WRONG '
          'and ITEM 15 needs re-motivating before the arms are interpreted.' % row_spread)
    # The folded state, by contrast, must NOT be degenerate -- that asymmetry is the
    # separability the arms attack.
    reset_cfg()
    ff = TU.get_graph(x, one_hot, emb, mask, gc)
    check('folded D block is NOT degenerate (the asymmetry exists)',
          ff[0, :16].std().item() > 1e-6)

    print('\n[8] coil_b: fixed is protein-independent, fitted is not')
    x2, oh2, emb2, mask2 = fake_protein(n=24, seed=7)
    x2 = x2 * 1.35                      # a genuinely different folded geometry
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_b='fixed')
    _, b1 = raw_coil_D(x, mask, gc)
    _, b2 = raw_coil_D(x2, mask2, gc)
    check('coil_b=fixed gives the same b for both proteins',
          torch.equal(b1, b2) and abs(float(b1) - TU._COIL_B_FIXED) < 1e-6,
          'b1=%s b2=%s (expected both %s)' % (float(b1), float(b2), TU._COIL_B_FIXED))

    # 8b -- THE UNITS TRAP. train.normalize_batch multiplies coords by
    # NANO_TO_ANGSTROM = 0.1 BEFORE any graph is built, so the coordinates the coil sees
    # are 10x smaller than Angstrom. A literal 5.82 in model units would be ~15x the
    # fitted b, the coil distances would saturate the Gaussian kernel to ~0 everywhere,
    # and the 'fixed' arm would silently be measuring "coil switched off" while being
    # reported as "coil with a fixed b". Pin fixed b to the same order as fitted b.
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_b='fitted')
    _, b_fit_real = raw_coil_D(x, mask, gc)
    ratio = float(TU._COIL_B_FIXED) / float(b_fit_real)
    check('fixed b is the same ORDER as fitted b (units sanity, ratio %.2f)' % ratio,
          (1.0 / 3.0) < ratio < 3.0,
          'ratio %.2f. fixed=%.4f fitted=%.4f. If ratio ~15 the Angstrom->model-unit '
          'conversion (_COIL_COORD_SCALE) is MISSING and the fixed arm is invalid. If '
          'ratio ~0.07 it has been applied twice.'
          % (ratio, float(TU._COIL_B_FIXED), float(b_fit_real)))
    check('_COIL_COORD_SCALE matches train.NANO_TO_ANGSTROM (0.1)',
          abs(TU._COIL_COORD_SCALE - 0.1) < 1e-12,
          'got %r -- if normalize_batch changed, change this together with it'
          % (TU._COIL_COORD_SCALE,))
    # The fixed coil must not collapse the kernel: if every channel comes out ~0 the arm
    # is degenerate regardless of the ratio test.
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_b='fixed')
    Dfix, _ = raw_coil_D(x, mask, gc)
    check('fixed-b coil does not saturate the kernel to ~0',
          Dfix.max().item() > 1e-3 and Dfix.std().item() > 1e-9,
          'max=%g std=%g -- the coil carries no information in this arm'
          % (Dfix.max().item(), Dfix.std().item()))
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_b='fitted')
    _, b1f = raw_coil_D(x, mask, gc)
    _, b2f = raw_coil_D(x2, mask2, gc)
    check('coil_b=fitted differs between proteins (the leak it measures)',
          not torch.equal(b1f, b2f),
          'fitted b identical across two different geometries -- the U4 arms would be '
          'indistinguishable and the cell is wasted')
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_b='fixed')
    gfix = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_b='fitted')
    gfit = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
    check('coil_b changes the unfolded graph', not torch.equal(gfix, gfit))

    print('\n[9] shape invariance across all six (channels x b) combinations')
    reset_cfg()
    set_cfg(flory_unfolded=False)
    base_shape = TU.get_unfolded_graph(x, one_hot, emb, mask, gc).shape
    all_same = True
    for cc in ('broadcast', 'ca_only', 'offset'):
        for bb in ('fitted', 'fixed'):
            reset_cfg()
            set_cfg(flory_unfolded=True, coil_channels=cc, coil_b=bb)
            s = TU.get_unfolded_graph(x, one_hot, emb, mask, gc).shape
            if s != base_shape:
                all_same = False
                print('        %s/%s -> %s != %s' % (cc, bb, s, base_shape))
    check('all six arms return the baseline shape %s' % (tuple(base_shape),), all_same,
          'a shape change would require a hydro_net edit and would hit the known '
          'fc2_/inst_norm/GAT-residual traps')
    # Same again with the W5 solvation block on, since that block sits between Fb and emb.
    reset_cfg()
    set_cfg(burial_features=True, flory_unfolded=False)
    base_shape_s = TU.get_unfolded_graph(x, one_hot, emb, mask, gc).shape
    all_same_s = True
    for cc in ('broadcast', 'ca_only', 'offset'):
        for bb in ('fitted', 'fixed'):
            reset_cfg()
            set_cfg(burial_features=True, flory_unfolded=True, coil_channels=cc, coil_b=bb)
            if TU.get_unfolded_graph(x, one_hot, emb, mask, gc).shape != base_shape_s:
                all_same_s = False
    check('shape invariant with --burial_features on too', all_same_s)
    reset_cfg()

    print('\n[10] guard rails')
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_channels='nonsense')
    try:
        TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
        check('bad coil_channels raises', False, 'no exception')
    except ValueError:
        check('bad coil_channels raises', True)
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_b='nonsense')
    try:
        TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
        check('bad coil_b raises', False, 'no exception')
    except ValueError:
        check('bad coil_b raises', True)
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_channels='offset')
    CFG.coil_offsets = [0.0] * 15
    TU._COIL_OFFSETS_CACHE.clear()
    try:
        TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
        check('short offset table raises', False, 'no exception')
    except ValueError:
        check('short offset table raises', True)
    reset_cfg()

    print('\n[11] --coil_offsets_path overrides the fallback table')
    reset_cfg()
    vals = [0.11 * i for i in range(16)]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'off.json')
        with open(p, 'w') as fh:
            json.dump(vals, fh)
        set_cfg(flory_unfolded=True, coil_channels='offset', coil_offsets_path=p)
        TU._COIL_OFFSETS_CACHE.clear()
        got = TU._coil_channel_offsets(torch.device('cpu'), torch.float32)
        check('offsets loaded from file',
              torch.equal(got, torch.tensor(vals, dtype=torch.float32)))
        g_file = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
    reset_cfg()
    set_cfg(flory_unfolded=True, coil_channels='offset')
    TU._COIL_OFFSETS_CACHE.clear()
    g_fallback = TU.get_unfolded_graph(x, one_hot, emb, mask, gc)
    check('file offsets change the graph vs the fallback',
          not torch.equal(g_file, g_fallback))
    reset_cfg()

    print('\n[12] fallback offset table invariants')
    reset_cfg()
    fb = TU._COIL_OFFSETS_FALLBACK_ANGSTROM
    check('fallback table has 16 entries', len(fb) == 16, 'got %d' % len(fb))
    check('CA-CA fallback offset is exactly 0 (it is the reference channel)',
          fb[TU._CA_CA_CHANNEL] == 0.0, 'got %r' % (fb[TU._CA_CA_CHANNEL],))
    check('all four diagonal channels are 0 (parallel legs cancel)',
          all(fb[k * 4 + k] == 0.0 for k in range(4)))
    check('fallback offsets are physically sized (< 2 A)',
          max(abs(v) for v in fb) < 2.0,
          'max |offset| = %.3f A -- larger than a covalent leg, so probably wrong units'
          % max(abs(v) for v in fb))
    # The scaled table actually used must be ~10x smaller than the Angstrom one.
    got = TU._coil_channel_offsets(torch.device('cpu'), torch.float32)
    want = torch.tensor([v * TU._COIL_COORD_SCALE for v in fb], dtype=torch.float32)
    check('fallback is scaled into model units exactly once', torch.equal(got, want),
          'the Angstrom->model conversion is missing or doubled')

    print('\n' + '=' * 70)
    if FAILURES:
        print('GATE FAILED: %d of %d checks failed' % (len(FAILURES),
                                                       len(FAILURES) + len(PASSES)))
        for n, d in FAILURES:
            print('  - %s   %s' % (n, d))
        return 1
    print('GATE PASSED: %d checks' % len(PASSES))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print('\nGATE ERRORED -- treat as FAILED.')
        sys.exit(1)
