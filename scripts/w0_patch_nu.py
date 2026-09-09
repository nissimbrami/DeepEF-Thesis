"""Patch scripts/w0_dg.py to support --nu and --b (P1: the Flory exponent sweep).

Backward compatible: with neither flag given the script's behaviour and its printed
table are unchanged apart from two ADDED diagnostic columns.
"""
import io

p = 'scripts/w0_dg.py'
s = io.open(p, encoding='utf-8').read()

# ---------------------------------------------------------------- 1) CLI args
old_args = """ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
A = ap.parse_args()"""
new_args = """ap.add_argument('--tm_path', default='./data/ThermoMPNN/mega_test.csv')
# --- P1: the Flory nu / b sweep -------------------------------------------------
# nu is the coil scaling exponent in d(i,j) = b*|i-j|^nu. CFG.flory_nu defaults to
# 0.5 (theta / ideal chain); a denatured protein in water is a self-avoiding walk at
# nu ~= 0.588 (Kohn 2004 PNAS: Rg ~ N^0.598 over 28 proteins). NEVER swept before.
ap.add_argument('--nu', type=float, nargs='+', default=None,
                help='one or more coil exponents; each is run as its own condition. '
                     'Default None = no sweep (historical behaviour, CFG default 0.5).')
# b is the effective segment length, in ANGSTROM. train_utils stores it as a module
# constant _COIL_B_FIXED ALREADY multiplied by _COIL_COORD_SCALE (0.1), because the
# model is trained on coordinates in model units, not Angstrom. We therefore set that
# module constant, in the same units, rather than passing Angstrom into the graph.
ap.add_argument('--b', type=float, nargs='+', default=None,
                help='one or more segment lengths in ANGSTROM (5.82 = ours, 4.97 = Kohn). '
                     'Only meaningful with coil_b=fixed. Default None = leave at 5.82.')
ap.add_argument('--sweep_only', action='store_true',
                help='skip the five historical CONDS and run only the nu x b grid.')
ap.add_argument('--ref_fix', default='',
                help="optional 'PROT=value' override of the reference WT dG "
                     "(2K5H=4.805470 corrects the known bad reference row).")
A = ap.parse_args()"""
assert old_args in s, 'args anchor missing'
s = s.replace(old_args, new_args)

# ------------------------------------------------- 2) condition list + b setter
old_conds = """CONDS = ['base', 'coil', 'coil_fixed_b', 'coil_ca_only', 'noemb']
rows = []"""
new_conds = """import train_utils as _TU
_COORD_SCALE = getattr(_TU, '_COIL_COORD_SCALE', 0.1)
_B_DEFAULT_ANG = getattr(_TU, '_COIL_B_FIXED_ANGSTROM', 5.82)

CONDS = [] if A.sweep_only else ['base', 'coil', 'coil_fixed_b', 'coil_ca_only', 'noemb']
# SWEEP holds (name, nu, b_angstrom). Every entry is coil + coil_b=fixed + broadcast
# channels, i.e. EXACTLY the historical 'coil_fixed_b' condition with nu and b varied.
# The nu=0.5 / b=5.82 cell must therefore reproduce coil_fixed_b to machine precision:
# that equality is the sweep's own no-op gate.
SWEEP = []
if A.nu is not None:
    for _nu in A.nu:
        for _b in (A.b if A.b is not None else [_B_DEFAULT_ANG]):
            SWEEP.append(('nu%.3f_b%.2f' % (_nu, _b), float(_nu), float(_b)))
ALL = CONDS + [t[0] for t in SWEEP]
print('conditions: %s' % ', '.join(ALL))
_REF_FIX = {}
if A.ref_fix:
    _k, _v = A.ref_fix.split('=')
    _REF_FIX[_k] = float(_v)
    print('reference override: %s -> %s' % (_k, _REF_FIX[_k]))
rows = []"""
assert old_conds in s, 'conds anchor missing'
s = s.replace(old_conds, new_conds)

# ------------------------------------------------ 3) reference-row override
old_true = "            true = float(dg.reshape(-1)[0])"
new_true = """            true = float(dg.reshape(-1)[0])
            if name in _REF_FIX:
                true = _REF_FIX[name]"""
assert old_true in s, 'true anchor missing'
s = s.replace(old_true, new_true)

# ------------------------------------------------ 4) run the sweep conditions
old_reset = """            CFG.flory_unfolded=False; CFG.coil_b='fitted'; CFG.coil_channels='broadcast'; CFG.unfolded_emb='full'
            rows.append(r)"""
new_reset = """            for cname, cnu, cb in SWEEP:
                CFG.flory_unfolded = True
                CFG.coil_b = 'fixed'
                CFG.coil_channels = 'broadcast'
                CFG.unfolded_emb = 'full'
                CFG.flory_nu = cnu
                # b is a MODULE constant in model units, not a CFG field.
                _TU._COIL_B_FIXED = cb * _COORD_SCALE
                g = get_unfolded_graph(coords, oh0, emb0, mask).unsqueeze(0)
                E_u = float(model(g).reshape(-1)[0])
                r['pred_' + cname] = E_u - E_f
            CFG.flory_unfolded=False; CFG.coil_b='fitted'; CFG.coil_channels='broadcast'; CFG.unfolded_emb='full'
            CFG.flory_nu = 0.5
            _TU._COIL_B_FIXED = _B_DEFAULT_ANG * _COORD_SCALE
            rows.append(r)"""
assert old_reset in s, 'reset anchor missing'
s = s.replace(old_reset, new_reset)

# --------------------------------------- 5) report: std(b_p) + degeneracy column
old_rep = """true = np.array([r['true'] for r in rows])
print('\\n%-14s %10s %10s %10s %10s' % ('condition','MAE','corr','std(err)','bias'))
res={}
for c in CONDS:
    pred = np.array([r['pred_'+c] for r in rows])
    err = pred - true
    mae=float(np.abs(err).mean()); sd=float(err.std()); bias=float(err.mean())
    cr=float(np.corrcoef(pred,true)[0,1]) if pred.std()>0 else float('nan')
    res[c]=dict(mae=mae, corr=cr, std_err=sd, bias=bias)
    print('%-14s %10.4f %10.4f %10.4f %10.4f' % (c,mae,cr,sd,bias))

b=res['base']
print('\\n--- change vs base (negative MAE / std = BETTER) ---')
for c in CONDS[1:]:
    print('  %-14s dMAE=%+.4f  dstd(err)=%+.4f  dcorr=%+.4f'
          % (c, res[c]['mae']-b['mae'], res[c]['std_err']-b['std_err'], res[c]['corr']-b['corr']))"""
new_rep = """true = np.array([r['true'] for r in rows])
STD_TRUE = float(true.std())
# The degenerate attractor: a condition that DELETES the prediction sends pred -> const,
# hence b_p -> -true and std(b_p) -> std(true WT dG). A 'win' that lands there is void.
print('\\nstd(true WT dG) = %.4f   <-- the degenerate attractor for std(b_p)' % STD_TRUE)
print('\\n%-18s %10s %10s %12s %10s %12s %10s' %
      ('condition','MAE','corr','std(b_p)','bias','std(predWT)','n_under'))
res={}
for c in ALL:
    pred = np.array([r['pred_'+c] for r in rows])
    err = pred - true
    mae=float(np.abs(err).mean()); sd=float(err.std()); bias=float(err.mean())
    sp=float(pred.std())
    cr=float(np.corrcoef(pred,true)[0,1]) if pred.std()>0 else float('nan')
    cbp=float(np.corrcoef(err,true)[0,1]) if err.std()>0 else float('nan')
    res[c]=dict(mae=mae, corr=cr, std_err=sd, bias=bias, std_pred=sp,
                corr_bp_true=cbp, n_under=int((err<0).sum()), n=len(err))
    print('%-18s %10.4f %10.4f %12.4f %10.4f %12.4f %10d'
          % (c,mae,cr,sd,bias,sp,res[c]['n_under']))

if 'base' in res:
    b=res['base']
    print('\\n--- change vs base (negative MAE / std = BETTER) ---')
    for c in ALL[1:]:
        print('  %-18s dMAE=%+.4f  dstd(b_p)=%+.4f  dcorr=%+.4f'
              % (c, res[c]['mae']-b['mae'], res[c]['std_err']-b['std_err'], res[c]['corr']-b['corr']))"""
assert old_rep in s, 'report anchor missing'
s = s.replace(old_rep, new_rep)

old_json = """json.dump(dict(checkpoint=A.ckpt, n=len(rows), metric='ABSOLUTE wild-type dG',
               results=res, per_protein=rows), open(A.out,'w'), indent=2)"""
new_json = """json.dump(dict(checkpoint=A.ckpt, n=len(rows), metric='ABSOLUTE wild-type dG',
               std_true_wt_dg=STD_TRUE, ref_fix=_REF_FIX,
               sweep=[dict(name=n_, nu=nu_, b_angstrom=b_) for n_, nu_, b_ in SWEEP],
               results=res, per_protein=rows), open(A.out,'w'), indent=2)"""
assert old_json in s, 'json anchor missing'
s = s.replace(old_json, new_json)

io.open(p, 'w', encoding='utf-8').write(s)
print('patched OK, %d bytes' % len(s))
