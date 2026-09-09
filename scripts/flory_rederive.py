import json, math, statistics as st
d = json.load(open('results/08_data/w0_dg.json'))
pp = d['per_protein']
conds = ['base','coil','coil_fixed_b','coil_ca_only','noemb']
print('checkpoint:', d['checkpoint']); print('metric:', d['metric']); print('n rows:', len(pp))

def mean(x): return sum(x)/len(x)
def std(x):  # population, matches numpy default
    m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x))
def std1(x):
    m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/(len(x)-1))
def pcc(a,b):
    ma,mb=mean(a),mean(b)
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
    return num/(da*db)

for label, shift in [('RAW (2K5H uncorrected, n=28)',0.0), ('CORRECTED (2K5H +3.0824, n=28)',3.0824)]:
    print('\n'+'='*76); print(label); print('='*76)
    true = [p['true'] + (shift if p['protein']=='2K5H' else 0.0) for p in pp]
    print(f"  std(true WT dG) pop={std(true):.4f}  sample={std1(true):.4f}")
    print(f"  {'cond':<14}{'MAE':>9}{'|mean bp|':>11}{'mean bp':>10}{'std bp pop':>12}{'std bp smp':>12}{'n_under':>9}{'PCC':>8}")
    for c in conds:
        pred = [p['pred_'+c] for p in pp]
        bp   = [pr-tr for pr,tr in zip(pred,true)]   # b_p = pred - true (residual)
        mae  = mean([abs(v) for v in bp])
        nund = sum(1 for v in bp if v < 0)
        print(f"  {c:<14}{mae:>9.4f}{abs(mean(bp)):>11.4f}{mean(bp):>10.4f}{std(bp):>12.4f}{std1(bp):>12.4f}{nund:>9d}{pcc(true,pred):>8.4f}")
    # MAE identity check
    print('  --- MAE == |mean(b_p)| identity ---')
    for c in conds:
        pred=[p['pred_'+c] for p in pp]
        bp=[pr-tr for pr,tr in zip(pred,true)]
        print(f"    {c:<14} max|MAE-|mean bp|| = {abs(mean([abs(v) for v in bp]) - abs(mean(bp))):.3e}   all_negative={all(v<0 for v in bp)}")

# ---- SHIFT INVARIANCE OF PEARSON ----
print('\n'+'='*76); print('PEARSON SHIFT-INVARIANCE (numerical proof)'); print('='*76)
true=[p['true']+(3.0824 if p['protein']=='2K5H' else 0.0) for p in pp]
for c in ['base','coil_fixed_b']:
    pred=[p['pred_'+c] for p in pp]
    r0=pcc(true,pred)
    for k in [0.0, 1.0, -4.0622, 100.0, -1e3]:
        rk=pcc(true,[v+k for v in pred])
        print(f"  {c:<14} shift {k:>9.4f}  PCC={rk:.15f}  delta={rk-r0:+.3e}")
# the exact debias: remove each condition's own mean bias -> MAE minimised, PCC unchanged
print('\n  De-biasing each condition by its own mean(b_p):')
for c in conds:
    pred=[p['pred_'+c] for p in pp]
    bp=[pr-tr for pr,tr in zip(pred,true)]
    mb=mean(bp)
    pred_db=[v-mb for v in pred]
    bp_db=[pr-tr for pr,tr in zip(pred_db,true)]
    print(f"    {c:<14} MAE {mean([abs(v) for v in bp]):.4f} -> {mean([abs(v) for v in bp_db]):.4f} | "
          f"std(bp) {std(bp):.4f} -> {std(bp_db):.4f} | PCC {pcc(true,pred):.6f} -> {pcc(true,pred_db):.6f}")
