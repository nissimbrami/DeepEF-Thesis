"""P8: reconcile the three published headlines against explicit populations."""
import pandas as pd, numpy as np
R = pd.read_csv('/home/nissimb/DeepPEF/results/05_infrastructure/_p8_all_csvs.csv')

def rep(name, sub):
    if len(sub)==0:
        print('%-46s n=0' % name); return
    print('%-46s n=%2d  pooled %.4f +/- %.4f   oracle %.4f +/- %.4f   gain %+.4f'
          % (name, len(sub), sub.pooled.mean(), sub.pooled.std(ddof=1) if len(sub)>1 else 0.0,
             sub.oracle.mean(), sub.oracle.std(ddof=1) if len(sub)>1 else 0.0,
             sub.oracle.mean()-sub.pooled.mean()))

print('=== candidate bases (27 proteins, ddG, 2K5H dropped) ===')
rep('ALL 43 csvs', R)
rep('all non-D1 (D0 only), 35 csvs', R[R.D=='D0'])
rep('orig family, all epochs (10)', R[R.family=='orig'])
rep('orig family, val-selected epoch only', R[(R.family=='orig')])
rep('orig+gld, D0', R[(R.family!='p3')&(R.D=='D0')])
rep('p3 only D0', R[(R.family=='p3')&(R.D=='D0')])
rep('p3 only D1', R[(R.family=='p3')&(R.D=='D1')])
rep('gld only', R[R.family=='gld'])

# published-number reconstruction attempts
print()
print('=== reconstruction of published figures ===')
# 22 mixed csvs -> 0.4899 : likely all p3 (17) + 5 sigma = 22
sub = R[(R.family=='p3')|(R.run.str.startswith('sigma'))]
rep('p3(17)+sigma(5) = 22 mixed', sub)
# 20 non-D1
sub2 = R[R.D=='D0']
print()
# 9 "original population"
orig = R[R.family=='orig'].sort_values('file')
print('orig family files:')
for _,r in orig.iterrows():
    print('   %-40s ep%-3d pooled %.4f oracle %.4f' % (r.file, r.epoch, r.pooled, r.oracle))
rep('orig 10', orig)
for drop in orig.file:
    s = orig[orig.file!=drop]
    if abs(s.pooled.mean()-0.5772)<0.002:
        print('  -> drop %s gives pooled %.4f oracle %.4f' % (drop, s.pooled.mean(), s.oracle.mean()))
