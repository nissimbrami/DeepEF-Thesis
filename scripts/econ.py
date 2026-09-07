"""Early-stopping economics from the 40 harvested val curves. Honest accounting."""
import json,statistics,re,os,glob
S=json.load(open('results/ckpt_val_summary_all.json'))
full={k:v for k,v in S.items() if v['n']==15}
print("total runs with curves: %d ; reached 15 epochs: %d"%(len(S),len(full)))

# --- wall-clock epoch time, measured from checkpoint mtimes ---
INV='results/ckpt_inventory.json'
eph=None
if os.path.exists(INV):
    inv=json.load(open(INV)); gaps=[]
    for d,v in inv.items():
        m=v['mtimes']
        for a,b in zip(m,m[1:]):
            g=(b-a)/3600.0
            if 0.05<g<6: gaps.append(g)
    if gaps:
        eph=statistics.median(gaps)
        print("epoch wall-clock from %d ckpt mtime gaps: median=%.3f h mean=%.3f h p25=%.3f p75=%.3f"%
              (len(gaps),eph,statistics.mean(gaps),
               statistics.quantiles(gaps,n=4)[0],statistics.quantiles(gaps,n=4)[2]))
if eph is None: eph=0.53; print("fallback epoch_h=0.53")

print("\n=== WHERE IS THE BEST EPOCH? (n=%d full runs) ==="%len(full))
ab=[v['argbest'] for v in full.values()]
print("argbest distribution:", {i:ab.count(i) for i in sorted(set(ab))})
print("median argbest=%.1f mean=%.2f ; argbest>=10 in %d/%d (%.0f%%) ; argbest>=12 in %d/%d"%
      (statistics.median(ab),statistics.mean(ab),sum(1 for a in ab if a>=10),len(ab),
       100*sum(1 for a in ab if a>=10)/len(ab), sum(1 for a in ab if a>=12),len(ab)))
li=[v['lastimp'] for v in full.values()]
print("last >=0.005 improvement epoch: median=%.1f mean=%.2f ; >=10 in %d/%d"%
      (statistics.median(li),statistics.mean(li),sum(1 for a in li if a>=10),len(li)))

print("\n=== ORACLE TRUNCATION: if we had stopped at epoch E, what val-PCC do we lose? ===")
print("%3s %8s %8s %8s %8s %8s"%("E","meanloss","medloss","maxloss","GPUh/run","runs hurt >0.01"))
for E in range(4,15):
    loss=[]
    for k,v in full.items():
        best=v['best']; got=max(v['pcc'][:E+1]); loss.append(best-got)
    saved=(14-E)*eph
    print("%3d %8.4f %8.4f %8.4f %8.2f %8d"%(E,statistics.mean(loss),statistics.median(loss),
          max(loss),saved,sum(1 for l in loss if l>0.01)))

print("\n=== REALISTIC EARLY STOPPING (patience P, min-delta d) ===")
print("%2s %6s %8s %8s %8s %8s %8s"%("P","delta","mean_ep","saved_ep","GPUh/run","meanloss","maxloss"))
best_cfg=None
for P in (2,3,4,5,6):
  for d in (0.0,0.002,0.005):
    need=[];loss=[]
    for k,v in full.items():
        p=v['pcc']; bs=p[0]; bi=0; stop=14
        for i in range(1,15):
            if p[i]>bs+d: bs=p[i]; bi=i
            if i-bi>=P: stop=i; break
        need.append(stop+1); loss.append(v['best']-max(p[:stop+1]))
    me=statistics.mean(need); sv=15-me
    print("%2d %6.3f %8.2f %8.2f %8.2f %8.4f %8.4f"%(P,d,me,sv,sv*eph,statistics.mean(loss),max(loss)))

print("\n=== FINAL-EPOCH PENALTY: is scoring the LAST epoch right? ===")
gap=[v['best']-v['final'] for v in full.values()]
print("n=%d best-final: mean=%.4f median=%.4f max=%.4f ; final IS best in %d/%d runs"%
      (len(gap),statistics.mean(gap),statistics.median(gap),max(gap),
       sum(1 for g in gap if g<1e-9),len(gap)))
allg=[v['best']-v['final'] for v in S.values()]
print("ALL %d runs: mean=%.4f median=%.4f max=%.4f ; final IS best in %d"%
      (len(allg),statistics.mean(allg),statistics.median(allg),max(allg),sum(1 for g in allg if g<1e-9)))
# how much is lost by taking final vs best, in GPU-h-equivalent terms
print("\nTaking best-epoch instead of final costs 0 extra GPU time (checkpoints already saved)")
print("and recovers mean %.4f val PCC (median %.4f) across all %d runs."%
      (statistics.mean(allg),statistics.median(allg),len(allg)))
