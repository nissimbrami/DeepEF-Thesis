"""How much GPU time would early stopping actually save? Uses recorded val curves only."""
import json, statistics, math
C=json.load(open('results/ckpt_val_curves.json'))
EPOCH_H=32.6/60.0

print("EPOCH_H (median wall) = %.3f h\n" % EPOCH_H)
print("%-42s %2s %6s %6s %4s | %s" % ("run","n","best","final","argb","first epoch within tol of best"))
rows=[]
for tol in (0.0,0.005,0.01,0.02):
    pass
detail={}
for k,v in sorted(C.items()):
    p=[r['pcc'] for r in v]
    n=len(p); b=max(p); ab=p.index(b)
    d={}
    for tol in (0.0,0.005,0.01,0.02):
        d[tol]=next(i for i,x in enumerate(p) if x>=b-tol)
    detail[k]=dict(n=n,best=b,final=p[-1],argbest=ab,within=d,pcc=p)
    print("%-42s %2d %6.3f %6.3f %4d | tol0=%2d .005=%2d .01=%2d .02=%2d" %
          (k.split('_2')[0][:42], n, b, p[-1], ab, d[0.0],d[0.005],d[0.01],d[0.02]))

full=[k for k,v in detail.items() if v['n']==15]
print("\n--- RUNS THAT REACHED 15 EPOCHS (n=%d) ---" % len(full))
for tol in (0.0,0.005,0.01,0.02):
    stops=[detail[k]['within'][tol] for k in full]
    # epochs needed = stop+1 (0-indexed); patience p means you run stop+1+p to KNOW
    need=[s+1 for s in stops]
    saved=[15-x for x in need]
    print("tol=%.3f  median stop epoch=%.1f  mean epochs needed=%.2f/15  "
          "mean epochs saved=%.2f (%.0f%%)  GPU-h saved/run=%.2f" %
          (tol, statistics.median(stops), statistics.mean(need), statistics.mean(saved),
           100*statistics.mean(saved)/15, statistics.mean(saved)*EPOCH_H))

print("\n--- WITH PATIENCE (must observe P non-improving epochs before stopping) ---")
for P in (2,3,4,5):
    for tol in (0.005,):
        need=[]
        for k in full:
            p=detail[k]['pcc']; bestsofar=p[0]; bi=0; stop=None
            for i in range(1,15):
                if p[i]>bestsofar+tol: bestsofar=p[i]; bi=i
                if i-bi>=P: stop=i; break
            need.append((stop if stop is not None else 14)+1)
        realized=[]
        for k,nn in zip(full,need):
            p=detail[k]['pcc']
            realized.append(max(p[:nn]))
        lost=[detail[k]['best']-r for k,r in zip(full,realized)]
        print("patience=%d tol=%.3f  mean epochs=%.2f/15  saved=%.2f ep = %.2f GPU-h/run  "
              "mean val-PCC lost=%.4f (max %.4f)" %
              (P, tol, statistics.mean(need), 15-statistics.mean(need),
               (15-statistics.mean(need))*EPOCH_H, statistics.mean(lost), max(lost)))

print("\n--- FINAL-EPOCH PENALTY: is the last epoch the right one to score? ---")
gap=[detail[k]['best']-detail[k]['final'] for k in full]
print("n=%d  best-final val PCC: mean=%.4f median=%.4f max=%.4f; runs where final IS best: %d" %
      (len(full), statistics.mean(gap), statistics.median(gap), max(gap), sum(1 for g in gap if g<1e-9)))
json.dump(detail, open('results/ckpt_val_stop.json','w'), indent=1)
print("\nWROTE results/ckpt_val_stop.json")
