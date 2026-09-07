"""Layer attribution + weight-motion vs val-improvement coupling."""
import json,statistics,math,os,re
D=json.load(open('results/ckpt_dynamics_all.json' if os.path.exists('results/ckpt_dynamics_all.json')
                 else 'results/ckpt_dynamics.json'))
CEN=json.load(open('results/ckpt_param_census.json')) if os.path.exists('results/ckpt_param_census.json') else None
print("runs=%d"%len(D))
if CEN:
    cen=CEN['census']; tot=sum(c['n'] for c in cen.values())
    print("\n=== PARAM CENSUS total=%d ==="%tot)
    for g,c in sorted(cen.items(),key=lambda x:-x[1]['n']):
        print("  %-14s %10d %6.2f%%  eg %s"%(g,c['n'],100*c['n']/tot,c['keys'][0][:60]))

print("\n=== LAYER MOTION: total distance travelled from epoch 0 to last epoch ===")
print("(group_dist_e0 at final epoch, averaged over runs; and as a share of total motion)")
acc={}; accrel={}; nrun=0
for run,rows in D.items():
    if len(rows)<3: continue
    last=rows[-1]
    gd=last.get('group_dist_e0')
    if not gd: continue
    nrun+=1; tot=math.sqrt(sum(v*v for v in gd.values()))
    for g,v in gd.items():
        acc.setdefault(g,[]).append(v)
        accrel.setdefault(g,[]).append((v*v)/(tot*tot) if tot>0 else 0)
print("n runs=%d"%nrun)
print("%-14s %12s %10s"%("group","mean_dist","%% of total motion (var share)"))
rank=sorted(acc,key=lambda g:-statistics.mean(acc[g]))
for g in rank:
    print("  %-14s %12.4f %9.2f%%"%(g,statistics.mean(acc[g]),100*statistics.mean(accrel[g])))

print("\n=== RELATIVE motion (group_reldelta summed over epochs) = how much each layer CHANGES relative to its own size ===")
rel={}
for run,rows in D.items():
    per={}
    for r in rows[1:]:
        for g,v in (r.get('group_reldelta') or {}).items(): per[g]=per.get(g,0)+v
    for g,v in per.items(): rel.setdefault(g,[]).append(v)
for g in sorted(rel,key=lambda g:-statistics.mean(rel[g])):
    print("  %-14s cumulative rel-change=%8.4f  (n=%d runs)"%(g,statistics.mean(rel[g]),len(rel[g])))

print("\n=== DOES WEIGHT MOTION STOP? per-epoch delta, normalized to epoch-1 delta ===")
prof={}
for run,rows in D.items():
    if len(rows)<10: continue
    d=[r.get('delta') for r in rows[1:] if r.get('delta')]
    if len(d)<9: continue
    d0=d[0]
    for i,x in enumerate(d): prof.setdefault(i+1,[]).append(x/d0)
print("%5s %8s %8s %5s"%("epoch","mean","median","n"))
for e in sorted(prof):
    v=prof[e]
    if len(v)>=5: print("%5d %8.4f %8.4f %5d"%(e,statistics.mean(v),statistics.median(v),len(v)))

# ---- couple weight motion to val improvement ----
V=json.load(open('results/ckpt_val_summary_all.json'))
def norm(s):
    s=re.sub(r'^PEM_fine_tuned-trianed_models-light_attentionkf_','',s)
    s=re.sub(r'_\d+\.out$','',s); s=re.sub(r'\.out$','',s)
    return s
vmap={norm(k):v for k,v in V.items()}
print("\n=== WEIGHT MOTION vs VAL IMPROVEMENT, matched runs ===")
pairs=[]
for run,rows in D.items():
    key=norm(run)
    cand=[k for k in vmap if k==key or k.replace('gld_','')==key.replace('gld_','')]
    if not cand: continue
    v=vmap[cand[0]]; p=v['pcc']
    d=[r.get('delta') for r in rows[1:]]
    n=min(len(p)-1,len(d))
    if n<6: continue
    dp=[p[i+1]-p[i] for i in range(n)]; dd=[d[i] for i in range(n)]
    pairs.append((key,n,dd,dp))
print("matched runs: %d"%len(pairs))
try:
    from scipy.stats import pearsonr,spearmanr
    allr=[]
    for key,n,dd,dp in pairs:
        if len(set(dd))<3: continue
        r,pv=spearmanr(dd,dp); allr.append(r)
        print("  %-46s n=%2d spearman(delta_w, delta_valPCC)=%+.3f p=%.3f"%(key[:46],n,r,pv))
    if allr:
        print("\n  mean spearman=%+.4f  median=%+.4f  n=%d runs  ; positive in %d/%d"%
              (statistics.mean(allr),statistics.median(allr),len(allr),
               sum(1 for r in allr if r>0),len(allr)))
        print("  NOTE at n~14 epochs |rho|<0.53 is not significant at p=0.05")
except ImportError: print("scipy missing")
