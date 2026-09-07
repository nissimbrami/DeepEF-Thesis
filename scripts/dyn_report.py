"""Read results/ckpt_dynamics.json + val curves -> the two headline questions."""
import json, statistics, math, re, os
D=json.load(open('results/ckpt_dynamics.json'))
V=json.load(open('results/ckpt_val_curves.json'))
# map run dir -> log basename
def tag(run): return run.split('light_attentionkf_')[-1]
logmap={}
for lg in V:
    t=re.sub(r'_\d+\.out$','',lg)
    logmap[t]=lg

EPOCH_H=32.6/60.0
print("=== Q2: WHERE DOES THE WEIGHT MOVEMENT STOP? ===")
print("rel_delta = ||W_e - W_{e-1}|| / ||W_e||   (per-epoch relative movement)")
print()
print("%-40s %s" % ("run","rel_delta per epoch (e1..)"))
alltab={}
for run in sorted(D):
    rows=D[run]
    rd=[(r['epoch'], r.get('rel_delta')) for r in rows if r.get('rel_delta') is not None]
    if len(rd)<5: continue
    alltab[run]=rd
    print("%-40s %s" % (tag(run)[:40], " ".join("%.3f"%x for _,x in rd)))

print("\n--- normalised movement profile (each run scaled by its own epoch-1 movement) ---")
maxE=15
prof={e:[] for e in range(1,maxE)}
for run,rd in alltab.items():
    d=dict(rd); base=d.get(1)
    if not base: continue
    for e,x in rd:
        if e>=1: prof[e].append(x/base)
print("%-6s %6s %6s %6s %4s" % ("epoch","median","mean","p90","n"))
for e in range(1,maxE):
    v=prof[e]
    if not v: continue
    print("%-6d %6.3f %6.3f %6.3f %4d" % (e, statistics.median(v), statistics.mean(v),
          sorted(v)[int(0.9*(len(v)-1))], len(v)))

print("\n--- does movement ever fall below 25%% / 10%% of the epoch-1 rate? ---")
n_lt25=n_lt10=0; tot=0
for run,rd in alltab.items():
    d=dict(rd); base=d.get(1)
    if not base: continue
    tot+=1
    last=rd[-1][1]/base
    if last<0.25: n_lt25+=1
    if last<0.10: n_lt10+=1
print("runs whose FINAL epoch movement < 25%% of epoch-1: %d/%d ; < 10%%: %d/%d" % (n_lt25,tot,n_lt10,tot))

print("\n=== Q3: WHICH LAYER GROUPS MOVE? (relative movement per group, averaged over epochs) ===")
agg={}
for run,rows in D.items():
    for r in rows:
        for g,x in (r.get('group_reldelta') or {}).items():
            agg.setdefault(g,[]).append(x)
print("%-12s %8s %8s %6s" % ("group","mean_rel","median","n"))
for g in sorted(agg, key=lambda g:-statistics.mean(agg[g])):
    v=agg[g]; print("%-12s %8.4f %8.4f %6d" % (g, statistics.mean(v), statistics.median(v), len(v)))

print("\n=== Q3b: share of TOTAL squared movement, by group (mean over all epoch steps) ===")
shares={}
for run,rows in D.items():
    for r in rows:
        gd=r.get('group_delta')
        if not gd: continue
        tot=sum(x*x for x in gd.values())
        if tot<=0: continue
        for g,x in gd.items():
            shares.setdefault(g,[]).append(x*x/tot)
print("%-12s %9s %9s" % ("group","mean_share","median"))
for g in sorted(shares, key=lambda g:-statistics.mean(shares[g])):
    v=shares[g]; print("%-12s %9.4f %9.4f" % (g, statistics.mean(v), statistics.median(v)))

print("\n=== Q2b: weight movement vs val-PCC improvement, same epoch, SAME run ===")
print("%-34s %6s %6s %8s" % ("run","argmaxV","lowest-move-e","corr(reldelta,dPCC)"))
import itertools
cors=[]
for run,rows in D.items():
    t=tag(run); lg=logmap.get(t)
    if not lg: continue
    p=[r['pcc'] for r in V[lg]]
    rd=dict((r['epoch'],r.get('rel_delta')) for r in rows if r.get('rel_delta') is not None)
    xs=[];ys=[]
    for e in sorted(rd):
        if e<len(p) and e>=1:
            xs.append(rd[e]); ys.append(p[e]-p[e-1])
    if len(xs)<5: continue
    mx=statistics.mean(xs); my=statistics.mean(ys)
    num=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    den=math.sqrt(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))
    c=num/den if den>0 else float('nan')
    cors.append(c)
    lowe=min(rd, key=lambda e:rd[e])
    print("%-34s %6d %6d %14.3f" % (t[:34], p.index(max(p)), lowe, c))
if cors:
    print("\nmean corr(per-epoch rel movement, per-epoch val-PCC gain) = %+.3f over %d runs" %
          (statistics.mean(cors), len(cors)))
    print("n=%d runs; |r| threshold for a single run of ~14 points at p=.05 is 0.53" % len(cors))
